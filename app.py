import os, io, re, asyncio, tempfile, logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from telethon import TelegramClient
from telethon.errors import RPCError
from PIL import Image
import pandas as pd
import aiohttp
import cv2
import numpy as np

# ---------- Setup ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
if os.name == "nt" and "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.getcwd())

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("CUSTOM_SEARCH_CX", "")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SIM_THRESHOLD_PHOTO = float(os.getenv("SIM_THRESHOLD_PHOTO", "0.67"))
SIM_THRESHOLD_FRAME = float(os.getenv("SIM_THRESHOLD_FRAME", "0.70"))
FRAME_STRIDE_SEC = int(float(os.getenv("FRAME_STRIDE_SEC", "5")))

app = FastAPI(title="Piracy Scanner", version="3.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Globals ----------
model = SentenceTransformer("clip-ViT-B-32")
LIVE_LOGS: List[str] = []
KEYWORDS, KEYPHRASES = [], []
LAST_RESULTS: List[Dict[str, Any]] = []
reference_logo_embedding = None
reference_logo_name = None

def push_log(msg: str):
    LIVE_LOGS.append(msg)
    if len(LIVE_LOGS) > 400:
        del LIVE_LOGS[: len(LIVE_LOGS) - 400]
    logging.info(msg)

# ---------- Keyword loader ----------
def load_csv_keywords():
    global KEYWORDS, KEYPHRASES
    KEYWORDS, KEYPHRASES = [], []
    csvs = ["videos_orthopedics.csv", "videos_radiology.csv"]
    for path in csvs:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            for col in ["title", "description"]:
                if col in df.columns:
                    for val in df[col].fillna("").astype(str):
                        v = val.strip().lower()
                        if not v:
                            continue
                        KEYPHRASES.append(v)
                        for t in re.findall(r"[a-z0-9]+", v):
                            if len(t) > 2:
                                KEYWORDS.append(t)
        except Exception as e:
            push_log(f"CSV load error for {path}: {e}")
    KEYWORDS = list(dict.fromkeys(KEYWORDS))[:500]
    KEYPHRASES = list(dict.fromkeys(KEYPHRASES))[:500]
    push_log(f"📚 Loaded {len(KEYWORDS)} tokens, {len(KEYPHRASES)} phrases")

# ---------- Embedding helpers ----------
def compute_embedding(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return model.encode(img, convert_to_tensor=True)

def similarity_score(a, b):
    return float(util.cos_sim(a, b).cpu().numpy()[0][0])

def kw_score(text: str):
    if not text:
        return 0.0
    text_l = text.lower()
    ph = sum(1 for p in KEYPHRASES if p in text_l)
    tk = sum(1 for k in KEYWORDS if k in text_l.split())
    return min(1.0, ph * 0.5 + tk / 20.0)

# ---------- Google Search ----------
async def fetch_image_embedding(url: str):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=15) as r:
                if r.status == 200:
                    b = await r.read()
                    Image.open(io.BytesIO(b)).verify()
                    return compute_embedding(b)
    except Exception as e:
        push_log(f"Google image fetch error: {e}")
    return None

async def search_google(ref_emb):
    results = []
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        push_log("⚠️ Google API not configured")
        return results
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    queries = [" OR ".join(KEYWORDS[:10])] or ["education video"]
    push_log("🔍 Google Image Search started...")
    for q in queries:
        try:
            res = service.cse().list(q=q, cx=GOOGLE_CX, searchType="image", num=8).execute()
            for item in res.get("items", []):
                url = item.get("link", "")
                title = item.get("title", "")
                emb = await fetch_image_embedding(url)
                if not emb:
                    continue
                sim = similarity_score(ref_emb, emb)
                kw = kw_score(title)
                if sim >= SIM_THRESHOLD_PHOTO or kw > 0.7:
                    results.append({
                        "source": "Google",
                        "title": title,
                        "link": url,
                        "similarity": sim,
                        "kw_score": kw,
                        "score": max(sim, kw)
                    })
                    push_log(f"✅ Google: {title[:60]} sim={sim:.2f} kw={kw:.2f}")
        except Exception as e:
            push_log(f"Google error: {e}")
    return results

# ---------- Telegram ----------
async def scan_telegram(ref_emb):
    results = []
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        push_log("⚠️ Telegram not configured")
        return results
    client = TelegramClient("anon", TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()
    push_log("📡 Telegram connected")

    query = " ".join(KEYWORDS[:3]) or "education video"
    push_log(f"🌐 Telegram global search: {query}")
    try:
        async for msg in client.iter_messages(None, search=query, limit=50):
            await process_tg_msg(msg, ref_emb, results)
    except Exception as e:
        push_log(f"TG global search error: {e}")

    # Scan your accessible channels too
    async for dialog in client.iter_dialogs(limit=50):
        try:
            async for msg in client.iter_messages(dialog.entity, limit=40):
                await process_tg_msg(msg, ref_emb, results, getattr(dialog.entity, 'title', 'Unknown'))
        except RPCError:
            continue

    await client.disconnect()
    push_log("🔌 Telegram disconnected")
    return results

async def process_tg_msg(msg, ref_emb, results, channel=""):
    txt = getattr(msg, "message", "") or ""
    kw = kw_score(txt)
    link = f"https://t.me/{getattr(msg.peer_id, 'channel_id', '')}/{msg.id}"

    # --- Photo ---
    if getattr(msg, "photo", None):
        try:
            b = await msg.download_media(file=bytes)
            if not b:
                return
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                push_log(f"⚠️ TG {channel}: unsupported photo format for msg {msg.id}")
                return
            emb = model.encode(img, convert_to_tensor=True)
            sim = similarity_score(ref_emb, emb)
            if sim >= SIM_THRESHOLD_PHOTO or kw > 0.7:
                results.append({
                    "source": "Telegram",
                    "channel": channel,
                    "link": link,
                    "similarity": sim,
                    "kw_score": kw,
                    "score": max(sim, kw)
                })
                push_log(f"✅ TG photo {channel} sim={sim:.2f} kw={kw:.2f}")
        except Exception as e:
            push_log(f"❌ TG photo error {channel}: {e}")

    # --- Video ---
    elif getattr(msg, "video", None):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            await msg.download_media(file=tmp.name)
            cap = cv2.VideoCapture(tmp.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            stride = max(1, int(fps * FRAME_STRIDE_SEC))
            idx = 0
            best_sim = 0.0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        emb = compute_embedding(buf.tobytes())
                        sim = similarity_score(ref_emb, emb)
                        best_sim = max(best_sim, sim)
                        if os.getenv("VERBOSE_VIDEO_LOGS") == "true":
                            push_log(f"   frame {idx//fps}s → sim={sim:.3f}")
                idx += 1
            cap.release()
            os.remove(tmp.name)
            if best_sim >= SIM_THRESHOLD_FRAME or kw > 0.7:
                results.append({
                    "source": "Telegram",
                    "channel": channel,
                    "link": link,
                    "similarity": best_sim,
                    "kw_score": kw,
                    "score": max(best_sim, kw)
                })
                push_log(f"✅ TG video {channel} sim={best_sim:.2f} kw={kw:.2f}")
        except Exception as e:
            push_log(f"❌ TG video error {channel}: {e}")

    # --- Text-only message ---
    elif txt and kw >= 0.85:
        results.append({
            "source": "Telegram",
            "channel": channel,
            "link": link,
            "similarity": 0,
            "kw_score": kw,
            "score": kw
        })
        push_log(f"✅ TG text {channel} kw={kw:.2f}")


# ---------- Scan orchestrator ----------
async def run_scan(logo_bytes: bytes):
    global reference_logo_embedding
    reference_logo_embedding = compute_embedding(logo_bytes)
    push_log("📌 Logo embedded. Running searches...")
    g_task = search_google(reference_logo_embedding)
    t_task = scan_telegram(reference_logo_embedding)
    all_r = await asyncio.gather(g_task, t_task)
    merged = [i for sub in all_r for i in sub]
    merged.sort(key=lambda x: x["score"], reverse=True)
    push_log(f"✅ Scan complete: {len(merged)} hits.")
    return merged

# ---------- Routes ----------
@app.on_event("startup")
async def startup():
    load_csv_keywords()
    push_log("✅ System ready.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/logs", response_class=HTMLResponse)
async def get_logs():
    return HTMLResponse("<br/>".join(LIVE_LOGS[-200:]))

@app.post("/upload_logo")
async def upload_logo(file: UploadFile = File(...)):
    global LAST_RESULTS, reference_logo_name
    content = await file.read()
    try:
        Image.open(io.BytesIO(content)).verify()
    except Exception:
        return HTMLResponse("<h3>❌ Invalid image</h3>")
    reference_logo_name = file.filename
    LAST_RESULTS = await run_scan(content)
    return RedirectResponse("/results", status_code=303)

@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request, "results": LAST_RESULTS})

@app.get("/report", response_class=HTMLResponse)
async def report(request: Request, platform: str, url: str):
    platform = platform.lower()
    note = "Use in-app Report (Telegram → … → Report)" if platform == "telegram" else \
           "Use this <a href='https://support.google.com/legal/troubleshooter/1114905'>Google form</a>."
    html = f"""
    <html><body style='font-family:Arial;padding:2rem'>
      <h2>Report Violation</h2>
      <p>Platform: {platform.title()}</p>
      <p>URL: <a href="{url}" target="_blank">{url}</a></p>
      <p>{note}</p>
      <pre>
We are rights holders for educational content identified by our logo.
This URL hosts our material without authorization.
URL: {url}
Logo File: {reference_logo_name}
      </pre>
      <a href="/results">← Back</a>
    </body></html>
    """
    return HTMLResponse(html)
