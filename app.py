import os, io, asyncio, tempfile, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
import cv2
import aiohttp
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from telethon import TelegramClient

# ========================
# LOGGING SETUP
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ========================
# LOAD ENVIRONMENT
# ========================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("CUSTOM_SEARCH_CX")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

# ========================
# APP INIT
# ========================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
model = SentenceTransformer("clip-ViT-B-32")
reference_logo_embedding = None
last_results = []

# ========================
# EMBEDDING HELPERS
# ========================
def compute_embedding(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return model.encode(img, convert_to_tensor=True)

def similarity_score(emb1, emb2):
    return float(util.cos_sim(emb1, emb2).cpu().numpy()[0][0])

async def fetch_image_embedding(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    return compute_embedding(img_bytes)
    except Exception as e:
        logging.error(f"Google fetch error: {e}")
        return None
    return None

# ========================
# GOOGLE IMAGE SEARCH
# ========================
async def search_google(ref_emb):
    results = []
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("⚠️ Google API not configured; skipping search.")
        return results
    try:
        logging.info("🔎 Starting Google Image Search...")
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(
            q="education video logo", cx=GOOGLE_CX,
            searchType="image", num=5
        ).execute()
        for item in res.get("items", []):
            emb = await fetch_image_embedding(item["link"])
            if emb is not None:
                sim = similarity_score(ref_emb, emb)
                logging.info(f"Google image '{item['title']}' sim={sim:.3f}")
                if sim > 0.65:
                    results.append({
                        "source": "Google",
                        "title": item["title"],
                        "link": item["link"],
                        "thumbnail": item["image"].get("thumbnailLink", ""),
                        "similarity": sim
                    })
        logging.info(f"✅ Google search complete ({len(results)} matches).")
    except Exception as e:
        logging.error(f"Google search error: {e}")
    return results

# ========================
# TELEGRAM SEARCH (FULL PUBLIC + JOINED)
# ========================
async def search_telegram(ref_emb):
    results = []
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        logging.warning("⚠️ Telegram credentials not configured.")
        return results

    client = TelegramClient("anon", TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()
    logging.info("✅ Telegram client started")

    try:
        dialogs = await client.get_dialogs()
        logging.info(f"📂 Found {len(dialogs)} channels/groups to scan")

        for dialog in dialogs:
            if not getattr(dialog.entity, "megagroup", True):  # Skip tiny private chats
                continue

            channel_name = getattr(dialog.entity, "title", "Unknown Channel")
            channel_user = getattr(dialog.entity, "username", None)
            logging.info(f"🔍 Scanning: {channel_name} ({channel_user})")

            async for msg in client.iter_messages(dialog.entity, limit=20):
                if msg.photo:
                    img_bytes = await msg.download_media(file=bytes)
                    emb = compute_embedding(img_bytes)
                    sim = similarity_score(ref_emb, emb)
                    if sim > 0.65:
                        results.append({
                            "source": "Telegram",
                            "channel": channel_name,
                            "link": f"https://t.me/{channel_user}/{msg.id}" if channel_user else "Private Chat",
                            "similarity": sim
                        })

                elif msg.video:
                    video_bytes = await msg.download_media(file=bytes)
                    if not video_bytes:
                        continue
                    tmp_path = tempfile.mktemp(suffix=".mp4")
                    with open(tmp_path, "wb") as f:
                        f.write(video_bytes)

                    cap = cv2.VideoCapture(tmp_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
                    step = fps * 5
                    idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % step == 0:
                            _, buf = cv2.imencode(".jpg", frame)
                            emb = compute_embedding(buf.tobytes())
                            sim = similarity_score(ref_emb, emb)
                            if sim > 0.65:
                                results.append({
                                    "source": "Telegram",
                                    "channel": channel_name,
                                    "link": f"https://t.me/{channel_user}/{msg.id}" if channel_user else "Private Chat",
                                    "similarity": sim,
                                    "timestamp": f"{idx//fps}s"
                                })
                        idx += 1

                    cap.release()
                    os.remove(tmp_path)

        logging.info(f"📊 Telegram scan complete. Found {len(results)} matches.")
    except Exception as e:
        logging.error(f"Telegram search error: {e}")
    finally:
        await client.disconnect()
        logging.info("🔌 Telegram client disconnected")

    return results


# ========================
# FRONTEND (UI)
# ========================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎓 Education Logo Piracy Scanner</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f9fafc; padding: 40px; font-family: Arial, sans-serif; }
            .container { max-width: 700px; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); padding: 30px; }
            .btn-scan { background-color: #4b7bec; color: white; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container text-center">
            <h2>🎓 Education Logo Piracy Scanner</h2>
            <p class="text-muted">Upload your brand logo and scan across Telegram & Google for misuse.</p>
            <form action="/upload_logo" method="post" enctype="multipart/form-data">
                <input type="file" class="form-control mb-3" name="file" accept="image/*" required>
                <button type="submit" class="btn btn-scan w-100">🚀 Upload & Scan</button>
            </form>
        </div>
    </body>
    </html>
    """

# ========================
# UPLOAD & SCAN
# ========================
@app.post("/upload_logo")
async def upload_logo(file: UploadFile = File(...)):
    global reference_logo_embedding, last_results
    contents = await file.read()
    reference_logo_embedding = compute_embedding(contents)
    logging.info("📌 Logo uploaded & embedding computed")

    results = await asyncio.gather(
        search_google(reference_logo_embedding),
        search_telegram(reference_logo_embedding)
    )
    last_results = [r for sub in results for r in sub]
    logging.info(f"✅ Scan complete ({len(last_results)} matches found).")
    return RedirectResponse(url="/results", status_code=303)

# ========================
# RESULTS PAGE
# ========================
@app.get("/results", response_class=HTMLResponse)
async def results_page():
    if not last_results:
        return "<html><body><h3 style='color:#999;'>No piracy matches found.</h3></body></html>"

    rows = ""
    for i, r in enumerate(last_results, 1):
        link = f"<a href='{r['link']}' target='_blank'>Open</a>" if r['link'] != 'N/A' else "N/A"
        desc = r.get('title', r.get('channel', 'N/A'))
        rows += f"""
        <tr>
            <td>{i}</td>
            <td>{r['source']}</td>
            <td>{desc}</td>
            <td>{r['similarity']*100:.1f}%</td>
            <td>{link}</td>
        </tr>
        """

    return f"""
    <html>
    <head>
        <title>Scan Results</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body style="background:#f9fafc; padding:30px;">
        <div class="container bg-white p-4 rounded shadow">
            <h3 class="mb-4 text-center text-primary">Scan Results</h3>
            <table class="table table-striped table-hover">
                <thead class="table-dark">
                    <tr><th>Sr.</th><th>Source</th><th>Description</th><th>Similarity</th><th>Link</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            <div class="text-center mt-4">
                <a href="/" class="btn btn-outline-primary">🔄 New Scan</a>
            </div>
        </div>
    </body>
    </html>
    """
