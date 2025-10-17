// loading overlay
const uploadForm = document.getElementById('uploadForm');
const overlay = document.getElementById('loadingOverlay');
if (uploadForm && overlay) {
  uploadForm.addEventListener('submit', () => {
    overlay.style.display = 'flex';
  });
}

// live logs (both pages)
const logsBox = document.getElementById('logsBox');
const toggleBtn = document.getElementById('toggleLogs');
let logsPaused = false;

if (toggleBtn) {
  toggleBtn.addEventListener('click', () => {
    logsPaused = !logsPaused;
    toggleBtn.textContent = logsPaused ? 'Resume' : 'Pause';
  });
}

async function pollLogs() {
  if (!logsBox) return;
  try {
    const r = await fetch('/logs');
    const data = await r.json();
    if (!logsPaused) {
      logsBox.textContent = data.lines.join('\n');
      logsBox.scrollTop = logsBox.scrollHeight;
    }
  } catch (e) {
    // ignore polling errors
  } finally {
    setTimeout(pollLogs, 1500);
  }
}
pollLogs();

// Report buttons on results page
document.addEventListener('submit', async (e) => {
  const form = e.target;
  if (form.classList.contains('reportForm')) {
    e.preventDefault();
    const fd = new FormData(form);
    const res = await fetch('/report', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      if (data.mode === 'mailto' && data.mailto) {
        window.location.href = data.mailto;
      } else {
        alert(data.message || 'Report prepared.');
      }
    } else {
      alert('Report failed.');
    }
  }
});
