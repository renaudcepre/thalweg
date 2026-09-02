// Real-time log panel: consumes the {"type":"log", ...} messages
// emitted by the tracing bridge on the server side and displays them in #log-body
// (logs region of the sidebar).

const LOG_MAX_LINES = 400;

let logBodyEl = null;
let logCountEl = null;
let logTotal = 0;

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatTs(tsMs) {
  const d = new Date(tsMs);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function formatFields(fields) {
  if (!fields || typeof fields !== "object") return "";
  const keys = Object.keys(fields);
  if (keys.length === 0) return "";
  return keys.map((k) => `${k}=${JSON.stringify(fields[k])}`).join(" ");
}

export function appendLog(msg) {
  if (!logBodyEl) return;
  // Auto-scroll only if already all the way at the bottom: respects user scroll.
  const atBottom =
    logBodyEl.scrollHeight - logBodyEl.scrollTop - logBodyEl.clientHeight < 10;

  const level = msg.level || "INFO";
  const line = document.createElement("div");
  line.className = `log-line log-level-${level}`;
  const fields = formatFields(msg.fields);
  const target = msg.target ? msg.target.replace(/^hexsim_/, "") : "";

  const targetHtml = target ? `<span class="log-target">${escapeHtml(target)}</span>` : "";
  const fieldsHtml = fields ? `<span class="log-fields">${escapeHtml(fields)}</span>` : "";

  line.innerHTML = `<span class="log-ts">${formatTs(msg.ts_ms)}</span><span class="log-level">${level}</span><span class="log-body-col">${targetHtml}<span class="log-msg">${escapeHtml(msg.message || "")}</span>${fieldsHtml}</span>`;

  logBodyEl.appendChild(line);
  logTotal += 1;
  if (logCountEl) logCountEl.textContent = String(logTotal);

  while (logBodyEl.childElementCount > LOG_MAX_LINES) {
    logBodyEl.removeChild(logBodyEl.firstChild);
  }

  if (atBottom) {
    logBodyEl.scrollTop = logBodyEl.scrollHeight;
  }
}

export function initLogPanel() {
  logBodyEl = document.getElementById("log-body");
  logCountEl = document.getElementById("log-count");

  document.getElementById("log-clear")?.addEventListener("click", () => {
    if (!logBodyEl) return;
    logBodyEl.innerHTML = "";
    logTotal = 0;
    if (logCountEl) logCountEl.textContent = "0";
  });
}
