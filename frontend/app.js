const $ = (s) => document.querySelector(s);
const messages = $("#messages");
const form = $("#form");
const input = $("#message");
const status = $("#status");
const sendBtn = $("#send-btn");
const welcome = $("#welcome");
const settingsPanel = $("#settings-panel");

let history = [];
let generating = false;

const escapeHtml = (text) => text
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const renderMarkdown = (text) => {
  const source = protectMath(text || "");
  if (window.marked) return marked.parse(source);
  return escapeHtml(source)
    .split(/\n{2,}/)
    .map((part) => `<p>${part.replace(/\n/g, "<br>")}</p>`)
    .join("");
};

if (window.marked) marked.setOptions({ breaks: true, gfm: true });

const renderMath = (tex, displayMode = false) => {
  if (window.katex) {
    try {
      return katex.renderToString(tex, {
        displayMode,
        throwOnError: false,
        strict: "ignore",
      });
    } catch {
      return escapeHtml(tex);
    }
  }
  const className = displayMode ? "math-display" : "math-inline";
  return `<span class="${className}">${escapeHtml(tex)}</span>`;
};

function protectMath(text) {
  const codeParts = [];
  const stashCode = (match) => {
    const token = `@@INFENG_CODE_${codeParts.length}@@`;
    codeParts.push(match);
    return token;
  };

  let protectedText = text
    .replace(/```[\s\S]*?```/g, stashCode)
    .replace(/`[^`\n]+`/g, stashCode);

  protectedText = protectedText.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => (
    renderMath(tex.trim(), true)
  ));
  protectedText = protectedText.replace(/(^|[^\\$])\$([^\n$]+?)\$(?!\$)/g, (_, prefix, tex) => (
    `${prefix}${renderMath(tex.trim(), false)}`
  ));

  return protectedText.replace(/@@INFENG_CODE_(\d+)@@/g, (_, idx) => codeParts[idx]);
}

function enhanceMessage(container) {
  container.querySelectorAll("pre code").forEach((block) => {
    if (window.hljs) hljs.highlightElement(block);
  });
}

$("#settings-btn").onclick = () => settingsPanel.classList.toggle("open");
$("#settings-close").onclick = () => settingsPanel.classList.remove("open");
$("#new-chat-btn").onclick = () => { history = []; messages.innerHTML = ""; messages.append(welcome); welcome.style.display = ""; };
$("#load-btn").onclick = async () => {
  const btn = $("#load-btn");
  btn.disabled = true;
  status.textContent = "Loading...";
  await fetch("/api/load", { method: "POST" });
  const poll = setInterval(async () => {
    const data = await fetch("/api/status").then(r => r.json()).catch(() => null);
    if (data?.loaded) { status.textContent = data.device; btn.disabled = false; clearInterval(poll); }
    else if (!data?.loading) { status.textContent = ""; btn.disabled = false; clearInterval(poll); }
  }, 1000);
};

document.querySelectorAll('.slider-row input[type="range"]').forEach(el => {
  el.oninput = () => $(`#${el.id}-val`).textContent = el.value;
});

input.oninput = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; };
input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } };

async function pollStatus() {
  const data = await fetch("/api/status").then(r => r.json()).catch(() => null);
  if (data) status.textContent = data.loaded ? data.device : data.loading ? "Loading..." : "";
}

function addMsg(role, content) {
  if (welcome.parentNode) welcome.style.display = "none";
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const body = document.createElement("div");
  body.className = "msg-content";
  if (role === "user") { body.textContent = content; el.append(body); }
  else {
    body.innerHTML = renderMarkdown(content || "");
    enhanceMessage(body);
    el.append(body);
  }
  messages.append(el);
  messages.scrollTop = messages.scrollHeight;
  return body;
}

function getParams() {
  return {
    thinking: $("#thinking").checked,
    temperature: parseFloat($("#temperature").value),
    top_p: parseFloat($("#top_p").value),
    top_k: parseInt($("#top_k").value),
  };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || generating) return;
  input.value = ""; input.style.height = "auto";
  generating = true; sendBtn.disabled = true;

  history.push({ role: "user", content: text });
  addMsg("user", text);

  const params = getParams();
  const body = addMsg("assistant", "");
  let thinkDiv = null, thinkWrap = null;
  const respSpan = document.createElement("div");
  respSpan.className = "response-content";
  respSpan.textContent = "Thinking...";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  body.append(respSpan, cursor);

  if (params.thinking) {
    thinkWrap = document.createElement("details");
    thinkWrap.open = true;
    thinkWrap.innerHTML = "<summary>Thought for a few seconds</summary><div></div>";
    thinkDiv = thinkWrap.querySelector("div");
    body.prepend(thinkWrap);
  }

  let buf = "", thinking = "", response = "", finished = false;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history, ...params }),
    });
    if (!resp.ok || !resp.body) throw new Error(`Chat request failed (${resp.status})`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = JSON.parse(line.slice(6));
        if (data.error) throw new Error(data.error);
        if (data.status && !response) { respSpan.textContent = data.status; continue; }
        if (data.done) { thinking = data.thinking || ""; response = data.response || ""; finished = true; break; }
        thinking = data.thinking || thinking;
        response = data.response || response;
        if (thinkDiv) thinkDiv.textContent = thinking;
        respSpan.innerHTML = response ? renderMarkdown(response) : "Thinking...";
        enhanceMessage(respSpan);
        messages.scrollTop = messages.scrollHeight;
      }
    }
  } catch (err) {
    cursor.remove();
    respSpan.textContent = `Error: ${err.message}`;
    respSpan.classList.add("error");
    if (thinkWrap) thinkWrap.remove();
    generating = false; sendBtn.disabled = false;
    pollStatus();
    return;
  }

  cursor.remove();
  respSpan.innerHTML = response ? renderMarkdown(response) : "No response generated.";
  enhanceMessage(respSpan);
  if (thinkDiv) {
    thinkDiv.textContent = thinking;
    if (!thinking) thinkWrap.remove();
    else { thinkWrap.open = false; thinkWrap.querySelector("summary").textContent = "Thought for a few seconds"; }
  }

  history.push({ role: "assistant", content: response });
  generating = false; sendBtn.disabled = false;
  pollStatus();
});

pollStatus();
setInterval(pollStatus, 3000);
