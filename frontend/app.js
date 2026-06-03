const messages = document.querySelector("#messages");
const form = document.querySelector("#form");
const input = document.querySelector("#message");
const status = document.querySelector("#status");

async function updateStatus() {
  const data = await fetch("/api/status").then((r) => r.json()).catch(() => null);
  if (data) status.textContent = data.loaded ? data.device : data.loading ? "Loading" : "Idle";
}

function add(role, text) {
  const node = document.createElement("article");
  node.className = role;
  node.textContent = text;
  messages.append(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  add("user", message);
  status.textContent = "Generating";

  const pending = add("assistant", "");
  const thinkingEl = document.createElement("details");
  thinkingEl.open = true;
  thinkingEl.innerHTML = "<summary>Thinking</summary><div></div>";
  const thinkingDiv = thinkingEl.querySelector("div");
  pending.append(thinkingEl);
  const responseEl = document.createElement("div");
  pending.append(responseEl);

  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = JSON.parse(line.slice(6));
      if (data.error) {
        responseEl.textContent = data.error;
        thinkingEl.remove();
        status.textContent = "Error";
        return;
      }
      if (data.done) {
        thinkingDiv.textContent = data.thinking || "";
        responseEl.textContent = data.response || "";
        thinkingEl.open = !data.response;
        if (!data.thinking) thinkingEl.remove();
        status.textContent = "Ready";
        updateStatus();
        return;
      }
      thinkingDiv.textContent = data.thinking || "";
      responseEl.textContent = data.response || "";
      if (!data.thinking && data.response) thinkingEl.remove();
      messages.scrollTop = messages.scrollHeight;
    }
  }
});

updateStatus();
setInterval(updateStatus, 2000);
