const history = [];

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

function appendUser(text) {
  const wrap = el("div", "bubble user");
  wrap.textContent = text;
  document.getElementById("messages").appendChild(wrap);
  scrollBottom();
}

function appendBot(content, routeLabel, image, isError) {
  const wrap = el("div", "bubble bot" + (isError ? " error" : ""));
  const meta = el("span", "meta", routeLabel || "");
  wrap.appendChild(meta);
  const body = el("div", "", typeof marked !== "undefined" ? marked.parse(content) : content);
  wrap.appendChild(body);
  if (image && image.base64) {
    const img = el("img", "gen");
    img.alt = "Imagem gerada";
    img.src = `data:${image.mime};base64,${image.base64}`;
    wrap.appendChild(img);
  }
  document.getElementById("messages").appendChild(wrap);
  scrollBottom();
}

function scrollBottom() {
  const m = document.getElementById("messages");
  m.scrollTop = m.scrollHeight;
}

function setRouteUI(routeLabel) {
  document.getElementById("lastRoute").textContent = routeLabel || "—";
  document.getElementById("routeBadge").textContent = routeLabel || "🤖 LLM — Ollama";
}

function setLoading(on) {
  const bar = document.getElementById("loadingBar");
  const form = document.getElementById("form");
  const ta = document.getElementById("msg");
  const sendBtn = document.getElementById("sendBtn");
  const clearBtn = document.getElementById("clear");
  if (!bar || !form) return;
  bar.hidden = !on;
  bar.setAttribute("aria-busy", on ? "true" : "false");
  form.classList.toggle("is-busy", on);
  ta.disabled = on;
  sendBtn.disabled = on;
  clearBtn.disabled = on;
}

async function sendMessage(text) {
  const payload = {
    message: text,
    history: history.map(({ role, content }) => ({ role, content })),
  };
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Falha na rede: " + res.status);
  }
  return res.json();
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ta = document.getElementById("msg");
  const text = ta.value.trim();
  if (!text) return;
  ta.value = "";
  appendUser(text);
  setLoading(true);
  try {
    const data = await sendMessage(text);
    setRouteUI(data.route_label);
    appendBot(data.content, data.route_label, data.image, data.error);
    history.push({ role: "user", content: text });
    history.push({ role: "assistant", content: data.content });
  } catch (err) {
    appendBot(String(err), "Erro", null, true);
  } finally {
    setLoading(false);
  }
});

document.getElementById("clear").addEventListener("click", () => {
  history.length = 0;
  document.getElementById("messages").innerHTML = "";
  document.getElementById("lastRoute").textContent = "—";
  document.getElementById("routeBadge").textContent = "🤖 LLM — Ollama";
});

document.getElementById("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("form").requestSubmit();
  }
});
