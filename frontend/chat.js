(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const currentSessionStorageKey = "hesm.currentChatSession";
  const state = {
    history: [], running: false, result: null, timer: null,
    stateKey: localStorage.getItem(currentSessionStorageKey) || "web_chat",
  };
  const elements = {
    health: $("#health"), form: $("#chat-form"), input: $("#message"),
    send: $("#send"), messages: $("#messages"), empty: $("#trace-empty"),
    trace: $("#trace"), total: $("#total-time"), topic: $("#topic-result"),
    retrieval: $("#retrieval-result"), promptPreview: $("#prompt-preview"),
    promptText: $("#prompt-text"), answerPreview: $("#answer-preview"),
    store: $("#store-result"), model: $("#model-name"),
    dialog: $("#prompt-dialog"), toast: $("#toast"),
    sessionList: $("#session-list"), sessionTitle: $("#current-session-title"),
  };

  function milliseconds(value) {
    const number = Number(value || 0);
    return number >= 1000 ? `${(number / 1000).toFixed(2)} s` : `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 1800);
  }

  function setHealth(ok, label) {
    elements.health.className = `health ${ok ? "ready" : "error"}`;
    elements.health.querySelector("span").textContent = label;
  }

  async function request(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  async function checkHealth() {
    try {
      const data = await request("/api/health");
      setHealth(data.status !== "degraded", data.status === "ready" ? "对话服务就绪" : "记忆库为空");
    } catch (_) {
      setHealth(false, "请启动 WebServer");
    }
  }

  function avatar(role) {
    return role === "user" ? '<span class="user-avatar">我</span>' : '<span class="assistant-avatar">H</span>';
  }

  function addMessage(role, content, meta = "", id = "") {
    const node = document.createElement("article");
    node.className = `message ${role}`;
    if (id) node.id = id;
    node.innerHTML = `${avatar(role)}<div><strong>${role === "user" ? "你" : "HESM Assistant"}</strong><p class="message-content">${escapeHtml(content)}</p>${meta ? `<div class="message-meta">${meta}</div>` : ""}</div>`;
    elements.messages.appendChild(node);
    elements.messages.scrollTop = elements.messages.scrollHeight;
    return node;
  }

  function resetMessages(copy) {
    elements.messages.innerHTML = `<div class="welcome-message"><span class="assistant-avatar">H</span><div><strong>HESM Assistant</strong><p>${escapeHtml(copy)}</p></div></div>`;
  }

  function addTyping() {
    const node = addMessage("assistant", "", "", "typing");
    node.querySelector(".message-content").className = "message-content typing";
    node.querySelector(".message-content").innerHTML = "<i></i><i></i><i></i>";
    return node;
  }

  function setStep(step, status) {
    const order = ["extract", "retrieve", "prompt", "generate", "store"];
    const index = order.indexOf(step);
    document.querySelectorAll(".trace-step").forEach((node, position) => {
      node.classList.toggle("active", status !== "failed" && position === index);
      node.classList.toggle("done", status === "done" || position < index);
      node.classList.toggle("failed", status === "failed" && position === index);
    });
  }

  function startTrace() {
    elements.empty.hidden = true;
    elements.trace.hidden = false;
    elements.total.textContent = "处理中";
    elements.topic.innerHTML = "<p>正在分析主题与意图…</p>";
    elements.retrieval.innerHTML = "<p>等待主题提取完成…</p>";
    elements.promptPreview.textContent = "等待记忆检索完成…";
    elements.answerPreview.textContent = "等待 Prompt 组装完成…";
    elements.store.innerHTML = "<p>等待模型回答完成…</p>";
    document.querySelectorAll(".trace-step time").forEach((item) => { item.textContent = "—"; });
    setStep("extract");
    let index = 0;
    const stages = ["extract", "retrieve", "prompt", "generate"];
    clearInterval(state.timer);
    state.timer = setInterval(() => {
      index = Math.min(index + 1, stages.length - 1);
      setStep(stages[index]);
    }, 850);
  }

  function renderTrace(result) {
    clearInterval(state.timer);
    const timing = result.timing || {};
    const extraction = result.extraction || {};
    const retrieval = result.retrieval || {};
    const stored = result.stored?.memories?.[0] || {};
    elements.total.textContent = milliseconds(timing.chat_total_ms);
    elements.topic.innerHTML = `<div class="topic-grid"><div><small>主题</small><strong>${escapeHtml(extraction.topic || "—")}</strong></div><div><small>核心实体</small><strong>${escapeHtml(extraction.core_entity || "—")}</strong></div><div><small>意图</small><strong>${escapeHtml(extraction.intent || "—")}</strong></div><div><small>置信度</small><strong>${Math.round(Number(extraction.confidence || 0) * 100)}%</strong></div><div><small>相关实体</small><strong>${escapeHtml((extraction.entities || []).join("、") || "—")}</strong></div></div>`;
    elements.retrieval.innerHTML = `<div class="retrieval-counts"><span><b>${retrieval.experiences?.length || 0}</b>Experience</span><span><b>${retrieval.segments?.length || 0}</b>Segment</span><span><b>${retrieval.qas?.length || 0}</b>QA</span></div>`;
    elements.promptPreview.textContent = `已拼接主题结果、${retrieval.qas?.length || 0} 条 QA 证据、${result.history?.length || 0} 条会话历史与当前问题。`;
    elements.promptText.textContent = result.prompt || "";
    elements.answerPreview.textContent = result.answer || "—";
    elements.model.textContent = result.model || "Answer model";
    elements.store.innerHTML = `<div class="store-path"><code title="${escapeHtml(stored.experience_id)}">${escapeHtml(stored.experience_id || "—")}</code><i>→</i><code title="${escapeHtml(stored.segment_id)}">${escapeHtml(stored.segment_id || "—")}</code><i>→</i><code title="${escapeHtml(stored.qa_id)}">${escapeHtml(stored.qa_id || "—")}</code></div>`;
    const values = {
      extract: timing.topic_extraction_ms, retrieve: timing.retrieval_ms,
      prompt: timing.prompt_assembly_ms, generate: timing.generation_ms,
      store: timing.storage_ms,
    };
    Object.entries(values).forEach(([key, value]) => {
      document.querySelector(`[data-step="${key}"] time`).textContent = milliseconds(value);
    });
    setStep("store", "done");
  }

  function cacheCurrentHistory() {
    localStorage.setItem(currentSessionStorageKey, state.stateKey);
    localStorage.setItem(`hesm.chat.${state.stateKey}`, JSON.stringify(state.history));
  }

  async function loadSessions() {
    try {
      const sessions = (await request("/api/sessions")).items || [];
      elements.sessionList.innerHTML = sessions.length ? sessions.map((item) => `<button class="session-item ${item.session_id === state.stateKey ? "active" : ""}" data-session-id="${escapeHtml(item.session_id)}"><strong>${escapeHtml(item.title || "未命名会话")}</strong><span>${escapeHtml(item.preview || "暂无消息")}</span><small>${item.turn_count || 0} 轮 · ${escapeHtml((item.updated_at || "").replace("T", " ").slice(0, 16))}</small><i class="session-delete" data-delete-session="${escapeHtml(item.session_id)}" title="归档会话">×</i></button>`).join("") : '<p class="session-loading">暂无历史会话，点击 ＋ 新建。</p>';
    } catch (error) {
      elements.sessionList.innerHTML = `<p class="session-loading">加载失败：${escapeHtml(error.message)}</p>`;
    }
  }

  async function loadHistory(stateKey, notify = false) {
    state.stateKey = stateKey;
    localStorage.setItem(currentSessionStorageKey, stateKey);
    let history = [];
    try {
      const session = await request(`/api/sessions/${encodeURIComponent(stateKey)}`);
      history = session.messages || [];
      elements.sessionTitle.textContent = session.title || "智能对话";
    } catch (_) {
      try { history = JSON.parse(localStorage.getItem(`hesm.chat.${stateKey}`) || "[]"); } catch (_) { history = []; }
    }
    state.history = history.slice(-100);
    resetMessages(history.length ? "已恢复该会话的历史记录。" : "新会话已开始。长期记忆仍然保留，我会在每轮回答前进行检索。");
    for (let index = 0; index < state.history.length; index += 2) {
      const user = state.history[index];
      const assistant = state.history[index + 1];
      if (user?.role === "user") addMessage("user", user.content);
      if (assistant?.role === "assistant") addMessage("assistant", assistant.content, '<span class="stored-chip">历史记录</span>');
    }
    cacheCurrentHistory();
    await loadSessions();
    if (notify) showToast(`已恢复 ${Math.floor(state.history.length / 2)} 轮历史对话`);
  }

  async function send(message) {
    if (state.running) return;
    state.running = true;
    elements.send.disabled = true;
    elements.send.querySelector("span").textContent = "处理中";
    addMessage("user", message);
    const typing = addTyping();
    startTrace();
    try {
      const payload = await request("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message, history: state.history.slice(-20), state_key: state.stateKey,
          session_id: state.stateKey,
        }),
      });
      typing.remove();
      addMessage("assistant", payload.answer, `<span class="stored-chip">已存入 HESM</span><span>${milliseconds(payload.timing?.chat_total_ms)}</span>`);
      state.history.push({ role: "user", content: message }, { role: "assistant", content: payload.answer });
      state.history = state.history.slice(-100);
      cacheCurrentHistory();
      state.result = payload;
      renderTrace(payload);
      await loadSessions();
      setHealth(true, "对话服务就绪");
    } catch (error) {
      clearInterval(state.timer);
      typing.remove();
      addMessage("assistant", `本轮处理失败：${error.message || String(error)}`);
      setStep("generate", "failed");
      elements.total.textContent = "执行失败";
      showToast("对话执行失败，本轮未写入历史");
    } finally {
      state.running = false;
      elements.send.disabled = false;
      elements.send.querySelector("span").textContent = "发送";
      elements.input.focus();
    }
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = elements.input.value.trim();
    if (!message) return;
    elements.input.value = "";
    send(message);
  });
  elements.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });
  elements.sessionList.addEventListener("click", async (event) => {
    const remove = event.target.closest("[data-delete-session]");
    if (remove) {
      event.stopPropagation();
      if (state.running) return;
      await request(`/api/sessions/${encodeURIComponent(remove.dataset.deleteSession)}`, { method: "DELETE" });
      if (remove.dataset.deleteSession === state.stateKey) await createNewSession();
      else await loadSessions();
      showToast("会话已归档");
      return;
    }
    const item = event.target.closest("[data-session-id]");
    if (item && !state.running) loadHistory(item.dataset.sessionId, true);
  });

  async function createNewSession() {
    if (state.running) return;
    const session = await request("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "新会话" }),
    });
    state.result = null;
    elements.trace.hidden = true;
    elements.empty.hidden = false;
    elements.total.textContent = "等待输入";
    await loadHistory(session.session_id);
    showToast("已创建并保存新会话");
  }
  $("#new-chat").addEventListener("click", createNewSession);
  document.querySelectorAll("[data-dialog]").forEach((button) => button.addEventListener("click", () => {
    if (state.result) elements.dialog.showModal(); else showToast("暂无 Prompt");
  }));
  document.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => elements.dialog.close()));
  elements.dialog.addEventListener("click", (event) => { if (event.target === elements.dialog) elements.dialog.close(); });

  checkHealth();
  request("/api/sessions")
    .then((data) => {
      const exists = (data.items || []).some((item) => item.session_id === state.stateKey);
      return exists ? loadHistory(state.stateKey) : createNewSession();
    })
    .catch(() => createNewSession());
  elements.input.focus();
})();
