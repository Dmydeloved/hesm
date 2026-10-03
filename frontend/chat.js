(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const esc = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const storageKey = "hesm.currentChatSession";
  const state = {
    sessionId: localStorage.getItem(storageKey) || "web_chat",
    history: [], running: false, selectedChatId: "", poll: null,
    liveTimer: null, chatStartedAt: null, activeStage: "", stageStartedAt: null,
  };
  const el = {
    health: $("#health"), form: $("#chat-form"), input: $("#message"), send: $("#send"),
    messages: $("#messages"), empty: $("#trace-empty"), trace: $("#trace"),
    total: $("#total-time"), topic: $("#topic-result"), retrieval: $("#retrieval-result"),
    prompt: $("#prompt-text"), store: $("#store-result"), model: $("#model-name"),
    async: $("#async-result"), sessions: $("#session-list"), title: $("#current-session-title"),
  };

  const ms = (value) => {
    const number = Number(value || 0);
    return number >= 1000 ? `${(number / 1000).toFixed(2)} s` : `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
  };
  async function request(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }
  function toast(message) {
    const node = $("#toast");
    if (!node) return;
    node.textContent = message; node.classList.add("show");
    clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 1800);
  }
  function setHealth(ok, text) {
    el.health.className = `health ${ok ? "ready" : "error"}`;
    el.health.querySelector("span").textContent = text;
  }
  function avatar(role) { return role === "user" ? '<span class="user-avatar">我</span>' : '<span class="assistant-avatar">H</span>'; }
  function addMessage(role, content, meta = "", chatId = "") {
    const node = document.createElement("article");
    node.className = `message ${role}`;
    if (chatId) { node.dataset.chatId = chatId; node.tabIndex = 0; node.title = "点击查看本轮轨迹"; }
    node.innerHTML = `${avatar(role)}<div><strong>${role === "user" ? "你" : "HESM Assistant"}</strong><p class="message-content">${esc(content)}</p>${meta ? `<div class="message-meta">${meta}</div>` : ""}</div>`;
    el.messages.appendChild(node); el.messages.scrollTop = el.messages.scrollHeight; return node;
  }
  function resetMessages(copy) {
    el.messages.innerHTML = `<div class="welcome-message"><span class="assistant-avatar">H</span><div><strong>HESM Assistant</strong><p>${esc(copy)}</p></div></div>`;
  }
  function typing() {
    const node = addMessage("assistant", "");
    node.querySelector(".message-content").className = "message-content typing";
    node.querySelector(".message-content").innerHTML = "<i></i><i></i><i></i>";
    return node;
  }
  function step(step, status = "") {
    const order = ["extract", "retrieve", "prompt", "generate", "store"];
    const index = order.indexOf(step);
    document.querySelectorAll(".trace-step").forEach((node, position) => {
      node.classList.toggle("active", status !== "failed" && status !== "completed" && position === index);
      node.classList.toggle("done", status === "done" || position < index || (status === "completed" && position === index));
      node.classList.toggle("failed", status === "failed" && position === index);
    });
  }
  function setTime(name, value) {
    const node = document.querySelector(`[data-step="${name}"] time`); if (node) node.textContent = ms(value);
  }
  function tickLiveTime() {
    if (state.chatStartedAt !== null) el.total.textContent = ms(performance.now() - state.chatStartedAt);
    if (state.activeStage && state.stageStartedAt !== null) {
      setTime(state.activeStage, performance.now() - state.stageStartedAt);
    }
    document.querySelectorAll('[data-live-elapsed="true"]').forEach((node) => {
      const started = Date.parse(node.dataset.startedAt || "");
      if (Number.isFinite(started)) node.textContent = ms(Date.now() - started);
    });
  }
  function ensureLiveTimer(totalElapsedMs = 0, trackChat = true) {
    if (trackChat && state.chatStartedAt === null) state.chatStartedAt = performance.now() - Math.max(0, totalElapsedMs);
    if (!state.liveTimer) state.liveTimer = setInterval(tickLiveTime, 200);
  }
  function stopLiveTimer() {
    if (state.liveTimer) clearInterval(state.liveTimer);
    state.liveTimer = null; state.chatStartedAt = null; state.activeStage = ""; state.stageStartedAt = null;
  }
  function captureExpandedDetails() {
    const expanded = new Map();
    el.trace.querySelectorAll(".topic-history").forEach((node) => { node.dataset.openKey = "topic-history"; });
    el.trace.querySelectorAll(".retrieval-group").forEach((node, index) => { node.dataset.openKey = `retrieval-${index}`; });
    el.trace.querySelectorAll(".async-job details").forEach((node, index) => { node.dataset.openKey = `distillation-${index}`; });
    el.trace.querySelectorAll("details[data-open-key]").forEach((node) => expanded.set(node.dataset.openKey, node.open));
    return expanded;
  }
  function restoreExpandedDetails(expanded) {
    el.trace.querySelectorAll(".topic-history").forEach((node) => { node.dataset.openKey = "topic-history"; });
    el.trace.querySelectorAll(".retrieval-group").forEach((node, index) => { node.dataset.openKey = `retrieval-${index}`; });
    el.trace.querySelectorAll(".async-job details").forEach((node, index) => { node.dataset.openKey = `distillation-${index}`; });
    el.trace.querySelectorAll("details[data-open-key]").forEach((node) => {
      if (expanded.has(node.dataset.openKey)) node.open = expanded.get(node.dataset.openKey);
    });
  }
  function resetTrace() {
    el.empty.hidden = true; el.trace.hidden = false; el.total.textContent = "等待操作";
    el.topic.innerHTML = "<p>暂无主题提取记录</p>";
    el.retrieval.innerHTML = "<p>暂无检索记录</p>";
    el.prompt.textContent = "暂无 Prompt 记录";
    el.store.innerHTML = "<p>暂无记忆添加记录</p>";
    el.async.innerHTML = "<p>暂无 Segment / Experience 蒸馏记录</p>";
    document.querySelectorAll(".trace-step time").forEach((node) => { node.textContent = "—"; });
    document.querySelectorAll(".trace-step").forEach((node) => node.classList.remove("active", "done", "failed"));
  }
  function startTrace() {
    el.empty.hidden = true; el.trace.hidden = false; el.total.textContent = "处理中";
    el.topic.innerHTML = "<p>正在分析主题与意图…</p>";
    el.retrieval.innerHTML = "<p>等待检索完成…</p>";
    el.prompt.textContent = "等待 Prompt 拼接完成…"; el.store.innerHTML = "<p>等待记忆添加…</p>";
    el.async.innerHTML = "<p>等待异步更新…</p>";
    document.querySelectorAll(".trace-step time").forEach((node) => { node.textContent = "—"; });
    step("extract");
  }
  function renderExtraction(value, history = []) {
    const item = value || {};
    const historyHtml = history.length
      ? `<details class="topic-history"><summary>本会话主题提取历史（${history.length}）</summary>${history.map((record, index) => `<article><strong>${index + 1}. ${esc(record.created_at || "时间未知")} · ${esc(record.source || "")}</strong><p>${esc(record.user_input || "")}</p><pre>${esc(JSON.stringify(record.topic_extraction_result, null, 2))}</pre></article>`).join("")}</details>`
      : "";
    el.topic.innerHTML = `<div class="topic-grid"><div><small>主题</small><strong>${esc(item.topic || "—")}</strong></div><div><small>核心实体</small><strong>${esc(item.core_entity || "—")}</strong></div><div><small>意图</small><strong>${esc(item.intent || "—")}</strong></div><div><small>置信度</small><strong>${Math.round(Number(item.confidence || 0) * 100)}%</strong></div><div><small>相关实体</small><strong>${esc((item.entities || []).join("、") || "—")}</strong></div></div>${historyHtml}`;
  }
  function field(label, value) { return `<dt>${esc(label)}</dt><dd>${esc(typeof value === "object" ? JSON.stringify(value, null, 2) : (value ?? "—"))}</dd>`; }
  function retrievalItem(type, item) {
    const id = item[`${type}_id`] || item.id || "—";
    const fields = type === "experience" ? [["ID", id], ["主题", item.topic], ["核心实体", item.core_entity], ["状态", item.status], ["摘要", item.summary]] : type === "segment" ? [["ID", id], ["主题", item.topic], ["意图", item.intent], ["状态", item.status], ["摘要", item.summary]] : [["ID", id], ["时间", item.timestamp], ["用户输入", item.user_input], ["助手回答", item.assistant_output]];
    return `<article class="retrieval-item"><strong>${esc(item.topic || item.intent || item.user_input || id)}</strong><small>${esc(id)}</small><dl>${fields.map(([label, value]) => field(label, value)).join("")}</dl></article>`;
  }
  function group(type, label, items) {
    const records = Array.isArray(items) ? items : [];
    return `<details class="retrieval-group"><summary><b>${records.length}</b>${label}<span>点击查看</span></summary><div class="retrieval-items">${records.length ? records.map((item) => retrievalItem(type, item)).join("") : "<p>未检索到相关记录。</p>"}</div></details>`;
  }
  function renderRetrieval(value) {
    const data = value || {};
    el.retrieval.innerHTML = `${group("experience", "Experience", data.experiences)}${group("segment", "Segment", data.segments)}${group("qa", "QA", data.qas)}`;
  }
  function renderStored(item) {
    const value = item || {};
    el.store.innerHTML = `<div class="store-path"><code>${esc(value.experience_id || "—")}</code><i>→</i><code>${esc(value.segment_id || "—")}</code><i>→</i><code>${esc(value.qa_id || "—")}</code></div>`;
  }
  function renderUpdates(trace) {
    const updates = [...(trace.segment_updates || []), ...(trace.experience_updates || [])];
    const outbox = trace.outbox || [];
    const pending = outbox.filter((job) => ["pending", "processing"].includes(job.status)).length;
    const html = updates.map((item) => {
      const target = item.segment_id || item.experience_id || "";
      const kind = item.segment_id ? "Segment 蒸馏" : "Experience 蒸馏";
      return `<div class="async-job"><strong>${esc(kind)}</strong><span>${esc(item.status)} · ${esc(target)} · ${ms(item.elapsed_ms)}</span>${item.error_message ? `<p>${esc(item.error_message)}</p>` : ""}<details><summary>查看完整蒸馏轨迹</summary><pre>${esc(JSON.stringify({old: item.old_segment || item.old_experience, input: item.distill_input, prompt: item.prompt_text, output: item.output_text, new: item.new_segment || item.new_experience}, null, 2))}</pre></details></div>`;
    }).join("");
    el.async.innerHTML = `<p>${updates.length ? `已记录 ${updates.length} 条蒸馏，${pending} 个任务处理中` : "暂无蒸馏记录"}</p>${html}`;
    el.async.querySelectorAll(".async-job").forEach((node, index) => {
      const item = updates[index];
      if (item?.status !== "running") return;
      const label = node.querySelector("span");
      const startedAt = item.started_at || item.created_at || "";
      const started = Date.parse(startedAt);
      const time = document.createElement("time");
      time.dataset.liveElapsed = "true";
      time.dataset.startedAt = startedAt;
      time.textContent = ms(Number.isFinite(started) ? Date.now() - started : item.elapsed_ms);
      label.textContent = `${item.status} · ${item.segment_id || item.experience_id || ""} · `;
      label.appendChild(time);
    });
    const node = document.querySelector('[data-step="async"]'); if (node) node.classList.toggle("done", updates.length > 0 && pending === 0);
    return pending;
  }
  function renderResult(result) {
    const timing = result.timing || {};
    state.chatStartedAt = null; state.activeStage = ""; state.stageStartedAt = null;
    el.total.textContent = ms(timing.chat_total_ms); renderExtraction(result.extraction); renderRetrieval(result.retrieval); el.prompt.textContent = result.prompt || ""; el.model.textContent = result.model || "Answer model"; renderStored(result.stored?.memories?.[0]);
    Object.entries({extract: timing.topic_extraction_ms, retrieve: timing.retrieval_ms, prompt: timing.prompt_assembly_ms, generate: timing.generation_ms, store: timing.storage_ms}).forEach(([key, value]) => setTime(key, value));
    step("store", "done");
  }
  async function showTrajectory(chatId) {
    if (!chatId) return; state.selectedChatId = chatId; clearTimeout(state.poll);
    try {
      const trace = await request(`/api/chats/${encodeURIComponent(chatId)}/trajectory?session_id=${encodeURIComponent(state.sessionId)}`);
      if (state.selectedChatId !== chatId) return;
      const expanded = captureExpandedDetails();
      startTrace(); el.prompt.textContent = trace.chat?.prompt_text || "暂无已保存的 Prompt（旧记录可能没有该字段）"; const retriever = trace.retriever || {}; const add = trace.add || {};
      renderExtraction(retriever.topic_extract, trace.topic_extraction_history || []); renderRetrieval(retriever); renderStored({experience_id: add.experience_id, segment_id: add.segment_id, qa_id: (add.qa_ids || [])[0]});
      const savedTimings = trace.chat?.timing || {};
      Object.entries({extract: savedTimings.topic_extraction_ms, retrieve: savedTimings.retrieval_ms, prompt: savedTimings.prompt_assembly_ms, generate: savedTimings.generation_ms, store: savedTimings.storage_ms}).forEach(([key, value]) => { if (value !== undefined) setTime(key, value); });
      const chatIsRunning = trace.chat?.status === "running";
      const pending = renderUpdates(trace);
      const asyncWork = pending > 0 || (trace.outbox || []).some((job) => ["pending", "processing"].includes(job.status));
      const createdAt = Date.parse(trace.chat?.created_at || "");
      const elapsed = chatIsRunning && Number.isFinite(createdAt) ? Date.now() - createdAt : Number(trace.chat?.elapsed_ms || 0);
      el.total.textContent = ms(elapsed);
      restoreExpandedDetails(expanded);
      if (chatIsRunning || asyncWork) {
        ensureLiveTimer(chatIsRunning ? elapsed : 0, chatIsRunning);
        state.poll = setTimeout(() => showTrajectory(chatId), 1500);
      } else {
        stopLiveTimer();
      }
    } catch (error) { el.async.innerHTML = `<p>轨迹加载失败：${esc(error.message)}</p>`; }
  }
  function updateEvent(event) {
    if (event.event === "stage_started") { state.activeStage = event.stage; state.stageStartedAt = performance.now(); setTime(event.stage, 0); step(event.stage); return; }
    if (event.event !== "stage_completed") return;
    step(event.stage, "completed"); setTime(event.stage, event.timing_ms);
    if (state.activeStage === event.stage) { state.activeStage = ""; state.stageStartedAt = null; }
    if (event.stage === "extract") renderExtraction(event.extraction);
    if (event.stage === "retrieve") renderRetrieval(event.retrieval);
    if (event.stage === "prompt") el.prompt.textContent = event.prompt || "";
    if (event.stage === "generate") el.model.textContent = event.model || "Answer model";
    if (event.stage === "store") renderStored(event.stored?.memories?.[0]);
  }
  async function stream(url, body, onEvent) {
    const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let result = null;
    while (true) { const {value, done} = await reader.read(); if (done) break; buffer += decoder.decode(value, {stream: true}); const lines = buffer.split("\n"); buffer = lines.pop() || ""; for (const line of lines) { if (!line.trim()) continue; const event = JSON.parse(line); if (event.event === "error") throw new Error(event.message); onEvent(event); if (event.event === "final") result = event.result; } }
    if (buffer.trim()) { const event = JSON.parse(buffer); if (event.event === "error") throw new Error(event.message); onEvent(event); if (event.event === "final") result = event.result; }
    if (!result) throw new Error("服务端未返回最终结果"); return result;
  }
  async function loadSessions() {
    const sessions = (await request("/api/sessions")).items || [];
    el.sessions.innerHTML = sessions.length ? sessions.map((item) => `<button class="session-item ${item.session_id === state.sessionId ? "active" : ""}" data-session-id="${esc(item.session_id)}"><strong>${esc(item.title || "智能对话")}</strong><span>${esc(item.preview || "暂无消息")}</span><small>${item.turn_count || 0} 轮 · ${esc((item.update_at || "").replace("T", " ").slice(0, 16))}</small><i class="session-delete" data-delete-session="${esc(item.session_id)}" title="归档会话">×</i></button>`).join("") : "<p class='session-loading'>暂无历史会话。</p>";
  }
  async function loadHistory(sessionId) {
    state.sessionId = sessionId; localStorage.setItem(storageKey, sessionId); state.selectedChatId = ""; clearTimeout(state.poll); resetTrace();
    const session = await request(`/api/sessions/${encodeURIComponent(sessionId)}`); state.history = session.messages || []; el.title.textContent = session.title || "智能对话";
    resetMessages(state.history.length ? "已恢复该会话的历史记录。" : "新会话已开始。每轮回答后都会沉淀为 HESM 记忆。");
    let latestChatId = "";
    for (let index = 0; index < state.history.length; index += 2) { const user = state.history[index]; const assistant = state.history[index + 1]; const chatId = user?.metadata?.chat_id || assistant?.metadata?.chat_id || ""; if (chatId) latestChatId = chatId; if (user?.role === "user") addMessage("user", user.content, "", chatId); if (assistant?.role === "assistant") addMessage("assistant", assistant.content, '<span class="stored-chip">历史记录</span>', chatId); }
    await loadSessions();
    if (latestChatId) await showTrajectory(latestChatId);
  }
  async function createSession() {
    if (state.running) return; const session = await request("/api/sessions", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({title: ""})}); await loadHistory(session.session_id); toast("已创建新会话");
  }
  async function send(message) {
    if (state.running) return; state.running = true; el.send.disabled = true; const userNode = addMessage("user", message); const pending = typing(); startTrace(); ensureLiveTimer();
    try {
      const result = await stream("/api/chat/stream", {message, session_id: state.sessionId}, (event) => { if (event.event === "start" && event.chat_id) { userNode.dataset.chatId = event.chat_id; userNode.tabIndex = 0; } updateEvent(event); });
      pending.remove(); addMessage("assistant", result.answer, `<span class="stored-chip">已保存</span><span>${ms(result.timing?.chat_total_ms)}</span>`, result.chat_id); state.result = result; renderResult(result); state.selectedChatId = result.chat_id; showTrajectory(result.chat_id); await loadSessions(); setHealth(true, "对话服务就绪");
    } catch (error) { pending.remove(); addMessage("assistant", `本轮处理失败：${error.message}`); step("generate", "failed"); el.total.textContent = "执行失败"; toast("本轮处理失败"); }
    finally { if (state.chatStartedAt !== null) stopLiveTimer(); state.running = false; el.send.disabled = false; el.input.focus(); }
  }
  el.form.addEventListener("submit", (event) => { event.preventDefault(); const message = el.input.value.trim(); if (!message) return; el.input.value = ""; send(message); });
  el.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); el.form.requestSubmit(); } });
  el.messages.addEventListener("click", (event) => { const node = event.target.closest("[data-chat-id]"); if (node && !state.running) showTrajectory(node.dataset.chatId); });
  el.messages.addEventListener("keydown", (event) => { if (!["Enter", " "].includes(event.key)) return; const node = event.target.closest("[data-chat-id]"); if (node && !state.running) { event.preventDefault(); showTrajectory(node.dataset.chatId); } });
  el.sessions.addEventListener("click", async (event) => { const remove = event.target.closest("[data-delete-session]"); if (remove) { event.stopPropagation(); await request(`/api/sessions/${encodeURIComponent(remove.dataset.deleteSession)}`, {method: "DELETE"}); if (remove.dataset.deleteSession === state.sessionId) await createSession(); else await loadSessions(); return; } const item = event.target.closest("[data-session-id]"); if (item && !state.running) loadHistory(item.dataset.sessionId); });
  $("#new-chat").addEventListener("click", createSession);
  request("/api/health").then((data) => setHealth(data.status !== "degraded", data.status === "ready" ? "对话服务就绪" : "记忆库为空")).catch(() => setHealth(false, "WebServer 未连接"));
  request("/api/sessions").then((data) => (data.items || []).some((item) => item.session_id === state.sessionId) ? loadHistory(state.sessionId) : createSession()).catch(createSession);
  el.input.focus();
})();
