(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const state = { running: false, result: null, timer: null };
  const elements = {
    health: $("#health"), form: $("#retrieval-form"), query: $("#query"),
    submit: $("#submit"), pipeline: $("#pipeline"), welcome: $("#welcome"),
    results: $("#results"), confidence: $("#confidence"),
    extraction: $("#extraction"), reasoning: $("#reasoning"),
    total: $("#total-time"), timing: $("#timing"),
    diagnostics: $("#diagnostics"), summary: $("#summary"),
    tree: $("#tree"), context: $("#context"),
    contextText: $("#context-text"), toast: $("#toast"),
  };

  function toast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => elements.toast.classList.remove("show"), 1800);
  }

  function setHealth(ok, label) {
    elements.health.className = `health ${ok ? "ready" : "error"}`;
    elements.health.querySelector("span").textContent = label;
  }

  async function health() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      const ready = response.ok && data.status !== "degraded";
      setHealth(ready, ready ? "检索服务就绪" : "记忆库为空");
    } catch (_) {
      setHealth(false, "请启动 WebServer");
    }
  }

  function milliseconds(value) {
    const number = Number(value || 0);
    return number >= 1000
      ? `${(number / 1000).toFixed(2)} s`
      : `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
  }

  function percentage(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
  }

  function setStage(stage, error = false) {
    const order = ["extract", "recall", "assemble"];
    const index = order.indexOf(stage);
    elements.pipeline.querySelectorAll("article").forEach((item, position) => {
      item.classList.toggle("active", !error && position === index);
      item.classList.toggle("done", !error && (stage === "done" || position < index));
      item.classList.toggle("error", error && position === index);
    });
  }

  function setPipelineTimes(timing) {
    const values = {
      extract: timing.topic_extraction_ms,
      recall: timing.retrieval_ms,
      assemble: timing.response_assembly_ms,
    };
    Object.entries(values).forEach(([key, value]) => {
      elements.pipeline.querySelector(`[data-stage="${key}"] time`).textContent = milliseconds(value);
    });
  }

  function renderExtraction(data) {
    elements.confidence.textContent = `置信度 ${percentage(data.confidence)}`;
    const items = [
      ["主题", data.topic],
      ["核心实体", data.core_entity],
      ["意图", data.intent],
      ["相关实体", (data.entities || []).join("、") || "—"],
    ];
    elements.extraction.innerHTML = items.map(([key, value]) => (
      `<div><small>${key}</small><strong>${escapeHtml(value || "—")}</strong></div>`
    )).join("");
    elements.reasoning.textContent = data.reasoning || "未返回判断依据";
  }

  function renderTimings(result) {
    const timing = result.timing || {};
    const groups = [
      ["主题提取", timing.topic_extraction_ms],
      ["Experience 与上下文查询", timing.retrieval_ms],
      ["响应组装", timing.response_assembly_ms],
    ];
    const maximum = Math.max(1, ...groups.map(([, value]) => Number(value || 0)));
    elements.total.textContent = milliseconds(timing.total_ms);
    elements.timing.innerHTML = groups.map(([name, value]) => (
      `<div class="timing-group"><button type="button"><span>${name}</span><b>${milliseconds(value)}</b></button>`
      + `<div class="timing-bar"><i style="width:${Math.max(1, Number(value || 0) / maximum * 100)}%"></i></div></div>`
    )).join("");
  }

  function renderDiagnostics(result) {
    const limits = result.limits || {};
    const items = [
      ["Experience", result.experiences?.length || 0],
      ["Segment", result.segments?.length || 0],
      ["QA", result.qas?.length || 0],
      ["检索范围", `${limits.top_experience || 1}E / ${limits.top_segment || 2}S / ${limits.top_qa || 4}Q`],
      ["排序", "时间升序"],
      ["上下文", result.context ? "已生成" : "为空"],
    ];
    elements.diagnostics.innerHTML = items.map(([key, value]) => (
      `<div><small>${key}</small><strong>${escapeHtml(value)}</strong></div>`
    )).join("");
  }

  function informationGrid(rows) {
    return `<dl class="node-info">${rows.filter(([, value]) => (
      value !== undefined && value !== null && value !== ""
    )).map(([label, value]) => (
      `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(Array.isArray(value) ? value.join("、") || "—" : value)}</dd></div>`
    )).join("")}</dl>`;
  }

  function qaNode(qa, index) {
    return `<article class="tree-qa"><header><span>Q${index + 1}</span><div>`
      + `<strong>${escapeHtml(qa.intent || qa.topic || "QA")}</strong><small>${escapeHtml(qa.qa_id)}</small>`
      + `</div></header><details><summary>查看 QA 完整信息</summary>`
      + informationGrid([
        ["时间", qa.timestamp], ["主题", qa.topic], ["核心实体", qa.core_entity],
        ["意图", qa.intent], ["状态", qa.status], ["相关实体", qa.entities],
      ])
      + `<section class="qa-content"><b>用户输入</b><p>${escapeHtml(qa.user_input || "—")}</p>`
      + `${qa.assistant_output ? `<b>助手输出</b><p>${escapeHtml(qa.assistant_output)}</p>` : ""}`
      + `</section></details></article>`;
  }

  function segmentNode(segment, index) {
    return `<section class="tree-segment"><header><span>S${index + 1}</span><div>`
      + `<strong>${escapeHtml(segment.intent || segment.topic || "Segment")}</strong>`
      + `<small>${escapeHtml(segment.segment_id)}</small></div></header>`
      + `<details open><summary>查看 Segment 完整信息</summary>`
      + informationGrid([
        ["主题", segment.topic], ["核心实体", segment.core_entity],
        ["意图", segment.intent], ["状态", segment.status],
        ["更新时间", segment.updated_at],
      ])
      + `${segment.summary ? `<section class="node-summary"><b>Segment 摘要</b><p>${escapeHtml(segment.summary)}</p></section>` : ""}`
      + `</details><div class="tree-qas">${segment.qas.map(qaNode).join("") || '<p class="tree-empty">无 QA 节点</p>'}</div></section>`;
  }

  function experienceNode(experience, index) {
    return `<article class="tree-experience selected"><header><span>E${index + 1}</span><div>`
      + `<strong>${escapeHtml(experience.topic || "Experience")}</strong>`
      + `<small>${escapeHtml(experience.core_entity || "—")} · ${escapeHtml(experience.experience_id)}</small>`
      + `</div></header><details open><summary>查看 Experience 完整信息</summary>`
      + informationGrid([
        ["状态", experience.state?.status], ["关联意图", experience.intents_link || experience.intents],
        ["创建时间", experience.created_at], ["更新时间", experience.updated_at], ["版本", experience.version],
      ])
      + `${experience.summary ? `<section class="node-summary"><b>Experience 摘要</b><p>${escapeHtml(experience.summary)}</p></section>` : ""}`
      + `${experience.history_experience ? `<section class="node-summary"><b>历史经验</b><p>${escapeHtml(typeof experience.history_experience === "string" ? experience.history_experience : JSON.stringify(experience.history_experience, null, 2))}</p></section>` : ""}`
      + `</details><div class="tree-segments">${experience.segments.map(segmentNode).join("") || '<p class="tree-empty">无 Segment 节点</p>'}</div></article>`;
  }

  function buildTree(result) {
    const qas = result.qas || [];
    const segments = (result.segments || []).map((segment) => ({
      ...segment,
      qas: qas.filter((qa) => qa.segment_id === segment.segment_id),
    }));
    return (result.experiences || []).map((experience) => ({
      ...experience,
      segments: segments.filter((segment) => segment.experience_id === experience.experience_id),
    }));
  }

  function render(result) {
    state.result = result;
    elements.welcome.hidden = true;
    elements.results.hidden = false;
    renderExtraction(result.query_extraction || {});
    renderTimings(result);
    renderDiagnostics(result);
    setPipelineTimes(result.timing || {});
    elements.summary.innerHTML = `<span>返回 <b>${result.experiences?.length || 0}</b>E / <b>${result.segments?.length || 0}</b>S / <b>${result.qas?.length || 0}</b>Q</span>`;
    const tree = buildTree(result);
    elements.tree.innerHTML = tree.length
      ? tree.map(experienceNode).join("")
      : '<div class="error-box">没有可用的 Experience。</div>';
    elements.contextText.textContent = result.context || "未生成上下文";
  }

  function renderError(message) {
    elements.welcome.hidden = false;
    elements.results.hidden = true;
    elements.welcome.innerHTML = `<div class="error-box"><h2>检索执行失败</h2><p>${escapeHtml(message)}</p><small>请检查模型、向量服务配置和 HESM 数据。</small></div>`;
  }

  async function run(question) {
    if (state.running) return;
    state.running = true;
    elements.submit.disabled = true;
    elements.submit.querySelector("span").textContent = "正在检索";
    setStage("extract");
    state.timer = setTimeout(() => setStage("recall"), 500);
    try {
      const response = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      setStage("assemble");
      render(payload);
      setTimeout(() => setStage("done"), 180);
      setHealth(true, "检索服务就绪");
    } catch (error) {
      setStage("recall", true);
      renderError(error.message || String(error));
    } finally {
      clearTimeout(state.timer);
      state.running = false;
      elements.submit.disabled = false;
      elements.submit.querySelector("span").textContent = "开始检索";
    }
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = elements.query.value.trim();
    if (question) run(question);
  });
  document.querySelectorAll("[data-query]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.query.value = button.dataset.query;
      elements.query.focus();
    });
  });
  document.querySelector(".result-toolbar").addEventListener("click", (event) => {
    const button = event.target.closest("[data-view]");
    if (!button) return;
    document.querySelectorAll("[data-view]").forEach((item) => item.classList.toggle("active", item === button));
    const showContext = button.dataset.view === "context";
    $("#selected-view").hidden = showContext;
    elements.context.hidden = !showContext;
  });
  $("#copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(state.result?.context || "");
      toast("上下文已复制");
    } catch (_) {
      toast("复制失败");
    }
  });

  setStage("idle");
  health();
})();
