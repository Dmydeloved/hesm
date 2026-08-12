(function () {
  "use strict";

  const state = { result: null, running: false, stageTimer: null };
  const $ = (selector) => document.querySelector(selector);
  const elements = {
    form: $("#retrieval-form"),
    query: $("#retrieval-query"),
    button: $("#retrieve-button"),
    apiStatus: $("#api-status"),
    empty: $("#retrieval-empty"),
    workspace: $("#retrieval-workspace"),
    extractionConfidence: $("#extraction-confidence"),
    extractionFields: $("#extraction-fields"),
    extractionReasoning: $("#extraction-reasoning"),
    timingGrid: $("#timing-grid"),
    debugList: $("#debug-list"),
    resultSummary: $("#result-summary"),
    treeView: $("#tree-view"),
    contextView: $("#context-view"),
    contextText: $("#context-text"),
    copyContext: $("#copy-context"),
    toast: $("#retrieval-toast"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function shortId(id, length = 10) {
    if (!id) return "—";
    const [prefix, suffix = ""] = String(id).split("_");
    return `${prefix}_${suffix.slice(0, length)}${suffix.length > length ? "…" : ""}`;
  }

  function percentage(value) {
    const score = Number(value);
    return Number.isFinite(score) ? `${Math.round(score * 100)}%` : "—";
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 1800);
  }

  function setApiStatus(status, label) {
    elements.apiStatus.classList.toggle("error", status === "error");
    elements.apiStatus.classList.toggle("ready", status === "ready");
    elements.apiStatus.querySelector("span").textContent = label;
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      if (!response.ok) throw new Error("health check failed");
      const health = await response.json();
      setApiStatus(health.status === "ready" ? "ready" : "error", health.status === "ready" ? "真实检索服务已就绪" : "记忆文件缺失");
    } catch (_error) {
      setApiStatus("error", "请使用 server.py 启动");
    }
  }

  function setPipeline(stage) {
    const order = ["extract", "retrieve", "trace"];
    const activeIndex = order.indexOf(stage);
    document.querySelectorAll(".pipeline-step").forEach((step, index) => {
      step.classList.toggle("active", index === activeIndex);
      step.classList.toggle("done", stage === "done" || index < activeIndex);
    });
    document.querySelector(".pipeline-progress").classList.toggle("running", stage !== "done" && stage !== "idle");
  }

  function renderExtraction(extraction) {
    const fields = [
      ["TOPIC", extraction.topic, "violet"],
      ["CORE ENTITY", extraction.core_entity, "teal"],
      ["INTENT", extraction.intent, "orange"],
    ];
    elements.extractionConfidence.textContent = `置信度 ${percentage(extraction.confidence)}`;
    elements.extractionFields.innerHTML = fields
      .map(([label, value, tone]) => `<div class="extraction-field ${tone}"><small>${label}</small><strong>${escapeHtml(value)}</strong></div>`)
      .join("") + `
        <div class="extraction-field entities-field">
          <small>ENTITIES</small>
          <div>${(extraction.entities || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("") || "—"}</div>
        </div>`;
    elements.extractionReasoning.textContent = extraction.reasoning || "—";
  }

  function renderTimings(timing) {
    const items = [
      ["主题提取", timing.topic_extraction_ms, "ms"],
      ["层级检索", timing.retrieval_ms, "ms"],
      ["总耗时", timing.total_ms, "ms"],
    ];
    elements.timingGrid.innerHTML = items
      .map(([label, value, unit]) => `<div><small>${label}</small><strong>${escapeHtml(value)}<i>${unit}</i></strong></div>`)
      .join("");
  }

  function countText(counts) {
    if (!counts) return "—";
    return `E ${counts.experience ?? 0} / S ${counts.segment ?? 0} / Q ${counts.qa ?? 0}`;
  }

  function renderDebug(debug) {
    const pipeline = debug.pipeline_counts || {};
    const source = debug.source_candidates || {};
    const sourceTotal = Object.values(source).reduce((sum, value) => sum + Number(value || 0), 0);
    const rows = [
      ["检索策略", debug.strategy || "—"],
      ["候选规模", countText(pipeline.before_context_pruning)],
      ["最终选择", countText(pipeline.final_selected)],
      ["来源候选", `${sourceTotal} 条`],
      ["低置信度 QA 救援", debug.low_confidence_qa_rescue ? "已启用" : "未启用"],
      ["LLM 重排调用", `${debug.llm_calls ?? 0} 次`],
    ];
    elements.debugList.innerHTML = rows
      .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
      .join("");
  }

  const sourceLabels = {
    experience_vector: "Experience 向量",
    experience_relational: "Experience 关系",
    initial_experience_descendant: "Experience 后代",
    scoped_segment_vector: "Segment 向量",
    scoped_qa_vector: "QA 向量",
    global_qa_vector: "全局 QA 向量",
    global_qa_relational: "全局 QA 关系",
    qa_rescue_ancestor: "QA 救援祖先",
  };

  function sourceClass(source) {
    if (source.includes("vector")) return "vector";
    if (source.includes("relational")) return "relation";
    return "descendant";
  }

  function renderSources(sources) {
    if (!sources?.length) return '<span class="source-tag neutral">未标注来源</span>';
    return sources
      .map((source) => `<span class="source-tag ${sourceClass(source)}">${escapeHtml(sourceLabels[source] || source)}</span>`)
      .join("");
  }

  function renderScoreDetails(item) {
    const metrics = [
      ["向量", item.vector_similarity],
      ["关键词", item.keyword_score],
      ["关系", item.relation_score],
    ].filter(([, value]) => value !== undefined);
    return metrics.length
      ? `<div class="score-details">${metrics.map(([label, value]) => `<span>${label} <b>${percentage(value)}</b></span>`).join("")}</div>`
      : "";
  }

  function managementUrl(experienceId, segmentId, qaId = "") {
    const params = new URLSearchParams({ experience: experienceId, segment: segmentId });
    if (qaId) params.set("qa", qaId);
    return `index.html?${params.toString()}`;
  }

  function renderQa(qa, experienceId, segmentId, index) {
    return `
      <article class="retrieved-qa">
        <div class="qa-result-head">
          <span class="result-level-icon qa-level">Q${String(index + 1).padStart(2, "0")}</span>
          <div><small>QA EVIDENCE</small><strong>${escapeHtml(qa.topic || qa.intent || "原始对话证据")}</strong></div>
          <div class="result-score"><strong>${percentage(qa.score)}</strong><small>相关度</small></div>
        </div>
        <div class="source-path">
          <span>${escapeHtml(shortId(experienceId))}</span><i>›</i><span>${escapeHtml(shortId(segmentId))}</span><i>›</i><strong>${escapeHtml(shortId(qa.qa_id))}</strong>
        </div>
        <blockquote>${escapeHtml(qa.user_input || "")}</blockquote>
        ${qa.assistant_output ? `<div class="assistant-evidence"><small>ASSISTANT</small><p>${escapeHtml(qa.assistant_output)}</p></div>` : ""}
        <div class="evidence-tags">${(qa.entities || []).map((entity) => `<span>${escapeHtml(entity)}</span>`).join("")}</div>
        ${qa.llm_reason ? `<p class="selection-reason"><b>选择理由：</b>${escapeHtml(qa.llm_reason)}</p>` : ""}
        <footer>
          <div class="source-tags">${renderSources(qa.retrieval_sources)}</div>
          ${renderScoreDetails(qa)}
          <a href="${managementUrl(experienceId, segmentId, qa.qa_id)}">在记忆管理中查看 ↗</a>
        </footer>
      </article>`;
  }

  function renderSegment(segment, qas, experienceId, index) {
    return `
      <section class="retrieved-segment">
        <header>
          <span class="result-level-icon segment-level">S${String(index + 1).padStart(2, "0")}</span>
          <div><small>SEGMENT</small><h4>${escapeHtml(segment.intent || segment.topic)}</h4><code>${escapeHtml(segment.segment_id)}</code></div>
          <div class="result-score"><strong>${percentage(segment.score)}</strong><small>相关度</small></div>
        </header>
        <div class="segment-result-meta">
          <div class="source-tags">${renderSources(segment.retrieval_sources)}</div>
          ${renderScoreDetails(segment)}
        </div>
        ${segment.summary ? `<p class="segment-result-summary">${escapeHtml(segment.summary)}</p>` : ""}
        ${segment.llm_reason ? `<p class="selection-reason"><b>选择理由：</b>${escapeHtml(segment.llm_reason)}</p>` : ""}
        <div class="qa-results">${qas.map((qa, qaIndex) => renderQa(qa, experienceId, segment.segment_id, qaIndex)).join("")}</div>
      </section>`;
  }

  function renderTree(result) {
    const segmentsByExperience = new Map();
    const qasBySegment = new Map();
    result.segments.forEach((segment) => {
      if (!segmentsByExperience.has(segment.experience_id)) segmentsByExperience.set(segment.experience_id, []);
      segmentsByExperience.get(segment.experience_id).push(segment);
    });
    result.qas.forEach((qa) => {
      if (!qasBySegment.has(qa.segment_id)) qasBySegment.set(qa.segment_id, []);
      qasBySegment.get(qa.segment_id).push(qa);
    });

    if (!result.experiences.length) {
      elements.treeView.innerHTML = '<div class="no-results"><span>◇</span><h3>没有召回到结构完整的记忆</h3><p>可以调整问题表达或提高召回数量后重试。</p></div>';
      return;
    }

    elements.treeView.innerHTML = result.experiences
      .map((experience, index) => {
        const segments = segmentsByExperience.get(experience.experience_id) || [];
        return `
          <article class="retrieved-experience">
            <header class="experience-result-head">
              <span class="result-level-icon experience-level">E${String(index + 1).padStart(2, "0")}</span>
              <div><small>EXPERIENCE</small><h3>${escapeHtml(experience.topic)}</h3><p>${escapeHtml(experience.core_entity)} · <code>${escapeHtml(experience.experience_id)}</code></p></div>
              <div class="result-score"><strong>${percentage(experience.score)}</strong><small>相关度</small></div>
            </header>
            <div class="experience-result-meta">
              <div class="source-tags">${renderSources(experience.retrieval_sources)}</div>
              ${renderScoreDetails(experience)}
            </div>
            ${experience.summary ? `<details class="result-summary-details"><summary>查看 Experience 总结</summary><p>${escapeHtml(experience.summary)}</p></details>` : ""}
            ${experience.llm_reason ? `<p class="selection-reason"><b>选择理由：</b>${escapeHtml(experience.llm_reason)}</p>` : ""}
            <div class="segment-results">${segments.map((segment, segmentIndex) => renderSegment(segment, qasBySegment.get(segment.segment_id) || [], experience.experience_id, segmentIndex)).join("")}</div>
          </article>`;
      })
      .join("");
  }

  function renderResult(result) {
    state.result = result;
    elements.empty.hidden = true;
    elements.workspace.hidden = false;
    renderExtraction(result.extraction);
    renderTimings(result.timing);
    renderDebug(result.debug || {});
    elements.resultSummary.innerHTML = `
      <span><b>${result.experiences.length}</b> Experience</span>
      <span><b>${result.segments.length}</b> Segment</span>
      <span><b>${result.qas.length}</b> QA</span>`;
    renderTree(result);
    elements.contextText.textContent = result.context_text || "未生成上下文。";
  }

  function renderError(message) {
    elements.empty.hidden = false;
    elements.workspace.hidden = true;
    elements.empty.innerHTML = `<div class="error-mark">!</div><h2>真实检索执行失败</h2><p>${escapeHtml(message)}</p><small>请确认通过 <code>python -m frontend.server</code> 启动，并检查模型与向量服务配置。</small>`;
  }

  async function runRetrieval(question) {
    if (state.running) return;
    state.running = true;
    elements.button.disabled = true;
    elements.button.classList.add("loading");
    elements.button.querySelector("span").textContent = "正在执行";
    setPipeline("extract");
    state.stageTimer = window.setTimeout(() => setPipeline("retrieve"), 900);

    try {
      const response = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_experience: Number($("#top-experience").value),
          top_segment: Number($("#top-segment").value),
          top_qa: Number($("#top-qa").value),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      setPipeline("trace");
      renderResult(payload);
      window.setTimeout(() => setPipeline("done"), 250);
      setApiStatus("ready", "真实检索服务已就绪");
    } catch (error) {
      setPipeline("idle");
      renderError(error.message || String(error));
    } finally {
      window.clearTimeout(state.stageTimer);
      elements.button.disabled = false;
      elements.button.classList.remove("loading");
      elements.button.querySelector("span").textContent = "开始检索";
      state.running = false;
    }
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = elements.query.value.trim();
    if (question) runRetrieval(question);
  });

  document.querySelectorAll("[data-query]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.query.value = button.dataset.query;
      elements.query.focus();
    });
  });

  document.querySelector(".view-switch").addEventListener("click", (event) => {
    const button = event.target.closest("[data-view]");
    if (!button) return;
    document.querySelectorAll("[data-view]").forEach((item) => item.classList.toggle("active", item === button));
    const showTree = button.dataset.view === "tree";
    elements.treeView.hidden = !showTree;
    elements.contextView.hidden = showTree;
  });

  elements.copyContext.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(state.result?.context_text || "");
      showToast("上下文已复制");
    } catch (_error) {
      showToast("复制失败");
    }
  });

  elements.query.value = "Caroline 的彩色玻璃窗想表达什么？";
  setPipeline("idle");
  checkHealth();
})();
