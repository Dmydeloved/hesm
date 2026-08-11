(function () {
  "use strict";

  const data = window.HESM_MEMORY_DATA;
  const runtime = data.meta.runtime || {};
  const state = {
    experienceId:
      runtime.current_experience_id || data.experiences[0]?.id || "",
    segmentId: runtime.current_segment_id || "",
    experienceSearch: "",
    experienceFilter: "all",
    segmentSearch: "",
    qaSearch: "",
  };

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    experienceTotal: $("#experience-total"),
    experienceSearch: $("#experience-search"),
    experienceList: $("#experience-list"),
    experienceHeader: $("#experience-header"),
    breadcrumbTopic: $("#breadcrumb-topic"),
    globalExperienceCount: $("#global-experience-count"),
    globalSegmentCount: $("#global-segment-count"),
    globalQaCount: $("#global-qa-count"),
    experienceState: $("#experience-state"),
    intentList: $("#intent-list"),
    intentCount: $("#intent-count"),
    segmentSearch: $("#segment-search"),
    segmentCount: $("#segment-count"),
    segmentList: $("#segment-list"),
    segmentDetail: $("#segment-detail"),
    qaSearch: $("#qa-search"),
    qaCount: $("#qa-count"),
    qaList: $("#qa-list"),
    runtimeJump: $("#runtime-jump"),
    dialog: $("#qa-dialog"),
    dialogClose: $("#dialog-close"),
    dialogContent: $("#qa-dialog-content"),
    toast: $("#toast"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function normalized(value) {
    return String(value ?? "").toLocaleLowerCase().trim();
  }

  function shortId(id, length = 8) {
    if (!id) return "—";
    const [prefix, suffix = ""] = id.split("_");
    return `${prefix}_${suffix.slice(0, length)}${suffix.length > length ? "…" : ""}`;
  }

  function currentExperience() {
    return data.experiences.find((item) => item.id === state.experienceId);
  }

  function currentSegment() {
    const experience = currentExperience();
    return experience?.segments.find((item) => item.id === state.segmentId);
  }

  function experienceQaCount(experience) {
    return experience.segments.reduce((total, segment) => total + segment.qas.length, 0);
  }

  function isCurrentExperience(experience) {
    return experience.id === runtime.current_experience_id;
  }

  function isCurrentSegment(segment) {
    return segment.id === runtime.current_segment_id;
  }

  function selectExperience(id, preferredSegmentId) {
    const experience = data.experiences.find((item) => item.id === id);
    if (!experience) return;
    state.experienceId = id;
    const preferred = experience.segments.find(
      (item) => item.id === preferredSegmentId,
    );
    const experienceCurrent = experience.segments.find(
      (item) => item.id === experience.state?.current_segment_id,
    );
    state.segmentId = (preferred || experienceCurrent || experience.segments[0])?.id || "";
    state.segmentSearch = "";
    state.qaSearch = "";
    elements.segmentSearch.value = "";
    elements.qaSearch.value = "";
    renderAll();
  }

  function filteredExperiences() {
    const query = normalized(state.experienceSearch);
    return data.experiences.filter((experience) => {
      if (state.experienceFilter === "current" && !isCurrentExperience(experience)) {
        return false;
      }
      if (state.experienceFilter === "has-summary" && !String(experience.summary).trim()) {
        return false;
      }
      if (!query) return true;
      const content = [
        experience.id,
        experience.topic,
        experience.coreEntity,
        experience.summary,
        ...experience.intents,
      ]
        .join(" ")
        .toLocaleLowerCase();
      return content.includes(query);
    }).sort((left, right) => {
      if (left.id === state.experienceId) return -1;
      if (right.id === state.experienceId) return 1;
      if (isCurrentExperience(left)) return -1;
      if (isCurrentExperience(right)) return 1;
      return 0;
    });
  }

  function renderExperienceList() {
    const experiences = filteredExperiences();
    elements.experienceTotal.textContent = `${experiences.length}/${data.meta.experienceCount}`;
    if (!experiences.length) {
      elements.experienceList.innerHTML = emptyState(
        "没有匹配的 Experience",
        "请调整关键词或筛选条件",
      );
      return;
    }

    elements.experienceList.innerHTML = experiences
      .map((experience) => {
        const active = experience.id === state.experienceId;
        const current = isCurrentExperience(experience);
        return `
          <button class="experience-item ${active ? "active" : ""}" type="button" data-experience-id="${escapeHtml(experience.id)}">
            <span class="experience-glyph">E</span>
            <span class="experience-item-body">
              <span class="experience-item-title">
                <strong>${escapeHtml(experience.topic)}</strong>
                ${current ? '<i class="current-dot" title="Runtime 当前 Experience"></i>' : ""}
              </span>
              <small>${escapeHtml(experience.coreEntity)}</small>
              <span class="experience-item-meta">
                <span>${experience.segments.length} Segment</span>
                <span>${experienceQaCount(experience)} QA</span>
                <code>v${experience.version}</code>
              </span>
            </span>
            <span class="item-arrow">›</span>
          </button>`;
      })
      .join("");
  }

  function renderExperienceHeader(experience) {
    const status = experience.state?.status || "unknown";
    const current = isCurrentExperience(experience);
    elements.breadcrumbTopic.textContent = experience.topic;
    elements.experienceHeader.innerHTML = `
      <div class="experience-title-block">
        <span class="large-glyph">E</span>
        <div>
          <div class="title-badges">
            <span class="status-badge ${escapeHtml(status)}"><i></i>${escapeHtml(status.replaceAll("_", " "))}</span>
            ${current ? '<span class="runtime-badge">RUNTIME CURRENT</span>' : ""}
          </div>
          <h1>${escapeHtml(experience.topic)}</h1>
          <div class="id-line">
            <code>${escapeHtml(experience.id)}</code>
            <button type="button" data-copy="${escapeHtml(experience.id)}" title="复制 Experience ID">复制 ID</button>
          </div>
        </div>
      </div>
      <div class="experience-kpis">
        <div><strong>${experience.segments.length}</strong><span>Segments</span></div>
        <i></i>
        <div><strong>${experienceQaCount(experience)}</strong><span>QA memories</span></div>
        <i></i>
        <div><strong>v${experience.version}</strong><span>Version</span></div>
      </div>`;
  }

  function renderState(experience) {
    const currentId = experience.state?.current_segment_id || "—";
    const rows = [
      ["核心实体", experience.coreEntity],
      ["当前状态", experience.state?.status || "unknown"],
      ["当前 Segment", shortId(currentId)],
      ["已总结 Segment", `${experience.lastSummarizedSegmentCount}/${experience.segments.length}`],
      ["创建时间", experience.createdAt],
      ["更新时间", experience.updatedAt],
    ];
    elements.experienceState.innerHTML = rows
      .map(
        ([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`,
      )
      .join("");
  }

  function renderIntents(experience) {
    elements.intentCount.textContent = `${experience.intents.length} 个意图`;
    if (!experience.intents.length) {
      elements.intentList.innerHTML = '<span class="muted-copy">暂无关联意图</span>';
      return;
    }
    elements.intentList.innerHTML = experience.intents
      .map(
        (intent, index) => `
          <span class="intent-chip"><b>${String(index + 1).padStart(2, "0")}</b>${escapeHtml(intent)}</span>`,
      )
      .join('<span class="intent-link">→</span>');
  }

  function filteredSegments(experience) {
    const query = normalized(state.segmentSearch);
    if (!query) return experience.segments;
    return experience.segments.filter((segment) =>
      [segment.id, segment.topic, segment.intent, segment.coreEntity, segment.summary]
        .join(" ")
        .toLocaleLowerCase()
        .includes(query),
    );
  }

  function renderSegments(experience) {
    const segments = filteredSegments(experience);
    elements.segmentCount.textContent = `${segments.length}/${experience.segments.length} 个 Segment`;
    if (!segments.length) {
      elements.segmentList.innerHTML = emptyState("没有匹配的 Segment", "请更换筛选关键词");
      return;
    }

    elements.segmentList.innerHTML = segments
      .map((segment) => {
        const sequence = experience.segments.findIndex((item) => item.id === segment.id) + 1;
        const active = segment.id === state.segmentId;
        const current = isCurrentSegment(segment) || experience.state?.current_segment_id === segment.id;
        return `
          <button class="segment-row ${active ? "active" : ""}" type="button" data-segment-id="${escapeHtml(segment.id)}">
            <span class="segment-sequence"><i></i>${String(sequence).padStart(2, "0")}</span>
            <span class="segment-main">
              <strong>${escapeHtml(segment.intent)}</strong>
              <small>${escapeHtml(segment.topic)} · ${escapeHtml(shortId(segment.id, 10))}</small>
            </span>
            <span><i class="open-status"></i>${escapeHtml(segment.status)}</span>
            <span class="qa-number">${segment.qas.length}</span>
            <span class="segment-time">${escapeHtml(segment.updatedAt)}</span>
            <span class="row-end">${current ? '<b class="current-tag">当前</b>' : ""}<i>›</i></span>
          </button>`;
      })
      .join("");
  }

  function renderSegmentDetail(segment) {
    if (!segment) {
      elements.segmentDetail.innerHTML = emptyState("没有 Segment", "当前 Experience 中暂无阶段记忆");
      return;
    }
    const current = isCurrentSegment(segment);
    elements.segmentDetail.innerHTML = `
      <div class="detail-kicker"><span>SELECTED SEGMENT</span>${current ? '<b>RUNTIME CURRENT</b>' : ""}</div>
      <div class="detail-title">
        <span class="segment-glyph">S</span>
        <div><small>${escapeHtml(segment.topic)}</small><h2>${escapeHtml(segment.intent)}</h2></div>
      </div>
      <div class="detail-id"><code>${escapeHtml(segment.id)}</code><button type="button" data-copy="${escapeHtml(segment.id)}">复制</button></div>
      <div class="detail-meta">
        <span><small>STATUS</small><strong><i></i>${escapeHtml(segment.status)}</strong></span>
        <span><small>VERSION</small><strong>v${segment.version}</strong></span>
        <span><small>QA COUNT</small><strong>${segment.qas.length}</strong></span>
      </div>
      <div class="segment-summary">
        <div class="subheading"><span>阶段总结</span><small>${segment.lastSummarizedQaCount}/${segment.qas.length} 已总结</small></div>
        ${segment.summary ? `<p>${escapeHtml(segment.summary).replaceAll("\n", "<br>")}</p>` : '<p class="muted-copy">该 Segment 暂无阶段总结。</p>'}
      </div>
      <dl class="detail-dates">
        <div><dt>创建</dt><dd>${escapeHtml(segment.createdAt)}</dd></div>
        <div><dt>更新</dt><dd>${escapeHtml(segment.updatedAt)}</dd></div>
      </dl>`;
  }

  function filteredQas(segment) {
    const query = normalized(state.qaSearch);
    if (!segment || !query) return segment?.qas || [];
    return segment.qas.filter((qa) =>
      [
        qa.id,
        qa.userInput,
        qa.assistantOutput,
        qa.topic,
        qa.intent,
        qa.coreEntity,
        qa.reasoning,
        ...qa.entities,
      ]
        .join(" ")
        .toLocaleLowerCase()
        .includes(query),
    );
  }

  function speakerFrom(text) {
    const match = String(text).match(/^\[([^\]]+)\]/);
    return match?.[1] || "User";
  }

  function renderQas(segment) {
    const qas = filteredQas(segment);
    elements.qaCount.textContent = `${qas.length}/${segment?.qas.length || 0}`;
    if (!qas.length) {
      elements.qaList.innerHTML = emptyState(
        segment?.qas.length ? "没有匹配的 QA" : "暂无 QA 记忆",
        segment?.qas.length ? "尝试搜索其他内容或标签" : "该 Segment 尚未写入原始记忆",
      );
      return;
    }

    elements.qaList.innerHTML = qas
      .map(
        (qa, index) => `
          <button class="qa-card" type="button" data-qa-id="${escapeHtml(qa.id)}">
            <span class="qa-index">Q${String(index + 1).padStart(2, "0")}</span>
            <span class="qa-card-content">
              <span class="qa-card-head"><strong>${escapeHtml(speakerFrom(qa.userInput))}</strong><time>${escapeHtml(qa.timestamp)}</time></span>
              <span class="qa-excerpt">${escapeHtml(qa.userInput)}</span>
              <span class="qa-tags">
                ${qa.entities.slice(0, 3).map((entity) => `<i>${escapeHtml(entity)}</i>`).join("")}
                ${qa.entities.length > 3 ? `<i>+${qa.entities.length - 3}</i>` : ""}
              </span>
              <span class="qa-card-foot"><code>${escapeHtml(shortId(qa.id, 10))}</code><span>confidence ${(qa.confidence * 100).toFixed(0)}%</span></span>
            </span>
            <span class="qa-open">↗</span>
          </button>`,
      )
      .join("");
  }

  function renderAll() {
    const experience = currentExperience();
    if (!experience) return;
    if (!experience.segments.some((item) => item.id === state.segmentId)) {
      state.segmentId = experience.segments[0]?.id || "";
    }
    const segment = currentSegment();
    elements.globalExperienceCount.textContent = data.meta.experienceCount;
    elements.globalSegmentCount.textContent = data.meta.segmentCount;
    elements.globalQaCount.textContent = data.meta.qaCount;
    renderExperienceList();
    renderExperienceHeader(experience);
    renderState(experience);
    renderIntents(experience);
    renderSegments(experience);
    renderSegmentDetail(segment);
    renderQas(segment);
  }

  function emptyState(title, description) {
    return `<div class="empty-state"><span>◇</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p></div>`;
  }

  function showQaDialog(qa) {
    const tools = qa.tools && Object.keys(qa.tools).length
      ? `<pre>${escapeHtml(JSON.stringify(qa.tools, null, 2))}</pre>`
      : '<p class="muted-copy">无工具调用记录</p>';
    elements.dialogContent.innerHTML = `
      <div class="dialog-identity">
        <span class="qa-dialog-glyph">Q</span>
        <div><code>${escapeHtml(qa.id)}</code><strong>${escapeHtml(qa.topic)} · ${escapeHtml(qa.intent)}</strong><small>${escapeHtml(qa.timestamp)}</small></div>
        <span class="confidence-ring" style="--confidence:${qa.confidence * 360}deg"><b>${(qa.confidence * 100).toFixed(0)}%</b><small>置信度</small></span>
      </div>
      <dl class="dialog-meta">
        <div><dt>核心实体</dt><dd>${escapeHtml(qa.coreEntity)}</dd></div>
        <div><dt>状态</dt><dd>${escapeHtml(qa.status)}</dd></div>
        <div><dt>Topic</dt><dd>${escapeHtml(qa.topic)}</dd></div>
        <div><dt>Intent</dt><dd>${escapeHtml(qa.intent)}</dd></div>
      </dl>
      <section class="dialog-section"><h3>用户输入</h3><div class="conversation user-message">${escapeHtml(qa.userInput)}</div></section>
      <section class="dialog-section"><h3>助手输出</h3><div class="conversation assistant-message">${qa.assistantOutput ? escapeHtml(qa.assistantOutput) : '<span class="muted-copy">无助手输出</span>'}</div></section>
      <section class="dialog-section"><h3>抽取实体</h3><div class="dialog-tags">${qa.entities.map((entity) => `<span>${escapeHtml(entity)}</span>`).join("") || '<span class="muted-copy">无实体标签</span>'}</div></section>
      <section class="dialog-section"><h3>主题判断依据</h3><p class="reasoning-copy">${escapeHtml(qa.reasoning || "无判断依据")}</p></section>
      <details class="tool-details"><summary>工具调用数据</summary>${tools}</details>`;
    elements.dialog.showModal();
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(
      () => elements.toast.classList.remove("show"),
      1600,
    );
  }

  async function copyValue(value) {
    try {
      await navigator.clipboard.writeText(value);
      showToast("ID 已复制");
    } catch (_error) {
      showToast("当前浏览器不允许复制，请通过本地 HTTP 服务打开");
    }
  }

  elements.experienceSearch.addEventListener("input", () => {
    state.experienceSearch = elements.experienceSearch.value;
    renderExperienceList();
  });

  document.querySelector(".filter-row").addEventListener("click", (event) => {
    const button = event.target.closest("[data-exp-filter]");
    if (!button) return;
    state.experienceFilter = button.dataset.expFilter;
    document.querySelectorAll("[data-exp-filter]").forEach((item) =>
      item.classList.toggle("active", item === button),
    );
    renderExperienceList();
  });

  elements.experienceList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-experience-id]");
    if (button) selectExperience(button.dataset.experienceId);
  });

  elements.segmentSearch.addEventListener("input", () => {
    state.segmentSearch = elements.segmentSearch.value;
    renderSegments(currentExperience());
  });

  elements.segmentList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-segment-id]");
    if (!button) return;
    state.segmentId = button.dataset.segmentId;
    state.qaSearch = "";
    elements.qaSearch.value = "";
    renderSegments(currentExperience());
    renderSegmentDetail(currentSegment());
    renderQas(currentSegment());
  });

  elements.qaSearch.addEventListener("input", () => {
    state.qaSearch = elements.qaSearch.value;
    renderQas(currentSegment());
  });

  elements.qaList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-qa-id]");
    if (!button) return;
    const qa = currentSegment()?.qas.find((item) => item.id === button.dataset.qaId);
    if (qa) showQaDialog(qa);
  });

  elements.runtimeJump.addEventListener("click", () => {
    selectExperience(runtime.current_experience_id, runtime.current_segment_id);
    showToast("已定位到 Runtime 当前记忆");
  });

  document.body.addEventListener("click", (event) => {
    const button = event.target.closest("[data-copy]");
    if (button) copyValue(button.dataset.copy);
  });

  elements.dialogClose.addEventListener("click", () => elements.dialog.close());
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      elements.experienceSearch.focus();
    }
  });

  selectExperience(state.experienceId, state.segmentId);
})();
