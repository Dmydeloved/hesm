(function () {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const level = params.get("level") || "";
  const memoryId = params.get("id") || "";
  const allowedLevels = new Set(["experience", "segment", "qa"]);
  const statusLabel = { open: "进行中", completed: "已完成", deleted: "已删除" };
  const $ = (selector) => document.querySelector(selector);
  const esc = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function toast(message) {
    const element = $("#toast");
    element.textContent = message;
    element.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => element.classList.remove("show"), 1800);
  }

  async function api(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  function statusPill(status) {
    return `<span class="status-pill ${esc(status)}">${esc(statusLabel[status] || status || "未知")}</span>`;
  }

  function facts(rows) {
    return `<div class="node-facts">${rows.map(([label, value, mono]) => `
      <div class="node-fact">
        <small>${esc(label)}</small>
        <strong class="${mono ? "mono" : ""}">${esc(value || "—")}</strong>
      </div>`).join("")}</div>`;
  }

  function payloadSection(title, value) {
    const hasValue = value && (typeof value !== "object" || Object.keys(value).length > 0);
    const content = hasValue
      ? (typeof value === "string" ? value : JSON.stringify(value, null, 2))
      : "暂无内容";
    return `<section class="node-section"><h3>${esc(title)}</h3><pre>${esc(content)}</pre></section>`;
  }

  function statusEditor(levelName, id, status) {
    const options = levelName === "qa"
      ? [["open", "有效"], ["deleted", "已删除"]]
      : [["open", "进行中"], ["completed", "已完成"], ["deleted", "已删除"]];
    return `<div class="node-status-editor">
      <select aria-label="状态">${options.map(([value, label]) => `<option value="${value}" ${value === status ? "selected" : ""}>${label}</option>`).join("")}</select>
      <button type="button" data-save-status="${esc(levelName)}" data-id="${esc(id)}">保存状态</button>
    </div>`;
  }

  function qaNode(item, sequence) {
    return `<details class="tree-node qa-node">
      <summary>
        <span class="node-toggle"></span>
        <span class="node-badge qa">Q</span>
        <span class="node-sequence">${sequence}</span>
        <span class="node-heading"><strong>${esc(item.user_input || "空输入")}</strong><small>${esc(item.timestamp || "—")}</small></span>
        ${statusPill(item.status)}
      </summary>
      <div class="node-content">
        ${facts([
          ["主题", item.topic],
          ["核心实体", item.core_entity],
          ["意图", item.intent],
          ["QA ID", item.qa_id, true],
          ["Segment ID", item.segment_id, true],
          ["时间", item.timestamp],
          ["置信度", `${Math.round(Number(item.confidence || 0) * 100)}%`],
        ])}
        ${payloadSection("用户输入", item.user_input)}
        ${payloadSection("助手输出", item.assistant_output)}
        ${payloadSection("实体", item.entities)}
        ${payloadSection("主题判断依据", item.reasoning)}
        ${payloadSection("工具调用链", item.tools)}
        ${statusEditor("qa", item.qa_id, item.status)}
      </div>
    </details>`;
  }

  function segmentNode(item, sequence, open = false) {
    const qas = item.qas || [];
    return `<details class="tree-node segment-node" ${open ? "open" : ""}>
      <summary>
        <span class="node-toggle"></span>
        <span class="node-badge segment">S</span>
        <span class="node-sequence">${sequence}</span>
        <span class="node-heading"><strong>${esc(item.intent || "未命名阶段")}</strong><small>${esc(item.created_at || "—")}</small></span>
        <span class="node-count"><small>QA</small><b>${qas.length}</b></span>
        ${statusPill(item.status)}
      </summary>
      <div class="node-content">
        ${facts([
          ["主题", item.topic],
          ["核心实体", item.core_entity],
          ["意图", item.intent],
          ["Segment ID", item.segment_id, true],
          ["Experience ID", item.experience_id, true],
          ["创建时间", item.created_at],
          ["更新时间", item.updated_at],
        ])}
        ${payloadSection("Segment 总结", item.summary)}
        ${statusEditor("segment", item.segment_id, item.status)}
        <div class="node-children qa-children">
          ${qas.map((qa, index) => qaNode(qa, qa.sequence || index + 1)).join("") || "<p class=\"no-children\">暂无 QA</p>"}
        </div>
      </div>
    </details>`;
  }

  function experienceNode(item) {
    const segments = item.segments || [];
    const qaCount = segments.reduce((total, segment) => total + (segment.qas || []).length, 0);
    const status = item.state?.status || item.status || "open";
    return `<details class="tree-node experience-node" open>
      <summary>
        <span class="node-toggle"></span>
        <span class="node-badge experience">E</span>
        <span class="node-heading"><strong>${esc(item.topic || "未命名主题")}</strong><small>${esc(item.updated_at || "—")}</small></span>
        <span class="node-count"><small>Segment</small><b>${segments.length}</b></span>
        <span class="node-count"><small>QA</small><b>${qaCount}</b></span>
        ${statusPill(status)}
      </summary>
      <div class="node-content">
        ${facts([
          ["主题", item.topic],
          ["核心实体", item.core_entity],
          ["Experience ID", item.experience_id, true],
          ["创建时间", item.created_at],
          ["更新时间", item.updated_at],
          ["版本", item.version],
        ])}
        ${payloadSection("意图链", item.intents)}
        ${payloadSection("Experience 总结", item.summary)}
        ${payloadSection("历史经验", item.history_experience)}
        ${statusEditor("experience", item.experience_id, status)}
        <div class="node-children segment-children">
          ${segments.map((segment, index) => segmentNode(segment, segment.sequence || index + 1)).join("") || "<p class=\"no-children\">暂无 Segment</p>"}
        </div>
      </div>
    </details>`;
  }

  function renderTree(item) {
    let tree;
    if (level === "experience") tree = experienceNode(item);
    else if (level === "segment") tree = segmentNode(item, item.sequence || 1, true);
    else tree = qaNode(item, item.sequence || 1);
    $("#tree-panel").innerHTML = `<div class="mindmap">${tree}</div>`;
  }

  async function loadDetail() {
    if (!allowedLevels.has(level) || !memoryId) {
      $("#tree-panel").innerHTML = '<div class="tree-error">详情地址缺少有效的 level 或 id。</div>';
      return;
    }
    try {
      const item = await api(`/api/${level}/${encodeURIComponent(memoryId)}`);
      const title = level === "experience"
        ? item.topic
        : level === "segment"
          ? item.intent
          : item.user_input;
      $("#detail-level").textContent = `${level.toUpperCase()} MEMORY TREE`;
      $("#detail-title").textContent = title || `${level.toUpperCase()} 详情`;
      $("#health").className = "health ready";
      $("#health span").textContent = "数据已加载";
      renderTree(item);
    } catch (error) {
      $("#health").className = "health error";
      $("#health span").textContent = "加载失败";
      $("#tree-panel").innerHTML = `<div class="tree-error">${esc(error.message)}</div>`;
    }
  }

  $("#expand-all").addEventListener("click", () => {
    document.querySelectorAll(".tree-node").forEach((node) => { node.open = true; });
  });
  $("#collapse-all").addEventListener("click", () => {
    document.querySelectorAll(".tree-node").forEach((node) => { node.open = false; });
  });
  $("#tree-panel").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-save-status]");
    if (!button) return;
    const editor = button.closest(".node-status-editor");
    const status = editor.querySelector("select").value;
    try {
      await api(`/api/${button.dataset.saveStatus}/${encodeURIComponent(button.dataset.id)}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      toast("状态已更新");
      await loadDetail();
    } catch (error) {
      toast(error.message);
    }
  });

  loadDetail();
})();
