(function () {
  "use strict";

  const state = {
    level: "experience",
    page: 1,
    pageSize: 30,
    total: 0,
    pages: 1,
    search: "",
    status: "",
    parentId: "",
    experienceId: "",
    parentLabel: "",
  };

  const $ = (selector) => document.querySelector(selector);
  const esc = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const statusOptions = {
    experience: [["", "全部状态"], ["open", "进行中"], ["completed", "已完成"], ["deleted", "已删除"]],
    segment: [["", "全部状态"], ["open", "开放"], ["completed", "已完成"], ["deleted", "已删除"]],
    qa: [["", "全部状态"], ["open", "有效"], ["deleted", "已删除"]],
  };
  const statusLabel = { open: "进行中", completed: "已完成", deleted: "已删除" };
  const fields = {
    experience: ["顺序", "主题", "核心实体", "Experience ID", "状态", "Segment", "QA", "创建时间", "更新时间", "操作"],
    segment: ["顺序", "主题", "核心实体", "意图", "Segment ID", "Experience ID", "状态", "QA", "创建时间", "更新时间", "操作"],
    qa: ["顺序", "主题", "核心实体", "意图", "QA ID", "用户输入", "助手输出", "Segment ID", "状态", "置信度", "时间", "操作"],
  };
  const els = {
    health: $("#health"),
    stats: $("#stats"),
    tabs: $("#level-tabs"),
    search: $("#search"),
    status: $("#status"),
    head: $("#table-head"),
    body: $("#table-body"),
    empty: $("#empty"),
    info: $("#page-info"),
    prev: $("#prev"),
    next: $("#next"),
    context: $("#context-bar"),
    toast: $("#toast"),
  };
  let searchTimer;

  function toast(message) {
    els.toast.textContent = message;
    els.toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => els.toast.classList.remove("show"), 1800);
  }

  function setHealth(ok, label) {
    els.health.className = `health ${ok ? "ready" : "error"}`;
    els.health.querySelector("span").textContent = label;
  }

  async function api(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  async function loadStats() {
    try {
      const data = await api("/api/stats");
      ["experience", "segment", "qa"].forEach((key, index) => {
        els.stats.children[index].querySelector("strong").textContent = Number(data.counts[key] || 0).toLocaleString();
      });
      setHealth(true, "在线数据库");
    } catch (error) {
      setHealth(false, "服务不可用");
      toast(error.message);
    }
  }

  function configureStatus() {
    els.status.innerHTML = statusOptions[state.level]
      .map(([value, label]) => `<option value="${value}">${label}</option>`)
      .join("");
    state.status = "";
  }

  function renderHead() {
    els.head.closest("table").dataset.level = state.level;
    els.head.innerHTML = `<tr>${fields[state.level].map((value) => `<th>${value}</th>`).join("")}</tr>`;
  }

  function pill(status) {
    return `<span class="status-pill ${esc(status)}">${esc(statusLabel[status] || status || "未知")}</span>`;
  }

  function idCell(value) {
    return `<code class="memory-id" title="${esc(value)}">${esc(value || "—")}</code>`;
  }

  function textCell(value, fallback = "—") {
    return `<span class="cell-text" title="${esc(value || fallback)}">${esc(value || fallback)}</span>`;
  }

  function detailButton(level, id) {
    const url = `/detail.html?level=${encodeURIComponent(level)}&id=${encodeURIComponent(id)}`;
    return `<a class="row-action" href="${url}">详情</a>`;
  }

  function experienceRow(item) {
    const status = item.state?.status || "open";
    return `<tr>
      <td class="sequence">${item.sequence}</td>
      <td><strong>${esc(item.topic || "未命名主题")}</strong></td>
      <td>${textCell(item.core_entity)}</td>
      <td>${idCell(item.experience_id)}</td>
      <td>${pill(status)}</td>
      <td><button class="count-link" data-drill="segment" data-parent="${esc(item.experience_id)}" data-label="${esc(item.topic)}">${item.segment_count}</button></td>
      <td><button class="count-link" data-drill="qa" data-experience="${esc(item.experience_id)}" data-label="${esc(item.topic)}">${item.qa_count}</button></td>
      <td>${textCell(item.created_at)}</td>
      <td>${textCell(item.updated_at)}</td>
      <td>${detailButton("experience", item.experience_id)}</td>
    </tr>`;
  }

  function segmentRow(item) {
    return `<tr>
      <td class="sequence">${item.sequence}</td>
      <td><strong>${esc(item.topic || "未命名主题")}</strong></td>
      <td>${textCell(item.core_entity)}</td>
      <td>${textCell(item.intent, "未命名阶段")}</td>
      <td>${idCell(item.segment_id)}</td>
      <td>${idCell(item.experience_id)}</td>
      <td>${pill(item.status)}</td>
      <td><button class="count-link" data-drill="qa" data-parent="${esc(item.segment_id)}" data-label="${esc(item.intent)}">${item.qa_count}</button></td>
      <td>${textCell(item.created_at)}</td>
      <td>${textCell(item.updated_at)}</td>
      <td>${detailButton("segment", item.segment_id)}</td>
    </tr>`;
  }

  function qaRow(item) {
    return `<tr>
      <td class="sequence">${item.sequence}</td>
      <td><strong>${esc(item.topic || "未命名主题")}</strong></td>
      <td>${textCell(item.core_entity)}</td>
      <td>${textCell(item.intent)}</td>
      <td>${idCell(item.qa_id)}</td>
      <td>${textCell(item.user_input, "空输入")}</td>
      <td>${textCell(item.assistant_output, "无助手输出")}</td>
      <td>${idCell(item.segment_id)}</td>
      <td>${pill(item.status)}</td>
      <td>${Math.round(Number(item.confidence || 0) * 100)}%</td>
      <td>${textCell(item.timestamp)}</td>
      <td>${detailButton("qa", item.qa_id)}</td>
    </tr>`;
  }

  function renderContext() {
    const hasScope = Boolean(state.parentId || state.experienceId);
    els.context.hidden = !hasScope;
    if (!hasScope) return;
    const scopeName = state.experienceId ? "Experience 下全部 QA" : `下层 ${state.level}`;
    els.context.innerHTML = `正在查看 <b>${esc(state.parentLabel)}</b> 的 ${scopeName}<button data-clear-parent>清除范围</button>`;
  }

  async function loadList() {
    renderHead();
    const columnCount = fields[state.level].length;
    els.body.innerHTML = `<tr><td colspan="${columnCount}"><p>正在加载…</p></td></tr>`;
    const params = new URLSearchParams({ page: state.page, page_size: state.pageSize });
    if (state.search) params.set("q", state.search);
    if (state.status) params.set("status", state.status);
    if (state.parentId) params.set("parent_id", state.parentId);
    if (state.experienceId) params.set("experience_id", state.experienceId);
    try {
      const data = await api(`/api/${state.level}?${params}`);
      Object.assign(state, { total: data.total, pages: data.pages });
      const render = { experience: experienceRow, segment: segmentRow, qa: qaRow }[state.level];
      els.body.innerHTML = data.items.map(render).join("");
      els.empty.hidden = data.items.length > 0;
      els.info.textContent = `${data.page} / ${data.pages} · ${data.total}`;
      els.prev.disabled = state.page <= 1;
      els.next.disabled = state.page >= data.pages;
      renderContext();
    } catch (error) {
      els.body.innerHTML = "";
      els.empty.hidden = false;
      els.empty.querySelector("p").textContent = error.message;
    }
  }

  function switchLevel(level, scope = {}) {
    state.level = level;
    state.page = 1;
    state.parentId = scope.parentId || "";
    state.experienceId = scope.experienceId || "";
    state.parentLabel = scope.parentLabel || "";
    state.status = "";
    configureStatus();
    els.tabs.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.level === level);
    });
    loadList();
  }

  els.tabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-level]");
    if (button) switchLevel(button.dataset.level);
  });
  els.search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = els.search.value.trim();
      state.page = 1;
      loadList();
    }, 280);
  });
  els.status.addEventListener("change", () => {
    state.status = els.status.value;
    state.page = 1;
    loadList();
  });
  els.body.addEventListener("click", (event) => {
    const drill = event.target.closest("[data-drill]");
    if (!drill) return;
    switchLevel(drill.dataset.drill, {
      parentId: drill.dataset.parent,
      experienceId: drill.dataset.experience,
      parentLabel: drill.dataset.label,
    });
  });
  els.context.addEventListener("click", (event) => {
    if (event.target.closest("[data-clear-parent]")) switchLevel(state.level);
  });
  $("#refresh").addEventListener("click", () => {
    loadStats();
    loadList();
  });
  els.prev.addEventListener("click", () => {
    if (state.page > 1) {
      state.page -= 1;
      loadList();
    }
  });
  els.next.addEventListener("click", () => {
    if (state.page < state.pages) {
      state.page += 1;
      loadList();
    }
  });

  configureStatus();
  loadStats();
  loadList();
})();
