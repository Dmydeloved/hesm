(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = { items: [], selectedKey: "", selectedType: "", selectedId: "" };
  const labels = {
    chat: "Chat",
    retriever: "Retriever",
    add: "Add",
    segment_update: "Segment 蒸馏",
    experience_update: "Experience 蒸馏",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function request(url) {
    const response = await fetch(url);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  function operationKey(item) {
    return `${item.type}:${item.id}`;
  }

  function visibleItems() {
    const type = $("#type").value;
    const status = $("#status").value;
    return state.items.filter(
      (item) =>
        (!type || item.type === type) &&
        (!status || item.status === status),
    );
  }

  function render() {
    const items = visibleItems();
    $("#operation-count").textContent = `${items.length} 条记录`;
    $("#items").innerHTML = items.length
      ? items
          .map((item) => {
            const key = operationKey(item);
            return `<article class="operation-item ${
              key === state.selectedKey ? "active" : ""
            }" data-type="${escapeHtml(item.type)}" data-id="${escapeHtml(
              item.id,
            )}">
              <strong>${escapeHtml(labels[item.type] || item.type)}</strong>
              <code>${escapeHtml(item.id)}</code>
              <span class="badge ${escapeHtml(item.status)}">${escapeHtml(
                item.status,
              )}</span>
              <small>${escapeHtml(item.updated_at || item.created_at || "")}</small>
              <time>${Number(item.elapsed_ms || 0).toFixed(1)} ms</time>
            </article>`;
          })
          .join("")
      : '<p class="empty">暂无符合条件的操作记录</p>';
  }

  function block(title, value, raw = false) {
    if (value === undefined || value === null || value === "") return "";
    if (raw) {
      return `<section class="detail-section"><h3>${escapeHtml(
        title,
      )}</h3><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></section>`;
    }
    const entries = Object.entries(value || {})
      .map(([key, item]) => {
        const rendered =
          typeof item === "object" ? JSON.stringify(item, null, 2) : item;
        return `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(rendered)}</dd>`;
      })
      .join("");
    return `<section class="detail-section"><h3>${escapeHtml(
      title,
    )}</h3><dl>${entries}</dl></section>`;
  }

  async function loadDetail(type, id) {
    state.selectedType = type;
    state.selectedId = id;
    state.selectedKey = `${type}:${id}`;
    render();
    $("#detail-kind").textContent = labels[type] || type;
    $("#detail").dataset.loaded = "";
    $("#detail").innerHTML = '<p class="empty">正在加载详情…</p>';
    try {
      const data = await request(
        `/api/operations/${encodeURIComponent(type)}/${encodeURIComponent(id)}`,
      );
      const operation = data.operation || {};
      let html = block("操作信息", operation);
      if (type === "chat") {
        html += block("Retriever", data.retriever);
        html += block("Add", data.add);
        html += block("Segment 蒸馏", data.segment_updates, true);
        html += block("Experience 蒸馏", data.experience_updates, true);
      } else if (type === "add") {
        html += block("Segment 蒸馏", data.segment_updates, true);
        html += block("Experience 蒸馏", data.experience_updates, true);
      }
      $("#detail").innerHTML = html || '<p class="empty">暂无详情</p>';
      $("#detail").dataset.loaded = "true";
    } catch (error) {
      $("#detail").innerHTML = `<p class="empty">加载失败：${escapeHtml(
        error.message,
      )}</p>`;
    }
  }

  async function load() {
    const query = new URLSearchParams({ limit: "300" });
    if ($("#type").value) query.set("type", $("#type").value);
    if ($("#status").value) query.set("status", $("#status").value);
    try {
      state.items = (await request(`/api/operations?${query}`)).items || [];
      render();
      const items = visibleItems();
      const selected = items.find((item) => operationKey(item) === state.selectedKey);
      const next = selected || items[0];
      if (next) {
        const key = operationKey(next);
        if (key !== state.selectedKey || !$("#detail").dataset.loaded) {
          await loadDetail(next.type, next.id);
        }
      } else {
        state.selectedKey = "";
        $("#detail-kind").textContent = "—";
        $("#detail").innerHTML = '<p class="empty">暂无操作记录。发送消息或执行添加后，记录会自动出现在这里。</p>';
      }
    } catch (error) {
      $("#items").innerHTML = `<p class="empty">加载失败：${escapeHtml(
        error.message,
      )}</p>`;
    }
  }

  $("#items").addEventListener("click", (event) => {
    const item = event.target.closest(".operation-item");
    if (item) loadDetail(item.dataset.type, item.dataset.id);
  });
  $("#type").addEventListener("change", load);
  $("#status").addEventListener("change", load);
  $("#refresh").addEventListener("click", load);
  setInterval(load, 1500);
  load();
})();
