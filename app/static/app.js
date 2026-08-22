/* EDAgent frontend — talks to the Flask API in api.py. No framework,
   no build step: this is a small enough surface that vanilla JS keeps
   it easy to read end to end. */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const uploadSection = $("#upload-section");
  const caseSection = $("#case-section");
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const uploadError = $("#upload-error");
  const caseIdEl = $("#case-id");
  const overviewGrid = $("#overview-grid");
  const exhibitsGrid = $("#exhibits-grid");
  const newCaseBtn = $("#new-case-btn");
  const agenticBtn = $("#agentic-btn");
  const singleShotBtn = $("#single-shot-btn");
  const investigateStatus = $("#investigate-status");
  const evidenceLog = $("#evidence-log");
  const evidenceList = $("#evidence-list");
  const reportOutput = $("#report-output");
  const askForm = $("#ask-form");
  const askInput = $("#ask-input");
  const askBtn = $("#ask-btn");
  const qaList = $("#qa-list");

  let datasetId = null;

  // ---------------------------------------------------------------
  // fetch helper — the server can fail before ever reaching a route that
  // returns JSON (a 500 traceback page in debug mode, a 413 from a body
  // size limit, etc. are all plain HTML). Calling resp.json() directly on
  // those throws a cryptic "Unexpected token '<'" that hides the real
  // problem. This checks the content-type first and falls back to a
  // readable message from the response text instead.
  // ---------------------------------------------------------------

  async function parseJsonResponse(resp) {
    const contentType = resp.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return resp.json();
    }
    const text = await resp.text();
    throw new Error(`Server returned ${resp.status} (not JSON) — ${text.slice(0, 200).replace(/\s+/g, " ").trim()}`);
  }

  // ---------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  dropzone.setAttribute("tabindex", "0");
  dropzone.setAttribute("role", "button");

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    hideError();
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showError("That doesn't look like a CSV. Choose a .csv file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    dropzone.querySelector(".dropzone-title").textContent = "Reading the file…";
    try {
      const resp = await fetch("/datasets", { method: "POST", body: formData });
      const body = await parseJsonResponse(resp);
      if (!resp.ok) {
        showError(body.error || "Upload failed.");
        resetDropzoneLabel();
        return;
      }
      datasetId = body.dataset_id;
      openCase(body.profile);
    } catch (err) {
      showError("Couldn't reach the server: " + err.message);
      resetDropzoneLabel();
    }
  }

  function resetDropzoneLabel() {
    dropzone.querySelector(".dropzone-title").textContent = "Drop a CSV to open a case";
  }

  function showError(msg) {
    uploadError.textContent = msg;
    uploadError.hidden = false;
  }
  function hideError() {
    uploadError.hidden = true;
  }

  // ---------------------------------------------------------------
  // Case overview
  // ---------------------------------------------------------------

  function openCase(profile) {
    uploadSection.hidden = true;
    caseSection.hidden = false;
    caseIdEl.textContent = datasetId.slice(0, 8);

    renderOverview(profile);
    loadCharts();
  }

  function renderOverview(profile) {
    const cols = Object.values(profile.columns);
    const piiCount = cols.filter((c) => c.pii_flag).length;
    const flaggedCount = cols.filter(
      (c) => c.category_normalization_issues || c.duplicate_id_values ||
             c.mixed_numeric_text || c.unparseable_values
    ).length;
    const outlierCount = cols.filter((c) => c.outliers && c.outliers.count > 0).length;

    const stats = [
      { label: "Rows", value: profile.shape.rows.toLocaleString() },
      { label: "Columns", value: profile.shape.cols },
      { label: "Duplicate rows", value: profile.duplicate_rows, flaggable: true, flag: profile.duplicate_rows > 0 },
      { label: "PII columns", value: piiCount, flaggable: true, flag: piiCount > 0 },
      { label: "Quality flags", value: flaggedCount, flaggable: true, flag: flaggedCount > 0 },
      { label: "Columns w/ outliers", value: outlierCount, flaggable: true, flag: outlierCount > 0 },
    ];

    overviewGrid.innerHTML = stats.map((s) => `
      <div class="overview-stat ${s.flag ? "flagged" : ""} ${s.flaggable && !s.flag ? "verified" : ""}">
        <span class="stat-value">${s.value}</span>
        <span class="stat-label">${escapeHtml(s.label)}</span>
      </div>
    `).join("");
  }

  async function loadCharts() {
    exhibitsGrid.innerHTML = "";
    try {
      const resp = await fetch(`/datasets/${datasetId}/charts`);
      const body = await parseJsonResponse(resp);
      if (!resp.ok) return;

      Object.entries(body.charts).forEach(([name, spec]) => {
        const card = document.createElement("div");
        card.className = "exhibit";
        card.innerHTML = `<div class="exhibit-tag">${escapeHtml(name.replace(/_/g, " "))}</div><div class="exhibit-plot"></div>`;
        exhibitsGrid.appendChild(card);
        Plotly.newPlot(card.querySelector(".exhibit-plot"), spec.data, {
          ...spec.layout,
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#B8AC97", family: "IBM Plex Sans, sans-serif", size: 11 },
          margin: { t: 36, r: 16, b: 40, l: 48 },
        }, { displayModeBar: false, responsive: true });
      });
    } catch (err) {
      // Charts are a nice-to-have on top of the profile — a failure here
      // shouldn't block the rest of the case from working.
      console.error("Failed to load charts:", err);
    }
  }

  newCaseBtn.addEventListener("click", async () => {
    if (datasetId) {
      try { await fetch(`/datasets/${datasetId}`, { method: "DELETE" }); } catch { /* best effort */ }
    }
    datasetId = null;
    caseSection.hidden = true;
    uploadSection.hidden = false;
    resetDropzoneLabel();
    fileInput.value = "";
    evidenceLog.hidden = true;
    reportOutput.hidden = true;
    investigateStatus.hidden = true;
    qaList.innerHTML = "";
    caseIdEl.textContent = "—";
  });

  // ---------------------------------------------------------------
  // Investigation (report generation)
  // ---------------------------------------------------------------

  agenticBtn.addEventListener("click", () => runReport("agentic"));
  singleShotBtn.addEventListener("click", () => runReport("single_shot"));

  async function runReport(mode) {
    setButtonsDisabled(true);
    evidenceLog.hidden = true;
    reportOutput.hidden = true;

    const baseLabel = mode === "agentic" ? "Investigating — running its own queries against the data" : "Reading the case file";
    const startedAt = Date.now();
    setStatus(`${baseLabel}…`, "loading");
    // Ticking elapsed-time label: without this, a genuinely slow-but-
    // working request (a rate-limited free-tier API can legitimately take
    // several minutes — this has actually happened in production) looks
    // identical to a hung one. A visible, moving number is the cheapest
    // possible signal that something is still happening.
    const tickInterval = setInterval(() => {
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      let label = `${baseLabel}… (${elapsed}s)`;
      if (elapsed > 30) label += " — free-tier API limits can make this slow, still working";
      setStatus(label, "loading");
    }, 1000);

    try {
      const resp = await fetch(`/datasets/${datasetId}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const body = await parseJsonResponse(resp);
      if (!resp.ok) {
        setStatus(body.error || "Investigation failed.", "error");
        setButtonsDisabled(false);
        return;
      }

      hideStatus();
      if (body.tool_calls && body.tool_calls.length > 0) {
        renderEvidenceLog(body.tool_calls);
      }
      reportOutput.innerHTML = renderReportMarkdown(body.report);
      reportOutput.hidden = false;
    } catch (err) {
      setStatus("Couldn't reach the server: " + err.message, "error");
    } finally {
      clearInterval(tickInterval);
      setButtonsDisabled(false);
    }
  }

  function setButtonsDisabled(disabled) {
    agenticBtn.disabled = disabled;
    singleShotBtn.disabled = disabled;
  }

  function setStatus(text, kind) {
    investigateStatus.hidden = false;
    investigateStatus.className = "status-line" + (kind === "error" ? " error" : "");
    investigateStatus.innerHTML = (kind === "loading" ? '<span class="status-dot"></span>' : "") + escapeHtml(text);
  }
  function hideStatus() { investigateStatus.hidden = true; }

  function renderEvidenceLog(toolCalls) {
    evidenceList.innerHTML = toolCalls.map((tc, i) => `
      <li class="evidence-item">
        <div><span class="evidence-num">Exhibit ${i + 1} —</span><span class="evidence-query">${escapeHtml(tc.expression)}</span></div>
        <div class="evidence-result">${escapeHtml(String(tc.result).slice(0, 800))}</div>
      </li>
    `).join("");
    evidenceLog.hidden = false;
  }

  // ---------------------------------------------------------------
  // Interrogate (ask a question)
  // ---------------------------------------------------------------

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = askInput.value.trim();
    if (!question) return;

    askBtn.disabled = true;
    askInput.disabled = true;
    const pendingId = "qa-pending-" + Date.now();
    qaList.insertAdjacentHTML("afterbegin", `
      <div class="qa-item" id="${pendingId}">
        <p class="qa-question">${escapeHtml(question)}</p>
        <p class="qa-answer">Looking through the data…</p>
      </div>
    `);

    const startedAt = Date.now();
    const answerEl = () => document.getElementById(pendingId)?.querySelector(".qa-answer");
    const tickInterval = setInterval(() => {
      const el = answerEl();
      if (!el) return;
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      el.textContent = elapsed > 20
        ? `Looking through the data… (${elapsed}s — free-tier API limits can make this slow)`
        : `Looking through the data… (${elapsed}s)`;
    }, 1000);

    try {
      const resp = await fetch(`/datasets/${datasetId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = await parseJsonResponse(resp);
      const item = document.getElementById(pendingId);
      if (!resp.ok) {
        item.querySelector(".qa-answer").textContent = body.error || "Couldn't answer that.";
        return;
      }
      item.querySelector(".qa-answer").innerHTML = renderReportMarkdown(body.answer);
      if (body.tool_calls && body.tool_calls.length > 0) {
        item.insertAdjacentHTML("beforeend", `<div class="qa-toolcount">${body.tool_calls.length} quer${body.tool_calls.length === 1 ? "y" : "ies"} run</div>`);
      }
    } catch (err) {
      document.getElementById(pendingId).querySelector(".qa-answer").textContent = "Couldn't reach the server: " + err.message;
    } finally {
      clearInterval(tickInterval);
      askBtn.disabled = false;
      askInput.disabled = false;
      askInput.value = "";
      askInput.focus();
    }
  });

  // ---------------------------------------------------------------
  // Minimal markdown rendering — just what the report prompts actually
  // produce (## headers, bullet lists, **bold**, `code`), not a general
  // markdown parser.
  // ---------------------------------------------------------------

  function renderReportMarkdown(text) {
    if (!text) return "";
    const lines = text.split("\n");
    let html = "";
    let inList = false;

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (line.startsWith("## ")) {
        if (inList) { html += "</ul>"; inList = false; }
        html += `<h2>${inline(line.slice(3))}</h2>`;
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${inline(line.slice(2))}</li>`;
      } else if (line === "") {
        if (inList) { html += "</ul>"; inList = false; }
      } else {
        if (inList) { html += "</ul>"; inList = false; }
        html += `<p>${inline(line)}</p>`;
      }
    }
    if (inList) html += "</ul>";
    return html;
  }

  function inline(s) {
    let out = escapeHtml(s);
    out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/`(.+?)`/g, "<code>$1</code>");
    return out;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
  }
})();
