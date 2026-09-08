const COMMAND_CONFIG = {
  style_response_train: {
    label: "风格先验",
    fields: [
      { name: "games", label: "games", type: "number", value: 24 },
      { name: "seedStart", label: "seed", type: "number", value: 42 },
      { name: "maxSteps", label: "steps", type: "number", value: 512 },
      { name: "outputDir", label: "out", type: "text", value: "runs" },
      { name: "runName", label: "name", type: "text", value: "style_response_ui" },
    ],
  },
  pool_self_play_eval: {
    label: "池对战",
    fields: [
      { name: "games", label: "games", type: "number", value: 8 },
      { name: "seedStart", label: "seed", type: "number", value: 42 },
      { name: "maxSteps", label: "steps", type: "number", value: 512 },
      { name: "outputDir", label: "out", type: "text", value: "runs" },
      { name: "runName", label: "name", type: "text", value: "pool_self_play_ui" },
    ],
  },
  hero_matchup_eval: {
    label: "Hero 对局",
    fields: [
      { name: "heroMode", label: "mode", type: "select", value: "adaptive_tabular", options: ["adaptive_tabular", "tabular", "random", "uniform_solver"] },
      { name: "heroSeat", label: "seat", type: "number", value: 1 },
      { name: "games", label: "games", type: "number", value: 12 },
      { name: "iterations", label: "iters", type: "number", value: 6 },
      { name: "maxDepth", label: "depth", type: "number", value: 4 },
      { name: "seed", label: "seed", type: "number", value: 42 },
      { name: "seedStart", label: "seedStart", type: "number", value: 500 },
      { name: "policySeedBase", label: "policy", type: "number", value: 1000 },
      { name: "maxSteps", label: "steps", type: "number", value: 512 },
      { name: "name", label: "solver", type: "text", value: "hero_ui" },
      { name: "outputDir", label: "out", type: "text", value: "runs" },
      { name: "runName", label: "name", type: "text", value: "hero_matchup_ui" },
      { name: "responsePriorPath", label: "prior", type: "prior-select", value: "" },
    ],
  },
  hero_seat_sweep: {
    label: "Hero 扫位",
    fields: [
      { name: "heroMode", label: "mode", type: "select", value: "adaptive_tabular", options: ["adaptive_tabular", "tabular", "random", "uniform_solver"] },
      { name: "games", label: "games", type: "number", value: 8 },
      { name: "iterations", label: "iters", type: "number", value: 6 },
      { name: "maxDepth", label: "depth", type: "number", value: 4 },
      { name: "seed", label: "seed", type: "number", value: 42 },
      { name: "seedStart", label: "seedStart", type: "number", value: 900 },
      { name: "policySeedBase", label: "policy", type: "number", value: 1600 },
      { name: "maxSteps", label: "steps", type: "number", value: 512 },
      { name: "name", label: "solver", type: "text", value: "hero_sweep_ui" },
      { name: "outputDir", label: "out", type: "text", value: "runs" },
      { name: "runName", label: "name", type: "text", value: "hero_sweep_ui" },
      { name: "responsePriorPath", label: "prior", type: "prior-select", value: "" },
    ],
  },
};

let state = null;
let selectedCommand = "style_response_train";
let selectedReportPath = "";
let selectedReportDetail = null;
const formValues = {};

const elements = {};

function $(id) {
  return document.getElementById(id);
}

function formatPercent(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(2)}`;
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}

function snapshotValues() {
  const config = COMMAND_CONFIG[selectedCommand];
  const result = {};
  for (const field of config.fields) {
    const key = `${selectedCommand}:${field.name}`;
    if (!(key in formValues)) {
      formValues[key] = field.value;
    }
    result[field.name] = formValues[key];
  }
  return result;
}

function renderCommandTabs() {
  elements.commandTabs.innerHTML = Object.entries(COMMAND_CONFIG)
    .map(([command, config]) => {
      const active = command === selectedCommand ? "active" : "";
      return `<button class="command-tab ${active}" data-command="${command}">${config.label}</button>`;
    })
    .join("");
}

function renderTrainingForm() {
  const config = COMMAND_CONFIG[selectedCommand];
  elements.trainingForm.innerHTML = config.fields
    .map((field) => {
      const key = `${selectedCommand}:${field.name}`;
      if (!(key in formValues)) {
        formValues[key] = field.value;
      }
      const value = formValues[key];
      let control = "";
      if (field.type === "select") {
        control = `
          <select data-field="${field.name}">
            ${field.options.map((option) => `<option value="${option}" ${String(option) === String(value) ? "selected" : ""}>${option}</option>`).join("")}
          </select>
        `;
      } else if (field.type === "prior-select") {
        const options = [{ path: "", directory: "(none)" }, ...(state?.priorFiles || [])];
        control = `
          <select data-field="${field.name}">
            ${options.map((option) => `<option value="${option.path}" ${String(option.path) === String(value) ? "selected" : ""}>${option.directory}</option>`).join("")}
          </select>
        `;
      } else {
        control = `<input data-field="${field.name}" type="${field.type}" value="${value}" />`;
      }
      return `
        <label class="field">
          <span>${field.label}</span>
          ${control}
        </label>
      `;
    })
    .join("");
}

function renderHeroStrip() {
  if (!state) {
    return;
  }
  elements.heroGameIndex.textContent = state.session.gameIndex;
  elements.heroTurnSeat.textContent = state.session.currentTurnSeat;
  elements.heroEventCount.textContent = state.session.history.length;
  elements.heroPoolCount.textContent = state.featurePool.length;
  elements.mySeatSelect.value = String(state.session.mySeat);
  elements.turnSeatSelect.value = String(state.session.currentTurnSeat);
  elements.targetRankSelect.value = String(state.session.targetRank);
  elements.topmostToggle.checked = Boolean(state.session.overlayTopmost);
}

function renderSeatStrip() {
  const seats = state?.session?.seats || [];
  elements.seatStatusStrip.innerHTML = seats
    .map((seat) => {
      const current = seat.seat === state.session.currentTurnSeat ? "current" : "";
      return `
        <div class="seat-strip-card ${current}">
          <h3>${seat.displayName}</h3>
          <p>${seat.status}</p>
          <p>枪位 ${seat.shotsTaken}/5 · 手牌 ${seat.handCount}</p>
          <p>${seat.lastAction || "待输入"}</p>
        </div>
      `;
    })
    .join("");
}

function renderReports() {
  const reports = state?.reports || [];
  elements.reportList.innerHTML = reports
    .slice(0, 16)
    .map((report) => {
      const active = report.path === selectedReportPath ? "active" : "";
      return `
        <button class="report-card ${active}" data-report-path="${report.path}">
          <h3>${report.directory}</h3>
          <p>${report.mode}</p>
          <p>${report.games ?? "-"} games · seat ${report.heroSeat ?? "-"}</p>
          <p>win ${formatPercent(report.winRate)} · util ${formatNumber(report.meanUtility)}</p>
        </button>
      `;
    })
    .join("");
}

function reportSummaryBoxes(detail) {
  const summary = detail?.summary || {};
  const items = [
    ["mode", detail?.mode || "-"],
    ["games", summary.games ?? detail?.games ?? "-"],
    ["hero", summary.hero_seat ?? "-"],
    ["win", formatPercent(summary.win_rate)],
    ["util", formatNumber(summary.mean_utility ?? summary.mean_terminal_utility)],
    ["models", (summary.seat_models || []).length],
  ];
  return items.map(([label, value]) => `
    <div class="summary-box">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

function renderReportDetail() {
  elements.reportMode.textContent = selectedReportDetail?.mode || "未选择";
  elements.reportUpdatedAt.textContent = selectedReportPath ? selectedReportPath : "-";
  elements.reportSummaryGrid.innerHTML = selectedReportDetail ? reportSummaryBoxes(selectedReportDetail) : "";
  elements.reportPreview.textContent = selectedReportDetail
    ? JSON.stringify(selectedReportDetail?.summary || selectedReportDetail, null, 2)
    : "";
}

function renderFeaturePool() {
  const pool = state?.featurePool || [];
  elements.poolMeta.textContent = `${pool.length} ids`;
  elements.featurePoolList.innerHTML = pool.length === 0
    ? `<div class="feature-card"><div><h3>暂无已命名玩家</h3><p>导入带真实 player_id 的报告后会在这里累计特征。</p></div></div>`
    : pool.map((feature) => `
        <div class="feature-card">
          <div>
            <h3>${feature.playerId}</h3>
            <p>${feature.publicStyleCluster} · ${feature.inferredProfileName}</p>
            <div class="feature-metrics">
              <div class="feature-pill">诈 ${formatPercent(feature.bluffRate)}</div>
              <div class="feature-pill">质 ${formatPercent(feature.challengeRate)}</div>
              <div class="feature-pill">压 ${formatPercent(feature.aggressionScore)}</div>
              <div class="feature-pill">变 ${formatPercent(feature.variabilityScore)}</div>
              <div class="feature-pill">受压 ${feature.pressureResponseLabel}</div>
              <div class="feature-pill">样本 ${feature.sampleCount}</div>
            </div>
          </div>
          <button class="ghost-btn" data-delete-player="${feature.playerId}">删除</button>
        </div>
      `).join("");
}

function renderTrainingStatus() {
  const training = state?.session?.training;
  if (!training) {
    return;
  }
  elements.trainingStatus.textContent = training.status;
  elements.trainingRunAt.textContent = training.lastRunAt ? formatDate(training.lastRunAt) : "-";
  elements.trainingOutput.textContent = training.output || "";
}

function render() {
  renderCommandTabs();
  renderTrainingForm();
  renderHeroStrip();
  renderSeatStrip();
  renderReports();
  renderReportDetail();
  renderFeaturePool();
  renderTrainingStatus();
}

async function loadReportDetail(reportPath) {
  if (!reportPath) {
    selectedReportDetail = null;
    renderReportDetail();
    return;
  }
  selectedReportDetail = await window.cardApi.readReport(reportPath);
  renderReportDetail();
}

function bindStaticEvents() {
  elements.commandTabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-command]");
    if (!button) {
      return;
    }
    selectedCommand = button.dataset.command;
    render();
  });

  elements.trainingForm.addEventListener("input", (event) => {
    const fieldName = event.target.dataset.field;
    if (!fieldName) {
      return;
    }
    const key = `${selectedCommand}:${fieldName}`;
    formValues[key] = event.target.value;
  });

  elements.startTrainingBtn.addEventListener("click", async () => {
    const values = snapshotValues();
    await window.cardApi.runTraining(selectedCommand, values);
  });

  elements.openOverlayBtn.addEventListener("click", () => window.cardApi.setOverlayVisible(true));
  elements.closeOverlayBtn.addEventListener("click", () => window.cardApi.setOverlayVisible(false));
  elements.topmostToggle.addEventListener("change", (event) => window.cardApi.setOverlayTopmost(event.target.checked));
  elements.mySeatSelect.addEventListener("change", (event) => window.cardApi.setMySeat(Number(event.target.value)));
  elements.turnSeatSelect.addEventListener("change", (event) => window.cardApi.setCurrentTurnSeat(Number(event.target.value)));
  elements.targetRankSelect.addEventListener("change", (event) => window.cardApi.updateSessionMeta({ targetRank: event.target.value }));
  elements.undoBtn.addEventListener("click", () => window.cardApi.undo());
  elements.redoBtn.addEventListener("click", () => window.cardApi.redo());
  elements.nextGameBtn.addEventListener("click", () => window.cardApi.nextGame());
  elements.refreshReportsBtn.addEventListener("click", async () => {
    const payload = await window.cardApi.bootstrap();
    state = payload;
    render();
  });

  elements.reportList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-report-path]");
    if (!button) {
      return;
    }
    selectedReportPath = button.dataset.reportPath;
    await window.cardApi.selectReport(selectedReportPath);
    await loadReportDetail(selectedReportPath);
    renderReports();
  });

  elements.importReportBtn.addEventListener("click", async () => {
    if (!selectedReportPath) {
      return;
    }
    await window.cardApi.importReport(selectedReportPath);
  });

  elements.featurePoolList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-delete-player]");
    if (!button) {
      return;
    }
    await window.cardApi.deleteFeature(button.dataset.deletePlayer);
  });
}

async function bootstrap() {
  elements.commandTabs = $("commandTabs");
  elements.trainingForm = $("trainingForm");
  elements.startTrainingBtn = $("startTrainingBtn");
  elements.trainingStatus = $("trainingStatus");
  elements.trainingRunAt = $("trainingRunAt");
  elements.trainingOutput = $("trainingOutput");
  elements.heroGameIndex = $("heroGameIndex");
  elements.heroTurnSeat = $("heroTurnSeat");
  elements.heroEventCount = $("heroEventCount");
  elements.heroPoolCount = $("heroPoolCount");
  elements.openOverlayBtn = $("openOverlayBtn");
  elements.closeOverlayBtn = $("closeOverlayBtn");
  elements.topmostToggle = $("topmostToggle");
  elements.mySeatSelect = $("mySeatSelect");
  elements.turnSeatSelect = $("turnSeatSelect");
  elements.targetRankSelect = $("targetRankSelect");
  elements.undoBtn = $("undoBtn");
  elements.redoBtn = $("redoBtn");
  elements.nextGameBtn = $("nextGameBtn");
  elements.refreshReportsBtn = $("refreshReportsBtn");
  elements.seatStatusStrip = $("seatStatusStrip");
  elements.reportList = $("reportList");
  elements.reportMode = $("reportMode");
  elements.reportUpdatedAt = $("reportUpdatedAt");
  elements.reportSummaryGrid = $("reportSummaryGrid");
  elements.reportPreview = $("reportPreview");
  elements.importReportBtn = $("importReportBtn");
  elements.featurePoolList = $("featurePoolList");
  elements.poolMeta = $("poolMeta");

  state = await window.cardApi.bootstrap();
  selectedReportPath = state.session.selectedReportPath || state.reports[0]?.path || "";
  if (selectedReportPath) {
    selectedReportDetail = await window.cardApi.readReport(selectedReportPath);
  }
  bindStaticEvents();
  render();
  window.cardApi.subscribeState((payload) => {
    state = payload;
    if (!selectedReportPath && payload.session.selectedReportPath) {
      selectedReportPath = payload.session.selectedReportPath;
    }
    render();
  });
}

bootstrap();
