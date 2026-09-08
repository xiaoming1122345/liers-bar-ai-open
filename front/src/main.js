const path = require("path");
const os = require("os");
const fs = require("fs");
const { spawn } = require("child_process");
const { app, BrowserWindow, ipcMain } = require("electron");

const { SessionStore } = require("./state/session-store");
const { discoverArtifacts, readJsonFile } = require("./training/report-service");

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const FRONT_ROOT = path.join(PROJECT_ROOT, "front");
const BACK_ROOT = path.join(PROJECT_ROOT, "back");
const PYTHON_ENTRY = path.join(BACK_ROOT, "run_training.py");

let mainWindow = null;
let overlayWindow = null;
let activeTrainingProcess = null;
let cachedArtifacts = {
  reports: [],
  priorFiles: [],
};

const store = new SessionStore({
  dataDir: path.join(app.getPath("userData"), "card-ui"),
});

function appPayload() {
  return {
    ...store.getState(),
    reports: cachedArtifacts.reports,
    priorFiles: cachedArtifacts.priorFiles,
    paths: {
      projectRoot: PROJECT_ROOT,
      backRoot: BACK_ROOT,
    },
  };
}

function refreshArtifacts() {
  cachedArtifacts = discoverArtifacts(BACK_ROOT);
  broadcastState();
  return cachedArtifacts;
}

function broadcastState() {
  const payload = appPayload();
  for (const windowRef of [mainWindow, overlayWindow]) {
    if (windowRef && !windowRef.isDestroyed()) {
      windowRef.webContents.send("state:changed", payload);
    }
  }
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 980,
    minWidth: 1320,
    minHeight: 860,
    backgroundColor: "#0d1118",
    autoHideMenuBar: true,
    title: "Card Ops",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function createOverlayWindow() {
  overlayWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 1100,
    minHeight: 740,
    autoHideMenuBar: true,
    frame: false,
    title: "Card Overlay",
    skipTaskbar: true,
    show: false,
    backgroundColor: "#090c12",
    alwaysOnTop: store.state?.overlayTopmost ?? true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  overlayWindow.loadFile(path.join(__dirname, "renderer", "overlay.html"));
  overlayWindow.on("closed", () => {
    overlayWindow = null;
    store.setOverlayState({ visible: false });
  });
}

function ensureOverlayWindow() {
  if (!overlayWindow || overlayWindow.isDestroyed()) {
    createOverlayWindow();
  }
  return overlayWindow;
}

function buildTrainingArgs(command, params) {
  const args = [PYTHON_ENTRY, command];
  const push = (flag, value) => {
    if (value == null || value === "") {
      return;
    }
    args.push(flag, String(value));
  };

  if (command === "style_response_train") {
    push("--games", params.games ?? 24);
    push("--seed-start", params.seedStart ?? 42);
    push("--max-steps", params.maxSteps ?? 512);
    push("--output-dir", params.outputDir ?? "runs");
    push("--run-name", params.runName ?? "style_response_ui");
    return args;
  }

  if (command === "pool_self_play_eval") {
    push("--games", params.games ?? 8);
    push("--seed-start", params.seedStart ?? 42);
    push("--max-steps", params.maxSteps ?? 512);
    push("--output-dir", params.outputDir ?? "runs");
    push("--run-name", params.runName ?? "pool_self_play_ui");
    return args;
  }

  if (command === "hero_matchup_eval") {
    push("--hero-mode", params.heroMode ?? "adaptive_tabular");
    push("--name", params.name ?? "hero_ui");
    push("--iterations", params.iterations ?? 6);
    push("--seed", params.seed ?? 42);
    push("--max-depth", params.maxDepth ?? 4);
    push("--games", params.games ?? 12);
    push("--hero-seat", params.heroSeat ?? 1);
    push("--hero-label", params.heroLabel ?? "我");
    push("--seed-start", params.seedStart ?? 500);
    push("--policy-seed-base", params.policySeedBase ?? 1000);
    push("--max-steps", params.maxSteps ?? 512);
    push("--output-dir", params.outputDir ?? "runs");
    push("--run-name", params.runName ?? "hero_matchup_ui");
    push("--response-prior-path", params.responsePriorPath ?? "");
    return args;
  }

  if (command === "hero_seat_sweep") {
    push("--hero-mode", params.heroMode ?? "adaptive_tabular");
    push("--name", params.name ?? "hero_sweep_ui");
    push("--iterations", params.iterations ?? 6);
    push("--seed", params.seed ?? 42);
    push("--max-depth", params.maxDepth ?? 4);
    push("--games", params.games ?? 8);
    push("--hero-label", params.heroLabel ?? "我");
    push("--seed-start", params.seedStart ?? 900);
    push("--policy-seed-base", params.policySeedBase ?? 1600);
    push("--max-steps", params.maxSteps ?? 512);
    push("--output-dir", params.outputDir ?? "runs");
    push("--run-name", params.runName ?? "hero_sweep_ui");
    push("--response-prior-path", params.responsePriorPath ?? "");
    return args;
  }

  throw new Error(`Unsupported training command: ${command}`);
}

function runRuntimeAnalysis(payload, priorPath = null) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "card-runtime-"));
  const inputPath = path.join(tempDir, "snapshot.json");
  fs.writeFileSync(inputPath, JSON.stringify(payload), "utf-8");

  const args = [path.join(BACK_ROOT, "run_runtime_advice.py"), "--input", inputPath];
  const resolvedPriorPath = priorPath || cachedArtifacts.priorFiles[0]?.path || "";
  if (resolvedPriorPath) {
    args.push("--prior-path", resolvedPriorPath);
  }

  // 严格优先锁定生产清单中的第 25 轮候选模型 (与 manifest 完全一致)
  const lockedProductionPath = path.join(
    BACK_ROOT,
    "runs_deep_cfr",
    "deep_cfr_v21_fixed_rule_rollout_512",
    "policy_iter_00025_baseline.pt"
  );
  let checkpointPath = null;
  if (fs.existsSync(lockedProductionPath)) {
    checkpointPath = lockedProductionPath;
  } else {
    // 兜底回退
    const fallbackPath = path.join(
      BACK_ROOT,
      "runs_deep_cfr",
      "deep_cfr_v21_fixed_rule_rollout_512",
      "policy_best.pt"
    );
    if (fs.existsSync(fallbackPath)) {
      checkpointPath = fallbackPath;
    }
  }
  if (checkpointPath) {
    args.push("--checkpoint", checkpointPath);
  }

  return new Promise((resolve, reject) => {
    const processRef = spawn("python", args, {
      cwd: BACK_ROOT,
      shell: false,
    });
    let stdout = "";
    let stderr = "";

    processRef.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    processRef.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    processRef.on("error", (error) => {
      fs.rmSync(tempDir, { recursive: true, force: true });
      reject(error);
    });
    processRef.on("close", (code) => {
      fs.rmSync(tempDir, { recursive: true, force: true });
      if (code !== 0) {
        reject(new Error(stderr || `runtime advice exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`runtime advice returned invalid JSON: ${error.message}\n${stdout}\n${stderr}`));
      }
    });
  });
}

async function runTraining(command, params) {
  if (activeTrainingProcess) {
    throw new Error("已有训练在运行，请先等待当前任务结束。");
  }

  const args = buildTrainingArgs(command, params);
  activeTrainingProcess = spawn("python", args, {
    cwd: BACK_ROOT,
    shell: false,
  });

  store.setTrainingState({
    status: "running",
    command,
    output: `python ${args.join(" ")}\n`,
    lastExitCode: null,
    lastRunAt: new Date().toISOString(),
  });

  return new Promise((resolve, reject) => {
    let output = `python ${args.join(" ")}\n`;

    const appendChunk = (chunk) => {
      output += chunk.toString();
      const trimmed = output.split(/\r?\n/).slice(-80).join("\n");
      store.setTrainingState({
        status: "running",
        command,
        output: trimmed,
      });
    };

    activeTrainingProcess.stdout.on("data", appendChunk);
    activeTrainingProcess.stderr.on("data", appendChunk);

    activeTrainingProcess.on("error", (error) => {
      activeTrainingProcess = null;
      store.setTrainingState({
        status: "error",
        command,
        output: `${output}\n${error.message}`,
        lastExitCode: -1,
      });
      reject(error);
    });

    activeTrainingProcess.on("close", (code) => {
      activeTrainingProcess = null;
      const status = code === 0 ? "done" : "error";
      store.setTrainingState({
        status,
        command,
        output,
        lastExitCode: code,
      });
      refreshArtifacts();
      if (code === 0) {
        resolve({
          code,
          output,
        });
        return;
      }
      reject(new Error(`训练退出码 ${code}`));
    });
  });
}

app.whenReady().then(() => {
  refreshArtifacts();
  createMainWindow();
  if (store.state.overlayVisible) {
    const windowRef = ensureOverlayWindow();
    windowRef.show();
  }
  store.subscribe(() => {
    broadcastState();
  });
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

ipcMain.handle("app:bootstrap", async () => appPayload());

ipcMain.handle("overlay:set-visible", async (_event, visible) => {
  const windowRef = ensureOverlayWindow();
  if (visible) {
    windowRef.show();
    windowRef.focus();
  } else {
    windowRef.hide();
  }
  store.setOverlayState({ visible: Boolean(visible) });
  return appPayload();
});

ipcMain.handle("overlay:set-topmost", async (_event, topmost) => {
  const windowRef = ensureOverlayWindow();
  windowRef.setAlwaysOnTop(Boolean(topmost), "screen-saver");
  store.setOverlayState({ topmost: Boolean(topmost) });
  return appPayload();
});

ipcMain.handle("session:set-my-seat", async (_event, seat) => store.setMySeat(seat));
ipcMain.handle("session:set-current-turn", async (_event, seat) => store.setCurrentTurnSeat(seat));
ipcMain.handle("session:set-hero-hand", async (_event, patch) => store.setHeroHandView(patch || {}));
ipcMain.handle("session:update-meta", async (_event, patch) => store.updateSessionMeta(patch || {}));
ipcMain.handle("session:update-seat", async (_event, { seat, patch }) => store.updateSeat(seat, patch || {}));
ipcMain.handle("session:set-seat-player-id", async (_event, { seat, playerId }) => store.setSeatPlayerId(seat, playerId));
ipcMain.handle("session:append-event", async (_event, payload) => store.appendEvent(payload || {}));
ipcMain.handle("session:undo", async () => store.undo());
ipcMain.handle("session:redo", async () => store.redo());
ipcMain.handle("session:next-game", async () => store.nextGame());

ipcMain.handle("reports:select", async (_event, reportPath) => store.setSelectedReportPath(reportPath));
ipcMain.handle("reports:read", async (_event, reportPath) => {
  if (!reportPath) {
    return null;
  }
  return readJsonFile(reportPath);
});
ipcMain.handle("reports:import", async (_event, reportPath) => {
  const payload = readJsonFile(reportPath);
  return store.importReport(reportPath, payload);
});

ipcMain.handle("profiles:delete", async (_event, playerId) => store.deleteFeature(playerId));

ipcMain.handle("training:run", async (_event, { command, params }) => runTraining(command, params || {}));
ipcMain.handle("runtime:analyze", async (_event, { payload, priorPath }) => runRuntimeAnalysis(payload, priorPath));

function runTrialLogCommand(action, payload = null) {
  const args = [path.join(BACK_ROOT, "run_trial_log.py"), "--action", action];
  let tempDir = null;
  if (payload) {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "card-trial-"));
    const inputPath = path.join(tempDir, "payload.json");
    fs.writeFileSync(inputPath, JSON.stringify(payload), "utf-8");
    args.push("--input", inputPath);
  }

  return new Promise((resolve, reject) => {
    const processRef = spawn("python", args, {
      cwd: BACK_ROOT,
      shell: false,
    });
    let stdout = "";
    let stderr = "";
    processRef.stdout.on("data", (c) => { stdout += c.toString(); });
    processRef.stderr.on("data", (c) => { stderr += c.toString(); });
    processRef.on("error", (err) => {
      if (tempDir) fs.rmSync(tempDir, { recursive: true, force: true });
      reject(err);
    });
    processRef.on("close", (code) => {
      if (tempDir) fs.rmSync(tempDir, { recursive: true, force: true });
      if (code !== 0) {
        reject(new Error(stderr || `run_trial_log exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (err) {
        reject(new Error(`run_trial_log returned invalid JSON: ${err.message}\n${stdout}`));
      }
    });
  });
}

ipcMain.handle("trial:get-manifest", async () => runTrialLogCommand("get_manifest"));
ipcMain.handle("trial:get-stats", async () => runTrialLogCommand("get_stats"));
ipcMain.handle("trial:log-decision", async (_event, payload) => runTrialLogCommand("log_decision", payload));
ipcMain.handle("trial:log-summary", async (_event, payload) => runTrialLogCommand("log_summary", payload));

