const { contextBridge, ipcRenderer } = require("electron");

function subscribe(channel, callback) {
  const handler = (_event, payload) => callback(payload);
  ipcRenderer.on(channel, handler);
  return () => {
    ipcRenderer.removeListener(channel, handler);
  };
}

contextBridge.exposeInMainWorld("cardApi", {
  bootstrap: () => ipcRenderer.invoke("app:bootstrap"),
  subscribeState: (callback) => subscribe("state:changed", callback),
  setOverlayVisible: (visible) => ipcRenderer.invoke("overlay:set-visible", visible),
  setOverlayTopmost: (topmost) => ipcRenderer.invoke("overlay:set-topmost", topmost),
  setMySeat: (seat) => ipcRenderer.invoke("session:set-my-seat", seat),
  setCurrentTurnSeat: (seat) => ipcRenderer.invoke("session:set-current-turn", seat),
  setHeroHandView: (patch) => ipcRenderer.invoke("session:set-hero-hand", patch),
  updateSessionMeta: (patch) => ipcRenderer.invoke("session:update-meta", patch),
  updateSeat: (seat, patch) => ipcRenderer.invoke("session:update-seat", { seat, patch }),
  setSeatPlayerId: (seat, playerId) => ipcRenderer.invoke("session:set-seat-player-id", { seat, playerId }),
  appendEvent: (payload) => ipcRenderer.invoke("session:append-event", payload),
  undo: () => ipcRenderer.invoke("session:undo"),
  redo: () => ipcRenderer.invoke("session:redo"),
  nextGame: () => ipcRenderer.invoke("session:next-game"),
  selectReport: (reportPath) => ipcRenderer.invoke("reports:select", reportPath),
  readReport: (reportPath) => ipcRenderer.invoke("reports:read", reportPath),
  importReport: (reportPath) => ipcRenderer.invoke("reports:import", reportPath),
  deleteFeature: (playerId) => ipcRenderer.invoke("profiles:delete", playerId),
  runTraining: (command, params) => ipcRenderer.invoke("training:run", { command, params }),
  analyzeRuntime: (payload, priorPath) => ipcRenderer.invoke("runtime:analyze", { payload, priorPath }),
  getTrialManifest: () => ipcRenderer.invoke("trial:get-manifest"),
  getTrialStats: () => ipcRenderer.invoke("trial:get-stats"),
  logTrialDecision: (payload) => ipcRenderer.invoke("trial:log-decision", payload),
  logTrialSummary: (payload) => ipcRenderer.invoke("trial:log-summary", payload),
});

