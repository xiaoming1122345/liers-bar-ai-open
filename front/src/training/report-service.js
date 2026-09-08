const fs = require("fs");
const path = require("path");

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function walk(directory, onFile) {
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath, onFile);
      continue;
    }
    onFile(fullPath);
  }
}

function summarizeReport(filePath) {
  const payload = readJsonFile(filePath);
  const stats = fs.statSync(filePath);
  const summary = payload.summary || {};
  const scoreboard = payload.scoreboard || [];
  const trainingSummary = payload.hero_training_summary || payload.training_summary || payload.evaluation_summary || {};
  const games = summary.games ?? payload.games ?? trainingSummary.sample_count ?? null;
  const seatModels = summary.seat_models || [];
  return {
    path: filePath,
    directory: path.basename(path.dirname(filePath)),
    fileName: path.basename(filePath),
    mode: payload.mode || "unknown",
    updatedAt: stats.mtime.toISOString(),
    mtimeMs: stats.mtimeMs,
    games,
    heroSeat: summary.hero_seat ?? null,
    winRate: summary.win_rate ?? null,
    meanUtility: summary.mean_utility ?? trainingSummary.mean_terminal_utility ?? null,
    scoreboardSize: Array.isArray(scoreboard) ? scoreboard.length : 0,
    seatModelCount: Array.isArray(seatModels) ? seatModels.length : 0,
  };
}

function discoverArtifacts(backDir) {
  const reports = [];
  const priorFiles = [];
  const runRoots = fs.readdirSync(backDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("runs"))
    .map((entry) => path.join(backDir, entry.name));

  for (const root of runRoots) {
    walk(root, (filePath) => {
      const baseName = path.basename(filePath);
      if (baseName === "report.json" || baseName === "sweep_report.json") {
        try {
          reports.push(summarizeReport(filePath));
        } catch {
          // Ignore partially-written files.
        }
        return;
      }
      if (baseName === "style_response_priors.json") {
        const stats = fs.statSync(filePath);
        priorFiles.push({
          path: filePath,
          directory: path.basename(path.dirname(filePath)),
          updatedAt: stats.mtime.toISOString(),
          mtimeMs: stats.mtimeMs,
        });
      }
    });
  }

  reports.sort((left, right) => right.mtimeMs - left.mtimeMs);
  priorFiles.sort((left, right) => right.mtimeMs - left.mtimeMs);

  return {
    reports,
    priorFiles,
  };
}

module.exports = {
  discoverArtifacts,
  readJsonFile,
};
