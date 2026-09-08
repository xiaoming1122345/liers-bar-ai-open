const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const FORCE = process.argv.includes("--force");
const PROJECT_ROOT = path.resolve(__dirname, "..");
const ELECTRON_ROOT = path.join(PROJECT_ROOT, "node_modules", "electron");
const DIST_DIR = path.join(ELECTRON_ROOT, "dist");
const ELECTRON_EXE = path.join(DIST_DIR, process.platform === "win32" ? "electron.exe" : "electron");
const MIRROR = process.env.ELECTRON_MIRROR || "https://npmmirror.com/mirrors/electron/";

function exists(filePath) {
  return fs.existsSync(filePath);
}

function readElectronVersion() {
  const packagePath = path.join(ELECTRON_ROOT, "package.json");
  const payload = JSON.parse(fs.readFileSync(packagePath, "utf-8"));
  return payload.version;
}

function ensureEmptyDir(directory) {
  fs.rmSync(directory, { recursive: true, force: true });
  fs.mkdirSync(directory, { recursive: true });
}

function runNodeInstall(version) {
  console.log(`[electron] trying install.js for ${version}`);
  const installScript = path.join(ELECTRON_ROOT, "install.js");
  const result = spawnSync(process.execPath, [installScript], {
    cwd: PROJECT_ROOT,
    stdio: "inherit",
    env: {
      ...process.env,
      ELECTRON_MIRROR: MIRROR,
    },
  });
  return result.status === 0 && exists(ELECTRON_EXE);
}

function manualWindowsInstall(version) {
  const archiveUrl = `${MIRROR.replace(/\/+$/, "")}/v${version}/electron-v${version}-win32-x64.zip`;
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "electron-manual-"));
  const archivePath = path.join(tempDir, `electron-v${version}-win32-x64.zip`);

  console.log(`[electron] manual download from ${archiveUrl}`);
  const downloadScript = [
    "$ProgressPreference='SilentlyContinue'",
    `[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12`,
    `Invoke-WebRequest -UseBasicParsing -Uri '${archiveUrl}' -OutFile '${archivePath}'`,
    `New-Item -ItemType Directory -Force -Path '${DIST_DIR}' | Out-Null`,
    `Expand-Archive -LiteralPath '${archivePath}' -DestinationPath '${DIST_DIR}' -Force`,
  ].join("; ");

  ensureEmptyDir(DIST_DIR);
  const result = spawnSync(
    "powershell",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", downloadScript],
    {
      cwd: PROJECT_ROOT,
      stdio: "inherit",
    },
  );

  fs.rmSync(tempDir, { recursive: true, force: true });
  return result.status === 0 && exists(ELECTRON_EXE);
}

function main() {
  if (!exists(ELECTRON_ROOT)) {
    console.log("[electron] package directory not found, skipping");
    return;
  }

  if (!FORCE && exists(ELECTRON_EXE)) {
    console.log("[electron] runtime already present");
    return;
  }

  const version = readElectronVersion();
  if (runNodeInstall(version)) {
    console.log("[electron] install.js recovered the runtime");
    return;
  }

  if (process.platform === "win32") {
    if (manualWindowsInstall(version)) {
      console.log("[electron] manual Windows download recovered the runtime");
      return;
    }
  }

  throw new Error(
    "Electron runtime is still missing after recovery. " +
      `Expected binary at ${ELECTRON_EXE}`,
  );
}

main();
