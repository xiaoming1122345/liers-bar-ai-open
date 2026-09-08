# Front Desktop Shell

## Run

```bash
npm install
npm start
```

If Electron did not download correctly before, run:

```bash
npm run reinstall:electron
```

## What It Includes

- Main page:
  - launch training runs against the Python backend
  - parse recent reports under `../back/runs*`
  - import named player features into a local feature pool
  - open / close the overlay
  - toggle overlay always-on-top

- Overlay:
  - 4-seat live table layout
  - current-turn highlight
  - manual action/event logging with undo / redo
  - save-and-next-game flow
  - per-seat status, shots taken, hand count, last action
  - player-id binding to the imported feature pool
  - target / non-target / ghost / wild belief inputs
  - hero hand abstraction input: target / non-target / ghost / wild counts
  - automatic Python runtime advice refresh with seat threat, challenge pressure, and suggested hero action

## Notes

- The frontend talks to the backend by spawning `python ../back/run_training.py ...`.
- The overlay runtime advice uses `python ../back/run_runtime_advice.py --input ...`.
- Topmost behavior is implemented through Electron `BrowserWindow.setAlwaysOnTop`.
- Generic ids such as `player_2` and `me` are intentionally excluded from the persistent feature pool.
- Electron recovery is handled by `scripts/ensure-electron.js`, which first retries Electron's installer and then falls back to a direct Windows download when needed.
