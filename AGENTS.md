# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python code (e.g., `simulation_async.py`, `redis_client.py`, `agents.py`).
- Root helpers: `main_async.py` (entry point), test scripts (`test_*.py`).
- `data/`, `logs/`, `world_configurations/`: Runtime data, logs, and world definitions.
- `simulation-dashboard/`: Tauri + React UI (`src/` for components; `src-tauri/` for desktop app).
- Config: `.env` (runtime secrets), `requirements.txt` (Python deps), `README.md` (detailed docs).

## Build, Test, and Development Commands
- Python setup: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt` (Windows PowerShell: `venv\\Scripts\\Activate.ps1`).
- Run simulation: `python main_async.py` (requires a local Redis on `127.0.0.1:6379`).
- Dashboard dev: `cd simulation-dashboard && npm install && npm run tauri:dev`.
- Dashboard build: `cd simulation-dashboard && npm run tauri:build`.
- Quick Redis check: `python test_redis_simple.py`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indents, `snake_case` for functions/vars, `PascalCase` for classes. Prefer type hints; keep modules focused.
- Logging: use `src/logger_config.py` conventions; avoid `print()` in library code.
- Frontend (TS/React): Components `PascalCase` (e.g., `GraphView.tsx`), hooks `useThing.ts`, utilities in `utils/`.
- Filenames: Python modules `snake_case.py`; assets and data use kebab-case or descriptive names.

## Testing Guidelines
- No unified test runner; use provided scripts:
  - `python test_redis_simple.py` (connectivity)
  - `python test_redis_integration.py` (pub/sub + handlers)
  - `python test_tauri_redis.py` (UI feed)
- Ensure Redis is running locally before tests. Add focused test scripts alongside root for new subsystems.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., `Fix Redis publisher retry`). Group related changes.
- PRs: include purpose, key changes, run steps, and risks. Link issues; add UI screenshots for dashboard changes. Confirm simulation run + Redis tests pass.

## Security & Configuration Tips
- Copy `.env.sample` to `.env`; never commit secrets. Validate config via `python test_redis_simple.py` before running.
- Avoid long‑running `print()` loops; prefer structured logs to `logs/`.
- Large files (images/logs) should not be in PRs unless essential.

