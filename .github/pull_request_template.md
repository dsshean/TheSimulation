# PR Title

## Summary
- What problem does this PR solve? Provide a concise overview.

## Type
- [ ] Feature
- [ ] Bug Fix
- [ ] Refactor
- [ ] Docs
- [ ] Infra/Chore

## Linked Issues
- Closes #
- Related #

## Changes
- Brief bullet list of key changes.
- Note any breaking changes or migrations.

## Screenshots / Logs (if UI/dashboard)
- Include before/after images or relevant logs.

## How to Test
- Redis running locally on `127.0.0.1:6379`.
- Steps:
  1. `make test-redis` (connectivity)
  2. `make run` (backend)
  3. `make ui` (dashboard dev) or `make test-tauri-redis` (feed publisher)

## Checklist
- [ ] Simulation runs locally without errors
- [ ] Redis tests pass (`test_redis_simple.py`, `test_redis_integration.py`)
- [ ] Docs updated (e.g., README, AGENTS.md) if needed
- [ ] No secrets committed; `.env` handled locally
- [ ] UI changes include screenshots (if applicable)

