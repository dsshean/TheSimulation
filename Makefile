.PHONY: setup ui-setup run ui ui-build test-redis test-redis-integration test-tauri-redis clean-logs

# Backend / Python
setup:
	python -m pip install -r requirements.txt

run:
	python main_async.py

test-redis:
	python test_redis_simple.py

test-redis-integration:
	python test_redis_integration.py

test-tauri-redis:
	python test_tauri_redis.py

clean-logs:
	-@echo Cleaning logs/
	-@del /q logs\* 2> NUL || true

# Dashboard / Tauri
ui-setup:
	cd simulation-dashboard && npm install

ui:
	cd simulation-dashboard && npm run tauri:dev

ui-build:
	cd simulation-dashboard && npm run tauri:build

