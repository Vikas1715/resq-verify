@echo off
REM DEMO backend: SIMULATED (mock) LLM, zero Gemini calls, data kept in .\demo_data (separate from .\data).
REM It does NOT edit .env. The two "set" lines below only apply to this window.
call .venv\Scripts\activate.bat
set LLM_PROVIDER=mock
set STORAGE_BACKEND=file
set DATA_DIR=%CD%\demo_data
if not exist demo_data mkdir demo_data
echo *** DEMO MODE: SIMULATED LLM - results are NOT experimental results. Data folder: %DATA_DIR% ***
uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000
