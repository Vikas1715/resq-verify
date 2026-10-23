@echo off
REM ONE genuine Gemini demo (free tier). Max 2 HTTP requests per click of "M3 - ResQ-Verify" (budget 2, retries OFF).
REM Reads GEMINI_API_KEY from .env (never printed). Overrides model/retries for THIS window only. Data goes to .\demo_data_gemini
call .venv\Scripts\activate.bat
findstr /r /b "GEMINI_API_KEY=.." .env >nul || (echo GEMINI_API_KEY is empty in .env - fix it first. & exit /b 1)
set LLM_PROVIDER=gemini
set GEMINI_MODEL=gemini-3.5-flash-lite
set LLM_MAX_RETRIES=0
set LLM_REQUEST_DELAY_S=0
set LLM_MAX_OUTPUT_TOKENS=16384
set STORAGE_BACKEND=file
set DATA_DIR=%CD%\demo_data_gemini
if not exist demo_data_gemini mkdir demo_data_gemini
echo *** GEMINI DEMO: model=%GEMINI_MODEL% retries=0 data=%DATA_DIR% ***
uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000
