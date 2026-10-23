@echo off
call .venv\Scripts\activate.bat
uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000
