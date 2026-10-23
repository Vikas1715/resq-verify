@echo off
REM First-time setup on Windows. Run from the project folder: setup.bat
python --version || (echo Python 3.10-3.12 is required. Install it from python.org and tick "Add to PATH". & exit /b 1)
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt || exit /b 1
if not exist .env copy .env.example .env
echo.
echo Setup done. Now open .env in Notepad and set GEMINI_API_KEY and GEMINI_MODEL.
