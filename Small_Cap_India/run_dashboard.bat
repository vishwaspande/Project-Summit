@echo off
echo ============================================
echo   Super Investor Stock Analyzer Dashboard
echo ============================================
echo.
echo Starting local server...
echo Dashboard will open at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
echo.
start http://localhost:8000
python -m http.server 8000
pause
