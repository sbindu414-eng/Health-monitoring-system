@echo off
echo ================================================================
echo Starting Modular ML Health Monitoring Framework
echo ================================================================
echo Starting Backend API Server (Port 5000)...
start "Health ML Backend API" python backend/server.py
echo Starting Frontend HTTP Server (Port 8000)...
start "Health ML Frontend" python -m http.server 8000
timeout /t 2 >nul
echo Opening application in default browser...
start http://localhost:8000/login.html
echo.
echo Framework is online!
echo - Login Portal:  http://localhost:8000/login.html
echo - Backend API:   http://localhost:5000/
echo ================================================================
