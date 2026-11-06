@echo off
cd /d "%~dp0"
echo ================================================================
echo Starting Modular ML Health Monitoring Framework (React Edition)
echo ================================================================
echo Starting Backend API Server (Port 5000)...
start "Health ML Backend API" python backend/server.py
echo Starting React Vite Frontend (Port 5173)...
start "Health ML React Frontend" cmd /c "cd /d "%~dp0frontend" && npm run dev"
ping -n 4 127.0.0.1 >nul
echo Opening React application in default browser...
start http://localhost:5173/
echo.
echo Framework is online!
echo - React Portal:  http://localhost:5173/
echo - Backend API:   http://localhost:5000/
echo ================================================================
