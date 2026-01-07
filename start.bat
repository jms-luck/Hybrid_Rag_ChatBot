@echo off
echo Starting Study Bot Application...
echo.

REM Start backend
echo Starting FastAPI backend...
start "FastAPI Backend" cmd /k "cd backend && python main.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting React frontend...
start "React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause > nul
