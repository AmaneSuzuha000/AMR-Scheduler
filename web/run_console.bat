@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set AMR_WEB_PORT=8077
cd /d "%~dp0.."
echo ================================================
echo   铁路货车检修车间 多AGV调度控制台
echo   启动后请用浏览器打开:  http://127.0.0.1:8077
echo   按 Ctrl+C 停止
echo ================================================
python web\server.py
pause