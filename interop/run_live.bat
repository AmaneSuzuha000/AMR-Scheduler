@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
echo ================================================
echo   AMR 实时状态推送（给 C# / RAWSim-O 用）
echo   端口: 8770
echo   启动后运行 interop\csharp\LiveProbe.exe 127.0.0.1 8770
echo   按 Ctrl+C 停止
echo ================================================
python -m amr_sched.live --port 8770 --n-tasks 50 --n-agvs 4 --rate 20
pause