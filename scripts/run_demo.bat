@echo off
rem Chay demo mot lenh.
rem   scripts\run_demo.bat                    -> chay mock (khong can model), ghi logs_demo\
rem   scripts\run_demo.bat VIDEO.mp4 best.pt  -> chay detector that tren video
cd /d "%~dp0.."
set OUT=logs_demo
if not "%~1"=="" if "%~2"=="" (
  echo Thieu tham so. Cach dung: %~nx0 [VIDEO.mp4 best.pt] ^(khong tham so = chay mock^)
  exit /b 1
)
if not "%~1"=="" if not "%~2"=="" goto real
python src\pipeline\stream.py --mock --reset-db --out-dir %OUT%
goto done
:real
python src\pipeline\stream.py --reset-db --source "%~1" --detector "%~2" --out-dir %OUT% --dump-dets %OUT%\dets.jsonl
:done
echo Xong. Xem %OUT%\events.db, %OUT%\evidence\, %OUT%\run_meta.json
