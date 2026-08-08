@echo off

echo ==========================================
echo Installing dependencies...
echo ==========================================

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ==========================================
echo Building...
echo ==========================================

pyinstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --windowed ^
    --name serialRobotApp ^
    serialRobotApp.py

echo.
echo ==========================================
echo Build finished.
echo Executable:
echo dist\RobotArmController.exe
echo ==========================================