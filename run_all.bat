@echo off
setlocal enabledelayedexpansion

:: Setup oneAPI runtime PATH if not already present
where sycl-ls >nul 2>nul
if %errorlevel% neq 0 (
    if exist "D:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin" (
        set "PATH=D:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin;D:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin\compiler;D:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\lib\ocloc;!PATH!"
    ) else if exist "C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin" (
        set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin\compiler;C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\lib\ocloc;!PATH!"
    )
)

:: Locate executable
set "EXE="
if exist "%~dp0x64\Release\AmplitudeCallibrationSYCL.exe" (
    set "EXE=%~dp0x64\Release\AmplitudeCallibrationSYCL.exe"
) else if exist "%~dp0..\x64\Release\AmplitudeCallibrationSYCL.exe" (
    set "EXE=%~dp0..\x64\Release\AmplitudeCallibrationSYCL.exe"
) else if exist "%~dp0AmplitudeCallibrationSYCL.exe" (
    set "EXE=%~dp0AmplitudeCallibrationSYCL.exe"
)

if "%EXE%"=="" (
    echo [ERROR] AmplitudeCallibrationSYCL.exe not found! Please build the solution first.
    pause
    exit /b 1
)

:: Change directory to project folder where 490_kV and test FLR files are located
if exist "%~dp0AmplitudeCallibrationSYCL\490_kV" (
    cd /d "%~dp0AmplitudeCallibrationSYCL"
) else if exist "%~dp0490_kV" (
    cd /d "%~dp0"
)

echo =================================================================
echo   Running SYCL Amplitude Calibration: ALL TESTS ^& BENCHMARKS
echo =================================================================
echo Executable: %EXE%
echo Working Dir: %CD%
echo.

"%EXE%" --all
set EXIT_CODE=%errorlevel%

echo.
echo =================================================================
if %EXIT_CODE% equ 0 (
    echo   RESULT: ALL TESTS AND PIPELINES COMPLETED SUCCESSFULLY [OK]
) else (
    echo   RESULT: FAILED WITH CODE %EXIT_CODE% [ERROR]
)
echo =================================================================
echo.

pause
exit /b %EXIT_CODE%
