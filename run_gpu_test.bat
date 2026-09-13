@echo off
setlocal enabledelayedexpansion

:: Setup oneAPI runtime PATH
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
    echo [ERROR] AmplitudeCallibrationSYCL.exe not found!
    pause
    exit /b 1
)

if exist "%~dp0AmplitudeCallibrationSYCL\490_kV" (
    cd /d "%~dp0AmplitudeCallibrationSYCL"
) else if exist "%~dp0490_kV" (
    cd /d "%~dp0"
)

echo =================================================================
echo   Running SYCL Validation Suite on GPU
echo =================================================================
echo.

"%EXE%" --test --device gpu
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
