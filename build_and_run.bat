@echo off
setlocal enabledelayedexpansion

echo =================================================================
echo   Building AmplitudeCallibrationSYCL in Release x64...
echo =================================================================

set "MSBUILD="
if exist "D:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" (
    set "MSBUILD=D:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" (
    set "MSBUILD=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
) else (
    where msbuild >nul 2>nul
    if %errorlevel% equ 0 set "MSBUILD=msbuild"
)

if "%MSBUILD%"=="" (
    echo [ERROR] MSBuild.exe not found!
    pause
    exit /b 1
)

cd /d "%~dp0"
"%MSBUILD%" AmplitudeCallibrationSYCL.sln /p:Configuration=Release /p:Platform=x64
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo =================================================================
echo   Build Successful! Launching SYCL Verification...
echo =================================================================
echo.

call "%~dp0run_all.bat"
