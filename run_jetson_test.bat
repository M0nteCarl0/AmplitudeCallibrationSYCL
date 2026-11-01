@echo off
setlocal

set "TARGET_HOST=192.168.8.66"
set "DEVICE=gpu"
set "MODE=test"
set "FETCH="

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--device" (
    set "DEVICE=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--bench" (
    set "MODE=bench"
    shift
    goto parse_args
)
if /i "%~1"=="--all" (
    set "MODE=all"
    shift
    goto parse_args
)
if /i "%~1"=="--package" (
    set "FETCH=-FetchArtifacts"
    shift
    goto parse_args
)
if /i "%~1"=="--host" (
    set "TARGET_HOST=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_jetson_test.ps1" -TargetHost "%TARGET_HOST%" -Device "%DEVICE%" -Mode "%MODE%" %FETCH%
exit /b %errorlevel%
