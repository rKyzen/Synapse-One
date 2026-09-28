@echo off
REM Synapse One - Root launcher delegating to core\start.bat
cd /d "%~dp0core"
call start.bat %*
