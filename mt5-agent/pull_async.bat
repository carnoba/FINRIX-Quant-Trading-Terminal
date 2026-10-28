@echo off
rem Background wrapper: runs data_pull.py and logs everything (incl. exit code)
python "%~dp0data_pull.py" > "%~dp0pull.log" 2>&1
echo EXITCODE=%errorlevel% >> "%~dp0pull.log"
