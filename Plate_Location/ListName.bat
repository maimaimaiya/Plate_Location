@echo off
if exist NameList.txt del NameList.txt
for %%i in (.\src\general_test\*.jpg) do echo %%~fi>>NameList.txt
