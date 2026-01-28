@echo off
setlocal

REM --- Configuration & Path Setup ---
REM Get the directory of this batch script
SET SCRIPT_DIR=%~dp0

REM Navigate to the parent directory of SCRIPT_DIR (em-learn)
SET EM_LEARN_DIR=%SCRIPT_DIR%..
REM The python script is in the python-scripts subfolder of em-learn.
SET PYTHON_SCRIPT_PATH=%EM_LEARN_DIR%\python-scripts\rfc-codegen-em.py

REM PROJECT_ROOT_PATH should be four levels above SCRIPT_DIR to reach the frontend directory
SET PROJECT_ROOT_PATH=%SCRIPT_DIR%..\..\..\..\

REM Directory where the C code will be generated and compiled
REM This path is relative to the PROJECT_ROOT_PATH
SET C_CODE_DIR_RELATIVE=data\output\rfc\em-codegen
SET C_CODE_DIR=%PROJECT_ROOT_PATH%%C_CODE_DIR_RELATIVE%

REM Name of the C inference file (without extension)
SET C_INFERENCE_FILE_BASE=rfc_em_inference
SET C_INFERENCE_FILE=%C_CODE_DIR%\%C_INFERENCE_FILE_BASE%.c
SET C_EXECUTABLE_FILE=%C_CODE_DIR%\%C_INFERENCE_FILE_BASE%.exe

REM Store the current directory to return to it at the end
set "ORIGINAL_DIR=%CD%"

REM --- Python Script Execution ---
echo Running RFC Model Training and C Code Generation (emlearn)...
echo Python script: %PYTHON_SCRIPT_PATH%

REM Check if Python script exists
if not exist "%PYTHON_SCRIPT_PATH%" (
    echo ERROR: Python script not found at %PYTHON_SCRIPT_PATH%
    goto :error_exit
)

REM Run the Python script to generate models and C code
python "%PYTHON_SCRIPT_PATH%"
if %errorlevel% neq 0 (
    echo ERROR: Python script execution failed.
    goto :error_exit
)
echo Python script executed successfully.

REM --- C Code Compilation and Execution ---

REM Dynamically find emlearn include directory
echo.
echo Finding emlearn include directory...
set "TEMP_PATH_FILE=%TEMP%\emlearn_inc_path.txt"
del "%TEMP_PATH_FILE%" 2>nul
python -c "import emlearn; print(emlearn.includedir)" > "%TEMP_PATH_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python command to get emlearn path failed. Ensure Python and emlearn are installed.
    if exist "%TEMP_PATH_FILE%" del "%TEMP_PATH_FILE%"
    goto :error_exit
)
if not exist "%TEMP_PATH_FILE%" (
    echo ERROR: Temp file for emlearn path not created.
    goto :error_exit
)

set /p EMLEARN_SYSTEM_INCLUDE_DIR=<"%TEMP_PATH_FILE%"
del "%TEMP_PATH_FILE%" 2>nul

if not defined EMLEARN_SYSTEM_INCLUDE_DIR (
    echo ERROR: Could not read emlearn include directory.
    goto :error_exit
)
set EMLEARN_SYSTEM_INCLUDE_DIR=%EMLEARN_SYSTEM_INCLUDE_DIR:"=%

echo Found emlearn system include directory: %EMLEARN_SYSTEM_INCLUDE_DIR%

REM Navigate to the C code output directory
echo.
echo Step 1: Navigating to C code output directory: %C_CODE_DIR%
cd /D "%C_CODE_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not change to C code output directory '%C_CODE_DIR%'.
    goto :error_exit
)

if not exist "%C_INFERENCE_FILE_BASE%.c" (
    echo ERROR: Generated C file '%C_INFERENCE_FILE_BASE%.c' not found in %C_CODE_DIR%.
    goto :error_exit
)

REM Compile the C code
echo.
echo Step 2: Compiling C code (%C_INFERENCE_FILE_BASE%.c)...
echo Using emlearn system include: %EMLEARN_SYSTEM_INCLUDE_DIR%
echo Using local model include: .\include

gcc %C_INFERENCE_FILE_BASE%.c -o %C_INFERENCE_FILE_BASE%.exe -I. -Iinclude -I"%EMLEARN_SYSTEM_INCLUDE_DIR%" -lm

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: C compilation failed.
    goto :error_exit
)

REM Run the compiled program
echo.
echo Step 3: Running C inference program (%C_INFERENCE_FILE_BASE%.exe)...
echo.
%C_INFERENCE_FILE_BASE%.exe

goto :success_exit

:error_exit
echo.
echo --- PROCESS FAILED ---
if exist "%TEMP_PATH_FILE%" del "%TEMP_PATH_FILE%" 2>nul
cd /D "%ORIGINAL_DIR%"
endlocal
exit /b 1

:success_exit
echo.
echo --- PROCESS COMPLETED SUCCESSFULLY ---
cd /D "%ORIGINAL_DIR%"
endlocal
exit /b 0

:eof 