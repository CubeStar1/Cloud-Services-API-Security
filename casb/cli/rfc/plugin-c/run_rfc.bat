@echo off
echo Running RFC Model Training and Testing...

:: Store the current directory
set ORIGINAL_DIR=%CD%

:: Run the Python script
echo.
echo Step 1: Training RFC model and generating C code...
python rfc-codegen.py

:: Check if Python script was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python script failed!
    exit /b 1
)

:: Change to the codegen directory
echo.
echo Step 2: Changing to codegen directory...
cd ..
cd ..
cd data\output\rfc\codegen

:: Compile the C code
echo.
echo Step 3: Compiling C code...
gcc api_classifier.c -o api_classifier.exe
if %ERRORLEVEL% NEQ 0 (
    echo Error: C compilation failed!
    cd %ORIGINAL_DIR%
    exit /b 1
)

:: Run the compiled program
echo.
echo Step 4: Running classifier...
echo.
api_classifier.exe

:: Return to original directory
cd %ORIGINAL_DIR%

echo.
echo Process completed successfully! 