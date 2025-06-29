@echo off
setlocal

:: Prompt user to paste the API key
set /p API_KEY=Please paste your GROQ API key and press Enter: 

:: Write the API key to the .env file
echo GROQ_API_KEY=%API_KEY% > .env

echo .env file created successfully with your API key!
pause
