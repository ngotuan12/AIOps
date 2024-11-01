:: Activate virtual environment
call venv\Scripts\activate

:: Start MLflow server
mlflow server ^
    --host 172.28.160.1 ^
    --port 7000 

:: Keep window open if there's an error
if %ERRORLEVEL% neq 0 (
    echo Error occurred while starting MLflow server
    pause
)