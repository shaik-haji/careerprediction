@echo off
echo Initializing virtual environment and installing dependencies...
python -m venv venv
call venv/Scripts/activate
pip install -r requirements.txt

echo Training the model...
python Model/model.py

echo Setup complete. Run the application using local_run.bat
