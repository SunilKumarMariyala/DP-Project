# Solar Panel Fault Detection System - Command Guide

This guide provides all the commands needed to run the Solar Panel Fault Detection System in various configurations.

## Environment Setup Commands

### Install Required Packages
```powershell
pip install -r requirements.txt
```

### Install MATLAB Engine for Python
```powershell
# Navigate to MATLAB engine directory
cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"

# Install the engine
python setup.py install
```

## Data Generation Commands

### Generate Healthy Data
```powershell
python solar_fault_detection.py --generate-data --scenario healthy --count 50
```

### Generate Fault Data
```powershell
# Line-Line Fault
python solar_fault_detection.py --generate-data --scenario line_line_fault --count 20

# Open Circuit
python solar_fault_detection.py --generate-data --scenario open_circuit --count 20

# Partial Shading
python solar_fault_detection.py --generate-data --scenario partial_shading --count 20

# Panel Degradation
python solar_fault_detection.py --generate-data --scenario degradation --count 20
```

### Generate Random Mix of Data
```powershell
python solar_fault_detection.py --generate-data --scenario random --count 100
```

## MATLAB Integration Commands

### Run with MATLAB Integration
```powershell
python solar_fault_detection.py --host 127.0.0.1 --port 8080 --matlab
```

### Run MATLAB Interface Directly
```powershell
python matlab_interface.py
```

### Run MATLAB Interface with Custom Paths
```powershell
python matlab_interface.py --matlab-path "C:\Program Files\MATLAB\R2023b\bin\matlab.exe" --model-path "C:\Path\To\Your\Model"
```

## API Commands

### Start Monitoring
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/start" -Method POST
```

### Stop Monitoring
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/stop" -Method POST
```

### Get Latest Data
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/data/latest?limit=10" -Method GET
```

### Get Latest Alerts
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/alerts/latest?limit=5" -Method GET
```

### Get System Status
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/status" -Method GET
```

## Database Management Commands

### Reset Database
```powershell
python solar_fault_detection.py --reset-db
```

### Backup Database
```powershell
Copy-Item -Path "solar_panel.db" -Destination "solar_panel_backup_$(Get-Date -Format 'yyyyMMdd').db"
```

## Installation Commands

### Install Required Packages
```powershell
pip install -r requirements.txt
```

### Install MATLAB Engine for Python
```powershell
# Navigate to MATLAB engine directory
cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"

# Install the engine
python setup.py install
```

## Troubleshooting Commands

### Check Python Version
```powershell
python --version
```

### Check Installed Packages
```powershell
pip list
```

### Check MATLAB Engine Availability
```powershell
python -c "try: import matlab.engine; print('MATLAB Engine is available'); except ImportError: print('MATLAB Engine is not available')"
```

### Check Database Integrity
```powershell
python -c "import sqlite3; conn = sqlite3.connect('solar_panel.db'); print('Database connection successful'); conn.close()"
```

### Clear Log Files
```powershell
Remove-Item -Path "solar_fault_detection.log"
Remove-Item -Path "matlab_interface.log"
```

## Complete System Restart
```powershell
# Stop any running instances
Get-Process | Where-Object {$_.MainWindowTitle -match "python"} | Stop-Process -Force

# Clear logs
Remove-Item -Path "solar_fault_detection.log" -ErrorAction SilentlyContinue
Remove-Item -Path "matlab_interface.log" -ErrorAction SilentlyContinue

# Start the application
python solar_fault_detection.py --host 127.0.0.1 --port 8080
```
