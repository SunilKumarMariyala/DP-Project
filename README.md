# Solar Panel Fault Detection System

A machine learning-based system for detecting faults in solar panels using electrical measurements. The system achieves 96.25% testing accuracy in identifying healthy panels and various fault conditions.

## Quick Start Guide

1. **Setup the Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup the MySQL Database**:
   ```bash
   # First ensure MySQL is installed and running
   python database_setup.py
   ```

3. **Set Environment Variables**:
   ```bash
   # For Windows PowerShell
   $env:DB_HOST = "localhost"
   $env:DB_USER = "solar_user"
   $env:DB_PASSWORD = "your_secure_password"
   $env:DB_NAME = "solar_panel_db"
   $env:MATLAB_PATH = "C:\Program Files\MATLAB\R2023b\bin\matlab.exe"
   $env:MATLAB_MODEL_PATH = "path\to\your\GridConnectedPVFarmExample"
   ```

4. **Run the Application**:
   ```bash
   # For basic functionality
   python app.py --host 127.0.0.1 --port 8080
   
   # For advanced features with MATLAB integration
   python solar_fault_detection.py --host 127.0.0.1 --port 8080 --matlab
   
   # For continuous data flow from MATLAB
   python matlab_continuous_demo.py
   ```

5. **Access the Dashboard**:
   - Open a web browser and navigate to: `http://127.0.0.1:8080`

6. **Start Monitoring**:
   - Click the "Start Monitoring" button on the dashboard
   - Or use the API: `Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/start" -Method POST`

## System Features

- **Real-time Fault Detection**: Identifies 4 different types of solar panel faults
- **MATLAB Integration**: Works with MATLAB's GridConnectedPVFarm model
- **Continuous Data Flow**: Monitors for new MATLAB data files and processes them
- **Web Dashboard**: Visual interface for monitoring solar panel performance
- **REST API**: Programmatic access to system functionality
- **Mobile App Support**: Cross-platform mobile application for on-the-go monitoring

## Documentation Guide

This project includes several README files for different aspects of the system:

1. **[README_PERSONAL.md](./README_PERSONAL.md)** - Beginner-friendly guide with step-by-step instructions
2. **[README_SYSTEM_ORGANIZATION.md](./README_SYSTEM_ORGANIZATION.md)** - Technical overview of system components
3. **[README_CODING_GUIDE.md](./README_CODING_GUIDE.md)** - Programming concepts and coding practices
4. **[README_MATLAB_INTEGRATION.md](./README_MATLAB_INTEGRATION.md)** - How to integrate with MATLAB
5. **[README_GITHUB.md](./README_GITHUB.md)** - Instructions for uploading the project to GitHub
6. **[README_ADVANCED_MONITORING.md](./README_ADVANCED_MONITORING.md)** - Details about advanced monitoring
7. **[README_COMMANDS.md](./README_COMMANDS.md)** - Comprehensive list of all commands
8. **[README_MYSQL_SETUP.md](./README_MYSQL_SETUP.md)** - MySQL database setup instructions
9. **[README_MATLAB_MYSQL.md](./README_MATLAB_MYSQL.md)** - MATLAB with MySQL integration guide
10. **[README_CONTINUOUS_DATA_FLOW.md](./README_CONTINUOUS_DATA_FLOW.md)** - Guide for continuous data processing

## System Requirements

- **Python**: 3.8 or higher
- **MySQL**: 5.7 or higher
- **MATLAB**: R2019b or higher (optional, for MATLAB integration)
- **Operating System**: Windows, macOS, or Linux

## Fault Types Detected

1. **Healthy Panel**: Normal operation
2. **Line-Line Fault**: Two points in the array are connected
3. **Open Circuit Fault**: Circuit is broken somewhere in the array
4. **Partial Shading**: Some panels are receiving less sunlight
5. **Panel Degradation**: Efficiency is lower than expected

## Project Structure

```
Solar-Fault-Detection-System/
├── app.py                         # Basic web application
├── solar_fault_detection.py       # Advanced monitoring with MATLAB
├── matlab_interface.py            # Interface to MATLAB
├── matlab_continuous_demo.py      # Continuous data flow demo
├── database_setup.py              # Database configuration
├── requirements.txt               # Python dependencies
├── static/                        # Web assets (CSS, JS)
├── templates/                     # HTML templates
└── README files                   # Documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MATLAB GridConnectedPVFarm model for solar panel simulation
- PyTorch for the machine learning framework
- Flask for the web interface
