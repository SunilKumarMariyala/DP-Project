# MATLAB Integration Guide for Solar Fault Detection System

This guide explains how to integrate your real MATLAB model with the Solar Fault Detection System for real-time monitoring and fault prediction.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Single-Computer Setup](#single-computer-setup)
3. [Multi-Computer Setup](#multi-computer-setup)
4. [Running the System](#running-the-system)
5. [Troubleshooting](#troubleshooting)
6. [Path Configuration Guide](#path-configuration-guide)

## Prerequisites

- **MATLAB Installation**: R2019b or newer
- **MATLAB Engine for Python**: Required for Python to communicate with MATLAB
- **GridConnectedPVFarm Model**: The MATLAB model for solar panel simulation
- **Python 3.8+**: For running the Solar Fault Detection System

## Single-Computer Setup

Follow these steps to set up the system on a single computer:

### 1. Install MATLAB Engine for Python

```bash
# Navigate to your MATLAB installation folder
cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"

# Install the engine
python setup.py install
```

### 2. Configure the MATLAB Interface

Open `matlab_interface.py` and update the MATLAB paths:

```python
self.matlab_path = matlab_path or r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe"
self.model_path = model_path or r"C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample"
```

Replace these paths with the actual paths on your system.

### 3. Test the MATLAB Connection

Run the MATLAB interface script to verify the connection:

```bash
python matlab_interface.py
```

You should see output showing a successful simulation with PV current, voltage, and power values.

## Multi-Computer Setup

If you want to run MATLAB on one computer and the Python application on another, follow these steps:

### Option 1: Shared Database Approach (Recommended)

#### On the MATLAB Computer:

1. **Copy the MATLAB script**:
   - Save `matlab_to_database.m` to your MATLAB computer

2. **Configure the database path**:
   - Open `matlab_to_database.m` and update the database path:
     ```matlab
     dbPath = '\\SHARED_LOCATION\solar_panel.db'; % Update with your shared path
     ```

3. **Run the script in MATLAB**:
   ```matlab
   matlab_to_database
   ```
   This will start generating data and saving it to the shared database.

#### On the Python Computer:

1. **Configure the database connection**:
   - When starting the application, specify the shared database path:
     ```bash
     python solar_fault_detection.py --db-path \\SHARED_LOCATION\solar_panel.db --host 127.0.0.1 --port 8080
     ```

### Option 2: Network Share Approach

1. **Share the MATLAB folder** on the computer where MATLAB is installed
2. **Map the shared folder** as a network drive on the Python computer
3. **Update the path** in `matlab_interface.py` to use the network path
4. **Install MATLAB Engine for Python** on the Python computer

### Option 3: REST API Approach

For the most flexible setup, create a web API on the MATLAB computer that the Python application can call.

## Running the System

### Single-Computer Mode

1. **Start the application**:
   ```bash
   python solar_fault_detection.py --host 127.0.0.1 --port 8080
   ```

2. **Access the dashboard**:
   - Open a web browser and navigate to `http://127.0.0.1:8080`

3. **Use the MATLAB integration**:
   - Click on "MATLAB Integration" in the dashboard
   - Set parameters (irradiance, temperature)
   - Click "Run Simulation" to get real-time data

### Multi-Computer Mode

1. **On the MATLAB computer**:
   - Run the MATLAB script to generate data
   - Verify data is being saved to the shared database

2. **On the Python computer**:
   - Start the application with the shared database path
   - Access the dashboard to view real-time data and predictions

## Troubleshooting

### MATLAB Engine Issues

- **Error**: "No module named 'matlab'"
  - **Solution**: Reinstall MATLAB Engine for Python

- **Error**: "Failed to initialize MATLAB engine"
  - **Solution**: Check MATLAB installation and path

### Database Connectivity Issues

- **Error**: "Unable to open database file"
  - **Solution**: Check file permissions and path

- **Error**: "Database is locked"
  - **Solution**: Make sure no other process is using the database

### Model Path Issues

- **Error**: "Cannot find model file"
  - **Solution**: Verify the model path in `matlab_interface.py`

## Path Configuration Guide

This section provides a comprehensive guide to all the paths you need to configure for the system to work properly.

### Essential Paths to Configure

1. **MATLAB Executable Path** (in `matlab_interface.py`):
   ```python
   self.matlab_path = matlab_path or r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe"
   ```
   Change this to the actual path of your MATLAB executable.

2. **MATLAB Model Path** (in `matlab_interface.py`):
   ```python
   self.model_path = model_path or r"C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample"
   ```
   Change this to the actual path of your GridConnectedPVFarm model.

3. **Database Path** (in multiple files):
   - In `matlab_interface.py`:
     ```python
     self.db_path = db_path
     ```
   - When running the application:
     ```bash
     python solar_fault_detection.py --db-path solar_panel.db
     ```
   - In `matlab_to_database.m` (for multi-computer setup):
     ```matlab
     dbPath = '\\SHARED_LOCATION\solar_panel.db';
     ```

### How to Change Paths

#### Method 1: Edit the Files Directly

1. Open the file in a text editor
2. Locate the path variable
3. Replace it with your actual path
4. Save the file

#### Method 2: Use Command Line Arguments

When starting the application, you can specify paths as arguments:

```bash
python solar_fault_detection.py --matlab-path "C:\path\to\matlab.exe" --model-path "C:\path\to\model" --db-path "path\to\database.db"
```

#### Method 3: Environment Variables

You can set environment variables that the application will use:

```bash
set MATLAB_PATH=C:\path\to\matlab.exe
set MODEL_PATH=C:\path\to\model
set DB_PATH=path\to\database.db
python solar_fault_detection.py
```

### Path Configuration for Multi-Computer Setup

For a multi-computer setup, ensure that:

1. **Shared Database Path** is accessible from both computers
2. **Network Paths** use the correct format: `\\SERVER\Share\Path`
3. **File Permissions** allow both computers to read/write to the shared locations

## Data Flow

1. **MATLAB Model** generates solar panel data (current, voltage, power)
2. **Data is saved** to the SQLite database
3. **Python application** reads data from the database
4. **Machine learning model** makes fault predictions
5. **Dashboard** displays results and alerts

## Advanced Configuration

For advanced users who want to customize the integration:

1. Modify `matlab_interface.py` to change how data is processed
2. Update `database_setup.py` to change the database schema
3. Modify `solar_fault_detector.py` to adjust the prediction algorithm

## System Architecture

The complete system works as follows:

1. **MATLAB Model** - Generates data from real solar panels
2. **Database** - Stores the solar panel data
3. **Python Prediction Model** - Processes the data and predicts faults
4. **Web Dashboard** - Displays the results

## Setup Instructions

### 1. Configure MATLAB Integration

1. Install MATLAB Engine for Python on the computer running this application:
   ```
   cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"
   python setup.py install
   ```
   (Replace `R2023b` with your actual MATLAB version)

2. Update the MATLAB model path in `matlab_interface.py`:
   - Open `matlab_interface.py`
   - Find the line: `self.model_path = model_path or r"C:\PATH\TO\YOUR\ACTUAL\MATLAB\MODEL"`
   - Replace with the actual path to your MATLAB model

3. Ensure your MATLAB model has the following output signals:
   - `PV_I` - PV current
   - `PV_V` - PV voltage
   - `PV_P` - PV power
   - `Grid_P` - Grid power

### 2. Configure Database Connection

The system uses SQLite by default, but you can modify it to use a different database if needed:

1. Default configuration (SQLite):
   - Database file: `solar_panel.db`
   - No additional setup required

2. For a shared database (if MATLAB is on another computer):
   - Consider using MySQL, PostgreSQL, or SQL Server
   - Update the connection string in `database_setup.py`

### 3. Real-time Data Flow

For real-time monitoring:

1. MATLAB generates data and saves it to the database
2. The Python application reads from the database at regular intervals
3. The prediction model processes the data
4. Results are displayed on the web dashboard

## Testing the Integration

1. Start the MATLAB model on your MATLAB computer
2. Run the Python application:
   ```
   python solar_fault_detection.py --host 127.0.0.1 --port 8080
   ```
3. Open a web browser and navigate to:
   ```
   http://127.0.0.1:8080
   ```

## Troubleshooting

If you encounter issues with the MATLAB integration:

1. Check the logs in `matlab_interface.log` for errors
2. Verify that MATLAB Engine for Python is installed correctly
3. Ensure your MATLAB model has the correct output signals
4. Check database connectivity

## Advanced Configuration

For advanced users who want to customize the integration:

1. Edit `matlab_interface.py` to modify how data is retrieved from MATLAB
2. Update `database_setup.py` to change the database schema
3. Modify `solar_fault_detector.py` to adjust the prediction algorithm

## Multi-Computer Setup for MATLAB Integration

Since your MATLAB model is on a different computer (`C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample`), you have two options:

### Option 1: Network Share Approach

1. **Share the MATLAB folder** on the computer where MATLAB is installed:
   - Right-click on the folder containing your MATLAB model
   - Select "Properties" → "Sharing" → "Share"
   - Set appropriate permissions

2. **Access the shared folder** from the computer running the Python application:
   - Map the shared folder as a network drive
   - Update the path in `matlab_interface.py` to use the network path:
     ```python
     self.model_path = model_path or r"\\COMPUTER_NAME\SharedFolder\GridConnectedPVFarmExample"
     ```
     (Replace COMPUTER_NAME with the actual computer name)

3. **Install MATLAB Engine for Python** on the computer running the Python application

### Option 2: Database-Only Integration

If sharing the MATLAB folder is not feasible, you can use a database-only approach:

1. **On the MATLAB computer**:
   - Modify your MATLAB model to save data directly to a shared database
   - Create a script that runs the model and saves results at regular intervals

2. **On the Python computer**:
   - Configure the application to read from the shared database
   - Disable the direct MATLAB integration in `matlab_interface.py`

### Option 3: REST API Approach (Most Flexible)

For the most flexible setup:

1. **On the MATLAB computer**:
   - Create a simple web server (using Python Flask or MATLAB's own web server capabilities)
   - Expose an API endpoint that runs the MATLAB model and returns results

2. **On the Python computer**:
   - Modify `matlab_interface.py` to call the API instead of trying to run MATLAB directly
   - Process the returned data and save it to the local database

This approach provides the most flexibility and doesn't require sharing folders or a shared database.

## Network Configuration

For any of these approaches to work:

1. Ensure both computers are on the same network
2. Configure firewalls to allow necessary connections
3. Test connectivity between the computers

## Network Configuration for Multi-Computer Setup

If running MATLAB on a separate computer:

1. Ensure both computers are on the same network
2. Use a shared database accessible from both computers
3. Update firewall settings to allow database connections
4. Consider using a message queue (like RabbitMQ) for more robust data transfer
