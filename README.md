# Solar Panel Fault Detection System

A machine learning-based system for detecting faults in solar panels using electrical measurements. The system achieves 96.25% testing accuracy in identifying healthy panels and various fault conditions.

## Quick Start Guide

1. **Setup the Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup the Database**:
   ```bash
   python database_setup.py
   ```

3. **Run the Application**:
   ```bash
   python solar_fault_detection.py --host 127.0.0.1 --port 8080
   ```

4. **Access the Dashboard**:
   - Open a web browser and navigate to: `http://127.0.0.1:8080`

5. **Start Monitoring**:
   - Click the "Start Monitoring" button on the dashboard
   - Or use the API: `Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/start" -Method POST`

## Documentation Guide

This project includes several README files for different aspects of the system:

1. **README_PERSONAL.md** - Beginner-friendly guide with step-by-step instructions for running the project
2. **README_SYSTEM_ORGANIZATION.md** - Technical overview of system components and how they interact
3. **README_CODING_GUIDE.md** - Programming concepts and coding practices used in the project
4. **README_MATLAB_INTEGRATION.md** - How to integrate with MATLAB for real solar panel data
5. **README_GITHUB.md** - Instructions for uploading the project to GitHub
6. **README_ADVANCED_MONITORING.md** - Details about the advanced monitoring features
7. **README_COMMANDS.md** - Comprehensive list of all commands to run the system

## Mobile Application

A cross-platform mobile application is available for monitoring your solar panels on the go. The app provides:

- Real-time monitoring of solar panel data
- Fault detection alerts and notifications
- Historical data viewing and analysis
- Alert management system

To set up and use the mobile app:

1. Navigate to the mobile app directory:
   ```bash
   cd solar_monitor_app
   ```

2. Follow the detailed instructions in the [mobile app README](./solar_monitor_app/README.md)

3. Connect the app to your running Solar Panel Fault Detection System server

See the [mobile app README](./solar_monitor_app/README.md) for comprehensive setup and usage instructions.

## Running with MATLAB Integration

To run the system with MATLAB integration:

1. **Install MATLAB Engine for Python**:
   ```bash
   cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"
   python setup.py install
   ```

2. **Run with MATLAB Integration**:
   ```bash
   python solar_fault_detection.py --host 127.0.0.1 --port 8080 --matlab
   ```

3. **Verify Integration**:
   - Check the console logs for "MATLAB interface initialized successfully"
   - The real-time charts will update with data from MATLAB when available

## Features

### 1. Manual Prediction
- **Simple Input**: Only requires two measurements:
  - PV Current
  - PV Voltage
- **Automatic Pattern Recognition**: The system automatically analyzes these inputs to detect:
  - Healthy Operation
  - Fault 1 (Line-Line Fault)
  - Fault 2 (Open Circuit)
  - Fault 3 (Partial Shading)
  - Fault 4 (Degradation)
- **High Accuracy**: 96.25% accuracy in testing, with strong performance across all fault types
- **Detailed Analysis**: Provides confidence scores, descriptions, and recommended actions

### 2. Real-time Monitoring
- Continuous monitoring of solar panel performance
- Automatic fault detection and alerts
- Historical data tracking and analysis
- Performance metrics and statistics

### 3. Performance Statistics
- Track total measurements processed
- View distribution of fault types
- Monitor system health over time
- Analyze prediction confidence and response times

### 4. MATLAB Integration
- **Real-time Simulation**: Connect to MATLAB's GridConnectedPVFarm model for realistic simulations
- **Parameterized Testing**: Adjust irradiance and temperature to test various operating conditions
- **Dataset Generation**: Create synthetic datasets with varying conditions for model training
- **Fault Visualization**: Interactive dashboard for visualizing simulation results and fault predictions
- **Performance Metrics**: Track efficiency, power output, and fault distribution across simulations

## Model Performance

- Training Accuracy: 95.64% (within target range of 95-98%)
- Testing Accuracy: 96.25% (exceeding target range of 90-95%)
- Class-specific Performance:
  - Healthy (Class 0): 98.77%
  - Fault 1 (Line-Line Fault): 95.06%
  - Fault 2 (Open Circuit): 95.00%
  - Fault 3 (Partial Shading): 97.50%
  - Fault 4 (Degradation): 94.87%

## Key Model Strengths

- **Strong Performance Areas**:
  - Fault_1 detection (100% accuracy)
  - Health detection with voltage deviations >1%
  - Positive high-current Fault_2 cases

- **Feature Importance**:
  - v_deviation: Critical for Health/Fault_1 distinction
  - i_deviation: Important for Fault_2 detection
  - power_deviation: Normalized for extreme values
  - z-scores: Amplified for voltage-based faults

## Running the Program

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Test Data** (Optional)
   ```bash
   python solar_fault_detection.py --generate-data --scenario random --count 10
   ```

3. **Start the Server**
   ```bash
   python solar_fault_detection.py --host 0.0.0.0 --port 5000
   ```

4. **Access the Dashboard**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

5. **Using the Dashboard**
   - Click "Start Monitoring" to begin real-time monitoring
   - View live data updates in the charts
   - Check fault predictions and confidence scores
   - Monitor alerts for detected faults
   - View performance statistics

6. **Manual Testing**
   - Use the "Test Panel" form to input voltage and current values
   - The system will automatically generate other parameters
   - View the prediction results and confidence scores

7. **Stopping the Server**
   Press Ctrl+C in the terminal to stop the server

## How to Use

1. **Manual Prediction**:
   - Enter PV Current and Voltage measurements
   - Click "Predict" to get instant fault analysis
   - System shows prediction with confidence level
   - Review recommended actions

2. **Real-time Monitoring**:
   - System automatically processes incoming measurements
   - Displays current status and alerts for faults
   - Maintains history of panel performance
   - Tracks system metrics and performance

3. **MATLAB Integration**:
   - Navigate to the MATLAB Dashboard
   - Adjust irradiance (200-1200 W/m²) and temperature (10-60°C) parameters
   - Run simulations to see how different conditions affect panel performance
   - Generate datasets with multiple samples for each fault type
   - View fault distribution and performance metrics in real-time charts

## Technical Details

### Dependencies
- Python 3.x
- PyTorch (Machine Learning)
- Flask (Web Interface)
- Pandas (Data Processing)
- NumPy (Numerical Operations)
- SQLAlchemy (Database)
- MATLAB Engine for Python (MATLAB Integration)

### Installation
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install MATLAB Engine for Python (if using MATLAB integration):
   ```bash
   # In MATLAB Command Window
   cd "matlabroot\extern\engines\python"
   python setup.py install
   ```

3. Start the application:
   ```bash
   python app.py
   ```

4. Access the web interface at: http://localhost:5000

5. For testing predictions:
   ```bash
   python test_predictions.py
   ```

6. For MATLAB integration testing:
   ```bash
   python matlab_integration.py
   ```

## Model Architecture

The fault detection model uses:
- Neural network with batch normalization
- Enhanced feature engineering with ratio features and class-specific indicators
- Multiple hidden layers for improved pattern recognition
- Optimized training process with learning rate scheduling
- Early stopping with increased patience for optimal convergence

## Fault Patterns

The system recognizes these typical patterns:
1. **Healthy Panel**: Normal current and voltage relationships
2. **Fault 1 (Line-Line Fault)**: Lower voltage, higher current
3. **Fault 2 (Open Circuit)**: Higher voltage, much lower current
4. **Fault 3 (Partial Shading)**: Slightly lower voltage, much lower current
5. **Fault 4 (Degradation)**: Normal voltage, lower current

## Project Structure

- **app.py**: Main Flask application with API endpoints
- **realtime_prediction.py**: Core prediction engine and model handling
- **preprocess_and_train.py**: Model training and preprocessing pipeline
- **database_setup.py**: Database schema and setup
- **test_predictions.py**: Test suite for validating predictions
- **matlab_interface.py**: Interface for connecting to MATLAB
- **matlab_integration.py**: Integration between MATLAB and the fault detection system
- **templates/**: Web interface HTML templates
- **static/**: CSS, JavaScript, and images for the web interface

## API Endpoints

- **/api/simple_predict**: Make predictions with PV current and voltage
- **/api/status**: Get system status and performance metrics
- **/api/fault_types**: Get information about fault types
- **/api/history**: Get historical prediction data
- **/api/matlab/run_simulation**: Run a MATLAB simulation with specified parameters
- **/api/matlab/get_simulations**: Get data from previous MATLAB simulations
- **/api/matlab/generate_dataset**: Generate a dataset with multiple simulation samples

## MATLAB Integration Details

The system integrates with MATLAB's GridConnectedPVFarm model to simulate solar panel behavior under various conditions:

- **Simulation Parameters**:
  - Irradiance: Controls the amount of solar energy (W/m²)
  - Temperature: Controls the cell temperature (°C)

- **Simulated Fault Conditions**:
  - **Line-Line Fault**: Simulated with normal irradiance but higher temperature
  - **Open Circuit**: Simulated with significantly reduced irradiance
  - **Partial Shading**: Simulated with moderately reduced irradiance and slightly higher temperature
  - **Degradation**: Simulated with moderately reduced irradiance and higher temperature

- **Simulation Outputs**:
  - PV Current and Voltage
  - PV Power and Grid Power
  - System Efficiency
  - Fault Prediction with Confidence Score

## Contributing

Feel free to contribute to this project by:
- Reporting issues
- Suggesting improvements
- Submitting pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.
