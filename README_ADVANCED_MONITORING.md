# Advanced Solar Fault Detection Monitoring System

This advanced monitoring system enhances the solar fault detection capabilities with real-time monitoring, sophisticated visualization, and comprehensive fault analysis.

## Features

- **Real-time Monitoring**: Continuous processing of solar panel data with immediate fault detection
- **Advanced Dashboard**: Interactive visualization of panel status, fault distribution, and model confidence
- **Comprehensive Alerting**: Severity-based alerts with acknowledgment system
- **Detailed Analytics**: Performance metrics and fault pattern analysis
- **MATLAB Integration**: Seamless processing of MATLAB simulation data
- **Maintenance Logging**: Track maintenance actions and their effectiveness

## System Architecture

The system consists of the following components:

1. **Data Processor**: Processes solar panel data in real-time, applying the enhanced model for fault detection
2. **Socket.IO Server**: Provides real-time updates to the dashboard
3. **Flask API**: Serves the dashboard and provides data endpoints
4. **SQLite Database**: Stores panel data, predictions, alerts, and maintenance logs
5. **MATLAB File Watcher**: Monitors for new MATLAB simulation data

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Flask
- Socket.IO
- Pandas, NumPy, Matplotlib
- SQLite

### Installation

1. Ensure all dependencies are installed:
   ```
   pip install torch flask flask-socketio flask-cors pandas numpy matplotlib watchdog
   ```

2. Ensure the enhanced model is trained and available:
   - The system looks for `enhanced_model_best.pth` and `enhanced_scaler.pkl`
   - If not found, it will initialize a new model

3. Set up the database:
   - The system will automatically create the required tables if they don't exist

## Running the System

### Starting the Monitoring System

1. Start the Flask application:
   ```
   python app.py
   ```

2. Access the advanced dashboard:
   - Open a web browser and navigate to `http://localhost:5000/advanced_dashboard`

3. Start the monitoring process:
   - Click the "Start Processing" button on the dashboard
   - Or use the API endpoint: `POST /api/advanced/start`

### Generating Test Data

Use the data generator to create synthetic data for testing:

```
python data_generator.py --mode continuous --interval 5 --scenario random
```

Options:
- `--mode`: `batch` or `continuous`
- `--count`: Number of records to generate in batch mode
- `--interval`: Time between records in seconds
- `--duration`: Duration in seconds for continuous mode
- `--scenario`: `random`, `healthy`, `fault_1`, `fault_2`, `fault_3`, or `fault_4`

## API Endpoints

### Dashboard

- `GET /advanced_dashboard`: Access the advanced monitoring dashboard

### Monitoring Control

- `POST /api/advanced/start`: Start the monitoring process
  - Parameters: `interval` (seconds), `batch_size` (records)
- `POST /api/advanced/stop`: Stop the monitoring process

### Data Access

- `GET /api/data/latest`: Get the latest processed data
  - Parameters: `limit` (number of records)
- `GET /api/alerts/latest`: Get the latest alerts
  - Parameters: `limit` (number of alerts)
- `POST /api/alerts/acknowledge`: Acknowledge an alert
  - Parameters: `alert_id`
- `GET /api/stats/summary`: Get summary statistics

### MATLAB Integration

- `POST /api/matlab/setup_watch`: Set up the MATLAB data watch
  - Parameters: `watch_directory`, `file_pattern`
- `POST /api/matlab/stop_watch`: Stop the MATLAB data watch

## Dashboard Features

### Current Status Panel

Displays the current status of the solar panel, including:
- Fault prediction with confidence
- Current voltage, current, and power readings
- Recommended action

### Real-time Monitoring Chart

Shows the voltage and current trends over time, updating in real-time.

### Fault Distribution Chart

Visualizes the distribution of fault types detected by the system.

### Model Confidence Chart

Displays the average confidence level for each fault type.

### Active Alerts Panel

Lists all active alerts, sorted by severity, with the ability to acknowledge them.

### Recent Predictions Panel

Shows the most recent predictions made by the system.

### System Performance Panel

Displays key performance metrics:
- System uptime
- Model accuracy
- Panel efficiency

### Control Panel

Provides buttons to:
- Start/stop processing
- Refresh data
- Acknowledge all alerts

### Prediction History Table

Lists all predictions with detailed information and the ability to log maintenance actions.

## Fault Types

1. **Line-Line Fault (Fault 1)**: Short circuit between conductors
   - Characteristics: Lower voltage, higher current
   - Recommended action: Inspect wiring for damaged insulation

2. **Open Circuit (Fault 2)**: Current flow is interrupted
   - Characteristics: Higher voltage, near-zero current
   - Recommended action: Check connections, fuses, and circuit breakers

3. **Partial Shading (Fault 3)**: Reduced current with normal voltage
   - Characteristics: Slightly lower voltage, moderately lower current
   - Recommended action: Check for physical obstructions

4. **Panel Degradation (Fault 4)**: Gradual reduction in both current and voltage
   - Characteristics: Lower voltage, lower current
   - Recommended action: Schedule maintenance or panel replacement

## Performance Metrics

The advanced monitoring system achieves:
- Training accuracy: 95.64%
- Testing accuracy: 96.25%

Class-specific performance:
- Healthy (Class 0): 98.77%
- Fault 1: 95.06%
- Fault 2: 95.00%
- Fault 3: 97.50%
- Fault 4: 94.87%

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Ensure the SQLite database file exists and is accessible
   - Check file permissions

2. **Model Loading Errors**:
   - Verify that the model and scaler files exist
   - Ensure PyTorch is properly installed

3. **Real-time Updates Not Working**:
   - Check that Socket.IO is properly configured
   - Ensure the browser supports WebSockets

4. **MATLAB Integration Issues**:
   - Verify MATLAB is properly installed and configured
   - Check the watch directory exists and is accessible

## Future Enhancements

1. **Predictive Maintenance**: Predict when faults are likely to occur before they happen
2. **Mobile Notifications**: Send alerts to mobile devices
3. **Weather Integration**: Incorporate weather data for more accurate predictions
4. **Expanded Analytics**: More detailed performance analysis and reporting
5. **Multi-panel Support**: Monitor multiple solar panel arrays simultaneously
