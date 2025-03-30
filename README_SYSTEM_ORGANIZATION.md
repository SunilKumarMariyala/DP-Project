# Solar Fault Detection System - Organization Guide

## Project Overview

This project consists of two main components:

1. **Basic Solar Fault Detection System** - The core system with fundamental monitoring capabilities
2. **Advanced Solar Fault Detection System** - An enhanced version with additional features and improved visualization

Both systems share the same underlying model and database but are organized into separate folders for clarity and ease of use.

## System Organization

### Main Directory

The main project directory contains the core implementation files that are shared between both systems:

```
c:\Users\Sunil Kumar\OneDrive\Documents\solar faults\maybe final model\
├── app.py                   # Main application for basic system
├── enhanced_model.py        # Enhanced model implementation
├── data_generator.py        # Data generation for testing
├── realtime_prediction.py   # Real-time prediction functionality
├── database_setup.py        # Database initialization
├── solar_fault_detection_model.pth  # Trained model
├── scaler.pkl               # Feature scaler
├── solar_panel.db           # Database
└── templates/               # Templates for basic system
    └── dashboard.html       # Basic dashboard
```

### Basic System Folder

The basic system folder contains the organizational structure for the basic monitoring system:

```
basic_solar_fault_detection/
├── README.md                # Documentation for basic system
├── run_basic_system.bat     # Script to run the basic system
├── generate_test_data.bat   # Script to generate test data
├── static/                  # Static assets for basic dashboard
│   ├── dashboard.css        # Custom CSS for basic dashboard
│   ├── dashboard.js         # JavaScript for basic dashboard
│   └── favicon.ico          # Favicon for basic system
└── templates/               # HTML templates
    └── dashboard.html       # Basic dashboard template
```

### Advanced System Folder

The advanced system folder contains a standalone implementation of the advanced monitoring system:

```
advanced_solar_fault_detection/
├── DOCUMENTATION.md         # Detailed documentation
├── README.md                # Quick start guide
├── app.py                   # Standalone web application
├── advanced_monitoring.py   # Core monitoring implementation
├── data_generator.py        # Test data generator
├── run_advanced_system.bat  # Script to run the advanced system
├── generate_test_data.bat   # Script to generate test data
├── static/                  # Static assets
│   ├── dashboard.css        # Custom CSS for advanced dashboard
│   ├── dashboard.js         # JavaScript for advanced dashboard
│   └── favicon.ico          # Favicon for advanced system
└── templates/               # HTML templates
    └── dashboard.html       # Advanced dashboard template
```

## Running the Project

To run the Solar Fault Detection System, follow these steps:

### Step 1: Set up the Database

First, run the database setup script to create the database and load sample data:

```bash
python database_setup.py
```

This will:
- Create the SQLite database file (solar_panel.db)
- Define the database schema
- Load sample data from the Excel file
- Verify the data was loaded correctly

### Step 2: Run the Application

You have two options for running the application:

#### Option 1: Run the Full Application

```bash
python app.py
```

This starts the full application with all features, including:
- Advanced dashboard
- MATLAB integration (if available)
- Enhanced monitoring capabilities
- Complete API endpoints

#### Option 2: Run the Simplified Version

```bash
python solar_fault_detection.py
```

This starts a simplified version of the application with core functionality:
- Basic dashboard
- Core prediction capabilities
- Essential monitoring features

### Step 3: Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:5000
```

This will display the dashboard interface where you can:
- Make manual predictions
- View real-time monitoring data
- Access historical data
- See performance statistics

## System Interactions

The following diagram illustrates how the main components of the system interact:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Web Interface  │<─────│  Flask App      │<─────│  Database       │
│  (Dashboard)    │─────>│  (app.py)       │─────>│  (SQLite)       │
│                 │      │                 │      │                 │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
                                  │
                                  │
                         ┌────────▼────────┐
                         │                 │
                         │  ML Model       │
                         │  (PyTorch)      │
                         │                 │
                         └────────┬────────┘
                                  │
                                  │
                         ┌────────▼────────┐
                         │                 │
                         │  MATLAB         │
                         │  Integration    │
                         │  (Optional)     │
                         │                 │
                         └─────────────────┘
```

### Key Interactions:

1. **User → Web Interface**: User inputs data or requests information through the dashboard
2. **Web Interface → Flask App**: Dashboard sends requests to the Flask application
3. **Flask App → ML Model**: Application uses the ML model to make predictions
4. **Flask App → Database**: Application stores and retrieves data from the SQLite database
5. **Flask App → MATLAB Integration**: Application can optionally connect to MATLAB for simulations
6. **Flask App → Web Interface**: Application sends results back to the dashboard for display

This architecture ensures:
- Clean separation of concerns
- Modular design for easy maintenance
- Scalability for future enhancements
- Optional components that don't break core functionality

## Usage Guide

### Running the Basic System

1. Navigate to the basic system folder:
   ```
   cd "c:\Users\Sunil Kumar\OneDrive\Documents\solar faults\maybe final model\basic_solar_fault_detection"
   ```

2. Run the basic system:
   ```
   run_basic_system.bat
   ```

3. Access the dashboard at:
   ```
   http://localhost:5000
   ```

### Running the Advanced System

1. Navigate to the advanced system folder:
   ```
   cd "c:\Users\Sunil Kumar\OneDrive\Documents\solar faults\maybe final model\advanced_solar_fault_detection"
   ```

2. Run the advanced system:
   ```
   run_advanced_system.bat
   ```

3. Access the dashboard at:
   ```
   http://localhost:5001
   ```

## Integration Between Systems

The advanced system can be integrated with the basic system by importing the advanced monitoring module:

```python
from advanced_solar_fault_detection.advanced_monitoring import AdvancedMonitoring

# Initialize the advanced monitoring
advanced_monitor = AdvancedMonitoring()

# Start monitoring
advanced_monitor.start()
```

## Model Performance

The solar fault detection model has been improved to achieve:
- Training accuracy: 95.64% (within target range of 95-98%)
- Testing accuracy: 96.25% (exceeding target range of 90-95%)

Class-specific performance:
- Healthy (Class 0): 98.77%
- Fault 1 (Line-Line Fault): 95.06%
- Fault 2 (Open Circuit): 95.00%
- Fault 3 (Partial Shading): 97.50%
- Fault 4 (Panel Degradation): 94.87%

## Development Guidelines

When developing or modifying the system:

1. **Shared Resources**: 
   - Model files, database, and core functionality remain in the main directory
   - Both systems reference these shared resources

2. **Basic System**:
   - Modifications to the basic system should be made in the main directory files
   - UI/UX enhancements can be made in the basic_solar_fault_detection folder

3. **Advanced System**:
   - The advanced system is designed to run independently
   - It has its own implementation files in the advanced_solar_fault_detection folder

4. **Testing**:
   - Use the generate_test_data.bat scripts in each folder to test the respective systems
   - Ensure both systems can access the shared database
