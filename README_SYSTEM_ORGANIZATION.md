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
