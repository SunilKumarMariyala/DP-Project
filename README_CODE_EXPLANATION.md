# Solar Panel Fault Detection System - Code Explanation

This document explains the code structure and functionality of the Solar Panel Fault Detection System, from basic to advanced concepts.

## Core Files Overview

### 1. app.py
This is the main entry point for the basic web application.

```python
# Main Flask application
from flask import Flask, render_template, request, jsonify
import logging
from solar_fault_detection import SolarFaultDetectionSystem

# Create Flask application
app = Flask(__name__)

# Initialize the fault detection system
fault_detector = SolarFaultDetectionSystem()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    data = request.get_json()
    result = fault_detector.predict_with_simple_inputs(
        data['pv_current'], 
        data['pv_voltage']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**What this does:**
- Creates a web server using Flask
- Sets up routes for the web interface and API
- Initializes the fault detection system
- Handles prediction requests

### 2. solar_fault_detection.py
This file contains the core functionality for detecting faults in solar panels.

```python
class SolarFaultDetectionSystem:
    def __init__(self):
        """Initialize the fault detection system"""
        # Load the trained model
        self.model = self.load_model()
        # Load the scaler for normalizing input data
        self.scaler = self.load_scaler()
        
    def predict(self, data):
        """Make a prediction based on input data"""
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        # Make prediction
        prediction = self.model.predict(processed_data)
        # Calculate confidence
        confidence = self.calculate_confidence(processed_data)
        # Return result
        return {
            'prediction': int(prediction[0]),
            'fault_type': self.get_fault_type(prediction[0]),
            'confidence': float(confidence[0])
        }
        
    def predict_with_simple_inputs(self, pv_current, pv_voltage):
        """Make prediction with just current and voltage"""
        # Generate synthetic data based on patterns
        synthetic_data = self.generate_synthetic_data(pv_current, pv_voltage)
        # Make prediction using the full data
        return self.predict(synthetic_data)
```

**What this does:**
- Loads the trained machine learning model
- Preprocesses input data for the model
- Makes predictions based on solar panel measurements
- Calculates confidence scores for predictions
- Provides a simplified interface for making predictions with just current and voltage

### 3. database_setup.py
This file handles database configuration and setup.

```python
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class SolarPanelData(Base):
    """Database model for storing solar panel data"""
    __tablename__ = 'solar_panel_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    pv_current = Column(Float)
    pv_voltage = Column(Float)
    power = Column(Float)
    temperature = Column(Float)
    irradiance = Column(Float)
    prediction = Column(Integer)
    confidence = Column(Float)
    
def setup_database(db_connection_str=None):
    """Setup the database for storing solar panel data"""
    # Get database connection from environment variables if not provided
    if db_connection_str is None:
        DB_HOST = os.environ.get('DB_HOST', 'localhost')
        DB_USER = os.environ.get('DB_USER', 'solar_user')
        DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_secure_password')
        DB_NAME = os.environ.get('DB_NAME', 'solar_panel_db')
        db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    
    # Create engine with connection pooling
    engine = create_engine(
        db_connection_str,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True
    )
    
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Create a session factory
    Session = sessionmaker(bind=engine)
    
    return engine, Session
```

**What this does:**
- Defines the database schema for storing solar panel data
- Creates a connection to the MySQL database
- Sets up connection pooling for better performance
- Creates tables if they don't exist
- Returns an engine and session factory for database operations

### 4. matlab_interface.py
This file provides integration with MATLAB for simulations and data generation.

```python
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class MatlabInterface:
    def __init__(self, matlab_path=None, model_path=None, db_connection_str=None):
        """Initialize the MATLAB interface"""
        # Get paths from environment variables if not provided
        self.matlab_path = matlab_path or os.environ.get('MATLAB_PATH')
        self.model_path = model_path or os.environ.get('MATLAB_MODEL_PATH')
        
        # Get database connection from environment variables if not provided
        if db_connection_str is None:
            self.db_connection_str = os.environ.get('DB_CONNECTION_STR')
            if self.db_connection_str is None:
                logger.error("Database connection string not provided")
                raise ValueError("Database connection string not provided")
        else:
            self.db_connection_str = db_connection_str
            
        # Try to initialize MATLAB engine
        self.matlab_available = False
        self.eng = None
        try:
            import matlab.engine
            self.eng = matlab.engine.start_matlab()
            self.matlab_available = True
            logger.info("MATLAB engine started successfully")
        except Exception as e:
            logger.warning(f"Could not start MATLAB engine: {e}")
            
    def run_simulation(self, irradiance=1000, temperature=25):
        """Run a simulation with the specified parameters"""
        if not self.matlab_available:
            logger.error("MATLAB is not available")
            return None
            
        try:
            # Add model path to MATLAB path
            self.eng.addpath(self.model_path)
            
            # Run simulation
            result = self.eng.run_pv_simulation(irradiance, temperature)
            
            # Convert result to Python dictionary
            return self.convert_matlab_result(result)
        except Exception as e:
            logger.error(f"Error running MATLAB simulation: {e}")
            return None
            
    def process_real_time_matlab_data(self, matlab_data_file):
        """Process real-time data from MATLAB"""
        try:
            # Check file extension
            file_ext = os.path.splitext(matlab_data_file)[1].lower()
            
            # Load data based on file type
            if file_ext == '.mat':
                # Load MAT file
                data = self.load_mat_file(matlab_data_file)
            elif file_ext == '.csv':
                # Load CSV file
                data = pd.read_csv(matlab_data_file)
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return None
                
            # Process data
            return self.process_data(data)
        except Exception as e:
            logger.error(f"Error processing MATLAB data: {e}")
            return None
```

**What this does:**
- Initializes a connection to MATLAB
- Runs simulations with specified parameters
- Processes data from MATLAB files (both MAT and CSV formats)
- Converts MATLAB results to Python data structures
- Handles errors and logging

### 5. matlab_continuous_demo.py
This file implements continuous data flow from MATLAB to the prediction system.

```python
import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from solar_fault_detection import SolarFaultDetectionSystem
from matlab_interface import MatlabInterface

class MatlabFileHandler(FileSystemEventHandler):
    def __init__(self, matlab_interface, fault_detector):
        self.matlab_interface = matlab_interface
        self.fault_detector = fault_detector
        self.processed_files = set()
        
    def on_created(self, event):
        """Handle new file creation events"""
        if event.is_directory:
            return
            
        # Check if file is a MATLAB data file
        file_path = event.src_path
        if file_path.endswith('.mat') or file_path.endswith('.csv'):
            # Wait for file to be completely written
            time.sleep(1)
            
            # Process the file if not already processed
            if file_path not in self.processed_files:
                self.process_file(file_path)
                self.processed_files.add(file_path)
                
    def process_file(self, file_path):
        """Process a MATLAB data file"""
        try:
            # Process the file with MATLAB interface
            data = self.matlab_interface.process_real_time_matlab_data(file_path)
            
            if data is not None:
                # Make prediction
                prediction = self.fault_detector.predict(data)
                
                # Log the prediction
                logging.info(f"Prediction for {file_path}: {prediction}")
                
                # Save to database
                self.matlab_interface.save_prediction_to_db(data, prediction)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

def main():
    """Main function for continuous data flow demo"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    matlab_interface = MatlabInterface()
    fault_detector = SolarFaultDetectionSystem()
    
    # Get watch directory from environment variable
    watch_dir = os.environ.get('MATLAB_OUTPUT_DIR', '.')
    
    # Create event handler and observer
    event_handler = MatlabFileHandler(matlab_interface, fault_detector)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    
    # Start the observer
    observer.start()
    logging.info(f"Watching directory {watch_dir} for MATLAB data files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    main()
```

**What this does:**
- Watches a directory for new MATLAB data files
- Processes new files as they appear
- Makes predictions based on the data
- Saves predictions to the database
- Handles errors and logging

## Advanced Concepts

### Machine Learning Model

The system uses a neural network model trained on solar panel data to detect faults:

```python
def create_model(input_dim):
    """Create a neural network model for fault detection"""
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(32, 16),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 5)  # 5 classes: Healthy + 4 fault types
    )
    return model
```

**Key features:**
- Multiple layers for complex pattern recognition
- Batch normalization for stable training
- Dropout for preventing overfitting
- ReLU activation functions for non-linearity

### Feature Engineering

The system generates additional features from the basic measurements:

```python
def generate_features(data):
    """Generate additional features from basic measurements"""
    features = {}
    
    # Basic measurements
    features['pv_current'] = data['pv_current']
    features['pv_voltage'] = data['pv_voltage']
    
    # Calculated features
    features['power'] = data['pv_current'] * data['pv_voltage']
    
    # Reference values (typical for healthy panel)
    ref_current = 8.0
    ref_voltage = 48.0
    ref_power = ref_current * ref_voltage
    
    # Deviation features
    features['current_deviation'] = (data['pv_current'] - ref_current) / ref_current
    features['voltage_deviation'] = (data['pv_voltage'] - ref_voltage) / ref_voltage
    features['power_deviation'] = (features['power'] - ref_power) / ref_power
    
    # Ratio features
    features['current_voltage_ratio'] = data['pv_current'] / data['pv_voltage']
    
    # Z-score features (if temperature and irradiance available)
    if 'temperature' in data and 'irradiance' in data:
        features['temp_irr_ratio'] = data['temperature'] / data['irradiance']
        
    return features
```

**What this does:**
- Calculates power from current and voltage
- Computes deviations from reference values
- Creates ratio features for better pattern detection
- Generates statistical features like z-scores

### Database Operations

The system uses SQLAlchemy for database operations:

```python
def save_data_to_db(data, prediction):
    """Save data and prediction to database"""
    # Get database session
    _, Session = setup_database()
    session = Session()
    
    try:
        # Create new record
        record = SolarPanelData(
            timestamp=datetime.now(),
            pv_current=data['pv_current'],
            pv_voltage=data['pv_voltage'],
            power=data['power'] if 'power' in data else data['pv_current'] * data['pv_voltage'],
            temperature=data.get('temperature', 25.0),
            irradiance=data.get('irradiance', 1000.0),
            prediction=prediction['prediction'],
            confidence=prediction['confidence']
        )
        
        # Add and commit
        session.add(record)
        session.commit()
        
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error saving to database: {e}")
        return False
    finally:
        session.close()
```

**What this does:**
- Creates a database session
- Creates a new record with data and prediction
- Handles transactions with commit and rollback
- Properly closes the session to prevent leaks

### Web API

The system provides a RESTful API for integration:

```python
@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for making predictions"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        if 'pv_current' not in data or 'pv_voltage' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Make prediction
        result = fault_detector.predict_with_simple_inputs(
            float(data['pv_current']), 
            float(data['pv_voltage'])
        )
        
        # Return result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status_api():
    """API endpoint for getting system status"""
    return jsonify({
        'status': 'online',
        'monitoring_active': monitoring_active,
        'last_update': last_update.isoformat() if last_update else None,
        'total_predictions': total_predictions,
        'fault_distribution': fault_distribution
    })
```

**What this does:**
- Defines API endpoints for predictions and status
- Validates input data
- Handles errors and returns appropriate status codes
- Returns JSON responses for easy integration

## Code Organization Principles

### Modular Design
The code is organized into modules with specific responsibilities:
- **app.py**: Web interface and API
- **solar_fault_detection.py**: Core prediction logic
- **database_setup.py**: Database configuration
- **matlab_interface.py**: MATLAB integration
- **matlab_continuous_demo.py**: Continuous data flow

### Error Handling
The code includes comprehensive error handling:
```python
try:
    # Operation that might fail
    result = some_function()
    return result
except SpecificError as e:
    # Handle specific error
    logging.error(f"Specific error: {e}")
    return fallback_value
except Exception as e:
    # Handle any other error
    logging.error(f"Unexpected error: {e}")
    raise
```

### Logging
The code uses Python's logging module for tracking events:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)

# Create logger
logger = logging.getLogger(__name__)

# Log messages
logger.info("Application started")
logger.warning("Unusual value detected")
logger.error("Failed to connect to database")
```

### Configuration Management
The code uses environment variables for configuration:
```python
import os

# Get configuration from environment variables
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'solar_user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_secure_password')
DB_NAME = os.environ.get('DB_NAME', 'solar_panel_db')
```

## Next Steps for Code Improvement

1. **Add Unit Tests**: Create comprehensive test coverage
2. **Implement Caching**: Add Redis for caching frequent database queries
3. **Add Authentication**: Implement user authentication for the API
4. **Improve Documentation**: Add docstrings to all functions and classes
5. **Optimize Performance**: Profile and optimize slow operations
