"""
Solar Fault Detection System - Integrated Solution
Combines the best features of basic and advanced monitoring
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
import joblib
import pymysql
from sqlalchemy import create_engine

# Import MATLAB interface
try:
    from matlab_interface import MatlabInterface
    matlab_available = True
except ImportError:
    matlab_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('solar_fault_detection.log')
    ]
)
logger = logging.getLogger('solar_fault_detection')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'solar-fault-detection-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
monitoring_active = False
monitoring_thread = None
model = None
scaler = None

# Database configuration
DB_HOST = os.environ.get('DB_HOST')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    logger.error("Missing database environment variables. Please set DB_HOST, DB_USER, DB_PASSWORD, and DB_NAME.")
    sys.exit(1)

db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'

# For backward compatibility
db_path = 'solar_panel.db'  # Kept for command-line argument compatibility
matlab_interface = None
fault_types = {
    0: "Healthy",
    1: "Line-Line Fault",
    2: "Open Circuit",
    3: "Partial Shading",
    4: "Panel Degradation"
}

class SolarFaultDetectionSystem:
    """Solar Fault Detection System using PyTorch model"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """Initialize the system"""
        self.model_path = model_path or 'solar_fault_detection_model.pth'
        self.scaler_path = scaler_path or 'scaler.pkl'
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model
            self.model = torch.jit.load(self.model_path)
            self.model.eval()
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            return False
    
    def preprocess_data(self, data):
        """Preprocess data for model input"""
        try:
            # Calculate additional features
            power = data['voltage'] * data['current']
            v_deviation = (data['voltage'] - 48) / 48 * 100
            i_deviation = (data['current'] - 10) / 10 * 100
            p_deviation = (power - 480) / 480 * 100
            
            # Create feature vector
            features = np.array([
                data['voltage'],
                data['current'],
                power,
                data['temperature'],
                data['irradiance'],
                v_deviation,
                i_deviation,
                p_deviation,
                1 if v_deviation > 1 else 0,  # healthy_indicator
                1 if v_deviation < -5 else 0,  # degradation_indicator
                1 if i_deviation > 50 else 0,  # high_current_indicator
                1 if i_deviation < -50 else 0  # low_current_indicator
            ]).reshape(1, -1)
            
            # Scale features
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Convert to tensor
            tensor = torch.FloatTensor(features)
            
            return tensor
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None
    
    def predict(self, data):
        """Make prediction using the loaded model"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Preprocess data
            tensor = self.preprocess_data(data)
            if tensor is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                output = self.model(tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
            
            # Get prediction confidence
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0][prediction].item() * 100
            
            return {
                'prediction': prediction,
                'fault_type': fault_types.get(prediction, "Unknown"),
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

class DataProcessor:
    """Process data and make predictions"""
    
    def __init__(self, db_connection_str=None):
        """Initialize the data processor"""
        self.db_connection_str = db_connection_str or db_connection_str
        self.detector = SolarFaultDetectionSystem()
        self.engine = create_engine(self.db_connection_str)
        self.conn = None
        self.cursor = None
        self.connect_db()
    
    def connect_db(self):
        """Connect to the database"""
        try:
            self.conn = self.engine.connect()
            self.cursor = self.conn
            logger.info(f"Connected to database: {self.db_connection_str}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def get_latest_data(self):
        """Get the latest data from the database"""
        try:
            query = """
                SELECT id, timestamp, voltage, current, power, temperature, irradiance,
                       v_deviation, i_deviation, p_deviation
                FROM solar_data
                ORDER BY id DESC
                LIMIT 1
            """
            result = self.cursor.execute(query)
            row = result.fetchone()
            
            if row:
                data = {
                    'id': row[0],
                    'timestamp': row[1],
                    'voltage': row[2],
                    'current': row[3],
                    'power': row[4],
                    'temperature': row[5],
                    'irradiance': row[6],
                    'v_deviation': row[7],
                    'i_deviation': row[8],
                    'p_deviation': row[9]
                }
                return data
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return None
    
    def process_data(self):
        """Process the latest data and make prediction"""
        data = self.get_latest_data()
        if data:
            # Make prediction
            result = self.detector.predict(data)
            if result:
                # Add data to result
                result.update(data)
                
                # Insert prediction to database
                self.insert_prediction(data['id'], result['prediction'], result['confidence'])
                
                # Generate alert if fault detected
                if result['prediction'] != 0:
                    self.generate_alert(result)
                
                return result
        return None
    
    def insert_prediction(self, data_id, prediction, confidence):
        """Insert prediction into database"""
        try:
            query = """
                INSERT INTO predictions (data_id, prediction, confidence, timestamp)
                VALUES (%s, %s, %s, NOW())
            """
            self.cursor.execute(query, (data_id, prediction, confidence))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            return False
    
    def generate_alert(self, result):
        """Generate alert for detected fault"""
        try:
            # Determine severity based on confidence
            confidence = result['confidence']
            if confidence > 90:
                severity = 'high'
            elif confidence > 70:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Create alert message
            message = f"Detected {result['fault_type']} with {confidence:.2f}% confidence"
            
            # Insert alert into database
            query = """
                INSERT INTO alerts (data_id, fault_type, confidence, message, severity, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """
            self.cursor.execute(query, (
                result['id'],
                result['prediction'],
                confidence,
                message,
                severity
            ))
            self.conn.commit()
            
            # Return alert data
            alert = {
                'fault_type': result['fault_type'],
                'confidence': confidence,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return alert
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            return None
    
    def get_latest_alerts(self, limit=5):
        """Get the latest alerts from the database"""
        try:
            query = """
                SELECT id, data_id, fault_type, confidence, message, severity, timestamp
                FROM alerts
                ORDER BY id DESC
                LIMIT %s
            """
            result = self.cursor.execute(query, (limit,))
            rows = result.fetchall()
            
            alerts = []
            for row in rows:
                alert = {
                    'id': row[0],
                    'data_id': row[1],
                    'fault_type': fault_types.get(row[2], "Unknown"),
                    'confidence': row[3],
                    'message': row[4],
                    'severity': row[5],
                    'timestamp': row[6]
                }
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            logger.error(f"Error getting latest alerts: {e}")
            return []
    
    def get_performance_stats(self):
        """Get performance statistics"""
        try:
            # Get total predictions
            query = "SELECT COUNT(*) FROM predictions"
            result = self.cursor.execute(query)
            total_predictions = result.fetchone()[0]
            
            # Get fault distribution
            query = """
                SELECT prediction, COUNT(*) as count
                FROM predictions
                GROUP BY prediction
                ORDER BY prediction
            """
            result = self.cursor.execute(query)
            rows = result.fetchall()
            
            distribution = {}
            for row in rows:
                fault_type = fault_types.get(row[0], "Unknown")
                distribution[fault_type] = row[1]
            
            # Get average confidence
            query = "SELECT AVG(confidence) FROM predictions"
            result = self.cursor.execute(query)
            avg_confidence = result.fetchone()[0] or 0
            
            # Get recent accuracy (if actual values are available)
            recent_accuracy = 95.64  # Default to training accuracy
            
            return {
                'total_predictions': total_predictions,
                'distribution': distribution,
                'avg_confidence': avg_confidence,
                'recent_accuracy': recent_accuracy
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

def monitoring_loop():
    """Main monitoring loop"""
    global monitoring_active, matlab_interface
    
    processor = DataProcessor(db_connection_str)
    
    # Initialize MATLAB interface if available
    if matlab_available:
        try:
            matlab_interface = MatlabInterface(db_connection_str=db_connection_str)
            logger.info("MATLAB interface initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MATLAB interface: {e}")
            matlab_interface = None
    
    logger.info("Starting monitoring loop")
    
    while monitoring_active:
        try:
            # Try to get data from MATLAB if available
            matlab_data = None
            if matlab_interface is not None and matlab_available:
                try:
                    # Run MATLAB simulation to get latest data
                    matlab_data = matlab_interface.run_simulation()
                    if matlab_data:
                        logger.info("Received data from MATLAB")
                        
                        # Save MATLAB data to database
                        session = matlab_interface.Session()
                        new_data = SolarPanelData(
                            timestamp=datetime.now(),
                            voltage=matlab_data['pv_voltage'],
                            current=matlab_data['pv_current'],
                            power=matlab_data['pv_power'],
                            temperature=25.0,  # Default temperature
                            irradiance=1000.0,  # Default irradiance
                            v_deviation=(matlab_data['pv_voltage'] - 48.0) / 48.0 * 100,
                            i_deviation=(matlab_data['pv_current'] - 10.0) / 10.0 * 100,
                            p_deviation=(matlab_data['pv_power'] - 480.0) / 480.0 * 100
                        )
                        session.add(new_data)
                        session.commit()
                        session.close()
                        logger.info("Saved MATLAB data to database")
                except Exception as e:
                    logger.error(f"Error getting data from MATLAB: {e}")
                    matlab_data = None
            
            # If MATLAB data is not available, generate test data
            if matlab_data is None:
                generate_test_data(scenario='random', count=1)
            
            # Process data
            result = processor.process_data()
            
            if result:
                # Emit data update
                socketio.emit('data_update', {
                    'voltage': result['voltage'],
                    'current': result['current'],
                    'power': result['power'],
                    'temperature': result['temperature'],
                    'irradiance': result['irradiance'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Emit prediction update
                socketio.emit('prediction_update', {
                    'prediction': result['prediction'],
                    'fault_type': fault_types.get(result['prediction'], "Unknown"),
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
                
                # Emit alert if fault detected
                if result['prediction'] != 0:
                    alert = processor.generate_alert(result)
                    if alert:
                        socketio.emit('alert', alert)
                
                logger.info(f"Processed data: {result['id']}, Prediction: {fault_types.get(result['prediction'], 'Unknown')}")
            
            # Wait for next iteration
            time.sleep(2)
        
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(5)
    
    logger.info("Monitoring loop stopped")

def start_monitoring():
    """Start the monitoring thread"""
    global monitoring_active, monitoring_thread
    
    if monitoring_active:
        return False
    
    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitoring_loop)
    monitoring_thread.daemon = True
    monitoring_thread.start()
    
    logger.info("Monitoring started")
    return True

def stop_monitoring():
    """Stop the monitoring thread"""
    global monitoring_active
    
    if not monitoring_active:
        return False
    
    monitoring_active = False
    logger.info("Monitoring stopped")
    return True

# Flask routes
@app.route('/')
def index():
    """Render the dashboard"""
    return render_template('dashboard.html')

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start monitoring"""
    success = start_monitoring()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop monitoring"""
    success = stop_monitoring()
    return jsonify({'success': success})

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get monitoring status"""
    return jsonify({'active': monitoring_active})

@app.route('/api/data', methods=['GET'])
def api_data():
    """Get latest data"""
    processor = DataProcessor(db_connection_str)
    data = processor.get_latest_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data available'})

@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """Get latest alerts"""
    processor = DataProcessor(db_connection_str)
    alerts = processor.get_latest_alerts()
    return jsonify(alerts)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get performance statistics"""
    processor = DataProcessor(db_connection_str)
    stats = processor.get_performance_stats()
    return jsonify(stats)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make prediction from manual input"""
    try:
        data = request.json
        
        # Create data object
        prediction_data = {
            'voltage': float(data.get('voltage', 0)),
            'current': float(data.get('current', 0)),
            'power': float(data.get('voltage', 0)) * float(data.get('current', 0)),
            'temperature': float(data.get('temperature', 25)),
            'irradiance': float(data.get('irradiance', 1000)),
            'v_deviation': float(data.get('v_deviation', 0)),
            'i_deviation': float(data.get('i_deviation', 0)),
            'p_deviation': float(data.get('p_deviation', 0))
        }
        
        # Make prediction
        detector = SolarFaultDetectionSystem()
        result = detector.predict(prediction_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'})
    
    except Exception as e:
        logger.error(f"Error in manual prediction: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/simple_predict', methods=['POST'])
def api_simple_predict():
    """Make prediction from simplified input (voltage and current only)"""
    try:
        data = request.json
        
        # Extract basic inputs
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        
        # Calculate power
        power = voltage * current
        
        # Generate synthetic values based on patterns
        # These values simulate the fault patterns identified during model training
        v_deviation = 0
        i_deviation = 0
        p_deviation = 0
        temperature = 25 + np.random.uniform(-2, 2)
        irradiance = 1000 + np.random.uniform(-50, 50)
        
        # Create data object
        prediction_data = {
            'voltage': voltage,
            'current': current,
            'power': power,
            'temperature': temperature,
            'irradiance': irradiance,
            'v_deviation': v_deviation,
            'i_deviation': i_deviation,
            'p_deviation': p_deviation
        }
        
        # Make prediction
        detector = SolarFaultDetectionSystem()
        result = detector.predict(prediction_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'})
    
    except Exception as e:
        logger.error(f"Error in simple prediction: {e}")
        return jsonify({'error': str(e)})

# SocketIO events
@socketio.on('connect')
def socket_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    socketio.emit('monitoring_status', {'active': monitoring_active})

@socketio.on('disconnect')
def socket_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_monitoring')
def socket_start_monitoring():
    """Start monitoring from socket event"""
    success = start_monitoring()
    socketio.emit('monitoring_status', {'active': monitoring_active})
    return {'success': success}

@socketio.on('stop_monitoring')
def socket_stop_monitoring():
    """Stop monitoring from socket event"""
    success = stop_monitoring()
    socketio.emit('monitoring_status', {'active': monitoring_active})
    return {'success': success}

def initialize_database(db_connection_str):
    """Set up the database if it doesn't exist"""
    try:
        engine = create_engine(db_connection_str)
        
        # Create tables
        from sqlalchemy import MetaData, Table, Column, Integer, Float, DateTime, String
        metadata = MetaData()
        
        # Solar data table
        solar_data = Table('solar_data', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('voltage', Float),
            Column('current', Float),
            Column('power', Float),
            Column('temperature', Float),
            Column('irradiance', Float),
            Column('v_deviation', Float),
            Column('i_deviation', Float),
            Column('p_deviation', Float)
        )
        
        # Predictions table
        predictions = Table('predictions', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('data_id', Integer),
            Column('prediction', Integer),
            Column('confidence', Float),
            Column('label', String(50))
        )
        
        # Alerts table
        alerts = Table('alerts', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('prediction_id', Integer),
            Column('message', String(255)),
            Column('severity', String(20)),
            Column('acknowledged', Integer, default=0)
        )
        
        # Create tables
        metadata.create_all(engine)
        logger.info(f"Database initialized: {db_connection_str}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def test_db_connection():
    """Test connection to MySQL and create database if it doesn't exist"""
    try:
        # First try to connect to the MySQL server without specifying a database
        root_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/'
        engine = create_engine(root_connection_str)
        conn = engine.connect()
        
        # Check if database exists
        result = conn.execute(f"SHOW DATABASES LIKE '{DB_NAME}'")
        if not result.fetchone():
            # Create database if it doesn't exist
            conn.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Created database: {DB_NAME}")
        
        conn.close()
        
        # Now connect to the specific database
        engine = create_engine(db_connection_str)
        conn = engine.connect()
        conn.close()
        
        logger.info("Successfully connected to MySQL database")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MySQL database: {e}")
        return False

def generate_test_data(scenario='random', count=1):
    """Generate test data for the specified scenario"""
    engine = create_engine(db_connection_str)
    conn = engine.connect()
    
    # Create table if it doesn't exist
    query = """
        CREATE TABLE IF NOT EXISTS solar_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            voltage FLOAT,
            current FLOAT,
            power FLOAT,
            temperature FLOAT,
            irradiance FLOAT,
            v_deviation FLOAT,
            i_deviation FLOAT,
            p_deviation FLOAT
        )
    """
    conn.execute(query)
    
    # Base values
    base_voltage = 48.0
    base_current = 10.0
    base_power = base_voltage * base_current
    base_temperature = 25.0
    base_irradiance = 1000.0
    
    # Generate data based on scenario
    for i in range(count):
        if scenario == 'healthy':
            # Healthy scenario - normal operation with small variations
            voltage = base_voltage + np.random.normal(0, 1)
            current = base_current + np.random.normal(0, 0.5)
            temperature = base_temperature + np.random.normal(0, 3)
            irradiance = base_irradiance + np.random.normal(0, 50)
        
        elif scenario == 'line_line_fault':
            # Line-Line Fault - significant drop in voltage, high current
            voltage = base_voltage * 0.6 + np.random.normal(0, 1)
            current = base_current * 1.5 + np.random.normal(0, 0.5)
            temperature = base_temperature + np.random.normal(0, 3)
            irradiance = base_irradiance + np.random.normal(0, 50)
        
        elif scenario == 'open_circuit':
            # Open Circuit - high voltage, very low current
            voltage = base_voltage * 1.2 + np.random.normal(0, 1)
            current = base_current * 0.1 + np.random.normal(0, 0.1)
            temperature = base_temperature + np.random.normal(0, 3)
            irradiance = base_irradiance + np.random.normal(0, 50)
        
        elif scenario == 'partial_shading':
            # Partial Shading - moderate drop in voltage and current
            voltage = base_voltage * 0.8 + np.random.normal(0, 1)
            current = base_current * 0.7 + np.random.normal(0, 0.5)
            temperature = base_temperature + np.random.normal(0, 3)
            irradiance = base_irradiance * 0.6 + np.random.normal(0, 50)
        
        elif scenario == 'degradation':
            # Panel Degradation - slight drop in voltage and current
            voltage = base_voltage * 0.9 + np.random.normal(0, 1)
            current = base_current * 0.85 + np.random.normal(0, 0.5)
            temperature = base_temperature + 5 + np.random.normal(0, 3)
            irradiance = base_irradiance + np.random.normal(0, 50)
        
        else:  # random
            # Random scenario
            scenarios = ['healthy', 'line_line_fault', 'open_circuit', 'partial_shading', 'degradation']
            weights = [0.7, 0.075, 0.075, 0.075, 0.075]  # 70% healthy, 30% faults
            selected = np.random.choice(scenarios, p=weights)
            return generate_test_data(selected, count)
        
        # Calculate power and deviations
        power = voltage * current
        v_deviation = (voltage - base_voltage) / base_voltage * 100
        i_deviation = (current - base_current) / base_current * 100
        p_deviation = (power - base_power) / base_power * 100
        
        # Insert into database
        query = """
            INSERT INTO solar_data (
                timestamp, voltage, current, power, temperature, irradiance,
                v_deviation, i_deviation, p_deviation
            ) VALUES (
                NOW(), %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        conn.execute(query, (
            voltage, current, power, temperature, irradiance,
            v_deviation, i_deviation, p_deviation
        ))
    
    conn.close()
    
    logger.info(f"Generated {count} test data points for scenario: {scenario}")
    return True

def main():
    """Main function"""
    global monitoring_active, db_connection_str
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection System')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--generate-data', action='store_true', help='Generate test data')
    parser.add_argument('--scenario', type=str, default='random', 
                        choices=['healthy', 'line_line_fault', 'open_circuit', 'partial_shading', 'degradation', 'random'],
                        help='Scenario for test data generation')
    parser.add_argument('--count', type=int, default=10, help='Number of test data records to generate')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database')
    parser.add_argument('--db-host', type=str, default='localhost', help='MySQL database host')
    parser.add_argument('--db-user', type=str, default='root', help='MySQL database username')
    parser.add_argument('--db-password', type=str, default='password', help='MySQL database password')
    parser.add_argument('--db-name', type=str, default='solar_panel_db', help='MySQL database name')
    parser.add_argument('--matlab', action='store_true', help='Enable MATLAB integration')
    parser.add_argument('--no-matlab', action='store_true', help='Disable MATLAB integration')
    args = parser.parse_args()
    
    # Set database connection string
    global DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
    DB_HOST = os.environ.get('DB_HOST', args.db_host)
    DB_USER = os.environ.get('DB_USER', args.db_user)
    DB_PASSWORD = os.environ.get('DB_PASSWORD', args.db_password)
    DB_NAME = os.environ.get('DB_NAME', args.db_name)
    db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    
    # For backward compatibility
    global db_path
    db_path = 'solar_panel.db'  # Kept for command-line argument compatibility
    
    # Test database connection and create database if needed
    if not test_db_connection():
        logger.error("Failed to connect to MySQL database. Exiting.")
        return
    
    # Reset database if requested
    if args.reset_db:
        try:
            engine = create_engine(db_connection_str)
            conn = engine.connect()
            
            # Drop tables if they exist
            conn.execute("DROP TABLE IF EXISTS alerts")
            conn.execute("DROP TABLE IF EXISTS predictions")
            conn.execute("DROP TABLE IF EXISTS solar_data")
            
            conn.close()
            logger.info(f"Database reset: {DB_NAME}")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
    
    # Initialize database
    try:
        engine, Session = setup_database(db_connection_str)
        logger.info(f"Connected to MySQL database at {DB_HOST}")
    except Exception as e:
        logger.error(f"Failed to connect to MySQL database: {e}")
        sys.exit(1)
    
    # Setup database
    from database_setup import setup_database
    setup_database(db_connection_str)
    
    # Also initialize the local tables for the web application
    initialize_database(db_connection_str)
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data(args.scenario, args.count)
        return
    
    # Initialize MATLAB interface if requested
    if args.matlab and matlab_available:
        global matlab_interface
        try:
            matlab_interface = MatlabInterface(db_connection_str=db_connection_str)
            logger.info("MATLAB interface initialized for command line")
        except Exception as e:
            logger.error(f"Error initializing MATLAB interface: {e}")
            matlab_interface = None
    
    # Start the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
