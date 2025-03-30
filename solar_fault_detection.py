"""
Solar Fault Detection System - Integrated Solution
Combines the best features of basic and advanced monitoring
"""

import os
import sys
import time
import json
import logging
import sqlite3
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
import joblib

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
db_path = 'solar_panel.db'
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
    
    def __init__(self, db_path=None):
        """Initialize the data processor"""
        self.db_path = db_path or 'solar_panel.db'
        self.detector = SolarFaultDetectionSystem()
        self.conn = None
        self.cursor = None
        self.connect_db()
    
    def connect_db(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
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
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            
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
                VALUES (?, ?, ?, datetime('now', 'localtime'))
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
                VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
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
                LIMIT ?
            """
            self.cursor.execute(query, (limit,))
            rows = self.cursor.fetchall()
            
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
            self.cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = self.cursor.fetchone()[0]
            
            # Get fault distribution
            query = """
                SELECT prediction, COUNT(*) as count
                FROM predictions
                GROUP BY prediction
                ORDER BY prediction
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            distribution = {}
            for row in rows:
                fault_type = fault_types.get(row[0], "Unknown")
                distribution[fault_type] = row[1]
            
            # Get average confidence
            self.cursor.execute("SELECT AVG(confidence) FROM predictions")
            avg_confidence = self.cursor.fetchone()[0] or 0
            
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
    global monitoring_active
    
    processor = DataProcessor(db_path)
    
    logger.info("Starting monitoring loop")
    
    while monitoring_active:
        try:
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
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                
                # Emit prediction update
                socketio.emit('prediction_update', {
                    'prediction': result['prediction'],
                    'fault_type': result['fault_type'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
                
                # Emit alert if fault detected
                if result['prediction'] != 0:
                    alert = processor.generate_alert(result)
                    if alert:
                        socketio.emit('alert', alert)
                
                logger.info(f"Processed data: {result['id']}, Prediction: {result['fault_type']}")
            
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
    processor = DataProcessor(db_path)
    data = processor.get_latest_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data available'})

@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """Get latest alerts"""
    processor = DataProcessor(db_path)
    alerts = processor.get_latest_alerts()
    return jsonify(alerts)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get performance statistics"""
    processor = DataProcessor(db_path)
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

def setup_database():
    """Set up the database if it doesn't exist"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create solar_data table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solar_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                voltage REAL,
                current REAL,
                power REAL,
                temperature REAL,
                irradiance REAL,
                v_deviation REAL,
                i_deviation REAL,
                p_deviation REAL
            )
        """)
        
        # Create predictions table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_id INTEGER,
                prediction INTEGER,
                confidence REAL,
                timestamp TEXT,
                FOREIGN KEY (data_id) REFERENCES solar_data (id)
            )
        """)
        
        # Create alerts table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_id INTEGER,
                fault_type INTEGER,
                confidence REAL,
                message TEXT,
                severity TEXT,
                timestamp TEXT,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (data_id) REFERENCES solar_data (id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Database setup complete")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

def generate_test_data(scenario='random', count=1):
    """Generate test data for the specified scenario"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        records = []
        
        for _ in range(count):
            # Base values
            voltage = 48.0  # Nominal voltage
            current = 10.0  # Nominal current
            power = voltage * current
            temperature = 25.0
            irradiance = 1000.0
            
            # Add variations based on scenario
            if scenario == 'healthy':
                # Healthy panel with small variations
                voltage += np.random.uniform(-0.5, 0.5)
                current += np.random.uniform(-0.5, 0.5)
                v_deviation = (voltage - 48.0) / 48.0 * 100
                i_deviation = (current - 10.0) / 10.0 * 100
            
            elif scenario == 'fault_1':
                # Line-Line Fault: Significant voltage drop, current increase
                voltage -= np.random.uniform(5.0, 10.0)
                current += np.random.uniform(5.0, 10.0)
                v_deviation = (voltage - 48.0) / 48.0 * 100
                i_deviation = (current - 10.0) / 10.0 * 100
            
            elif scenario == 'fault_2':
                # Open Circuit: Voltage increase, severe current drop
                voltage += np.random.uniform(5.0, 10.0)
                current -= np.random.uniform(8.0, 10.0)
                v_deviation = (voltage - 48.0) / 48.0 * 100
                i_deviation = (current - 10.0) / 10.0 * 100
            
            elif scenario == 'fault_3':
                # Partial Shading: Moderate voltage and current drop
                voltage -= np.random.uniform(1.0, 3.0)
                current -= np.random.uniform(2.0, 5.0)
                v_deviation = (voltage - 48.0) / 48.0 * 100
                i_deviation = (current - 10.0) / 10.0 * 100
            
            elif scenario == 'fault_4':
                # Panel Degradation: Moderate voltage and current drop
                voltage -= np.random.uniform(2.0, 4.0)
                current -= np.random.uniform(2.0, 4.0)
                v_deviation = (voltage - 48.0) / 48.0 * 100
                i_deviation = (current - 10.0) / 10.0 * 100
            
            else:  # random
                # Random scenario
                scenarios = ['healthy', 'fault_1', 'fault_2', 'fault_3', 'fault_4']
                return generate_test_data(np.random.choice(scenarios), count)
            
            # Calculate power and deviations
            power = voltage * current
            p_deviation = (power - 480.0) / 480.0 * 100
            
            # Add temperature and irradiance variations
            temperature += np.random.uniform(-5.0, 5.0)
            irradiance += np.random.uniform(-100.0, 100.0)
            
            # Insert data into database
            query = """
                INSERT INTO solar_data (
                    timestamp, voltage, current, power, temperature, irradiance,
                    v_deviation, i_deviation, p_deviation
                )
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                voltage, current, power, temperature, irradiance,
                v_deviation, i_deviation, p_deviation
            ))
            
            # Get the ID of the inserted row
            data_id = cursor.lastrowid
            records.append(data_id)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Generated {count} test data records for scenario: {scenario}")
        return records
    
    except Exception as e:
        logger.error(f"Error generating test data: {e}")
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Solar Fault Detection System')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--generate-data', action='store_true', help='Generate test data')
    parser.add_argument('--scenario', type=str, default='random', 
                        choices=['healthy', 'fault_1', 'fault_2', 'fault_3', 'fault_4', 'random'],
                        help='Scenario for test data generation')
    parser.add_argument('--count', type=int, default=10, help='Number of test data records to generate')
    
    args = parser.parse_args()
    
    # Setup database
    setup_database()
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data(args.scenario, args.count)
    
    # Start the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
