"""
Advanced Real-Time Monitoring System for Solar Fault Detection

This module implements a sophisticated real-time monitoring system that integrates
with the enhanced model to provide comprehensive fault detection and analysis.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
import threading
import queue
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
from enhanced_model import SolarFaultDetectionSystem
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import socketio
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_monitoring.log')
    ]
)
logger = logging.getLogger('advanced_monitoring')

# Path to database
DB_PATH = 'solar_panel.db'

# Socket.IO for real-time updates
sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
CORS(app)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

class DataProcessor:
    """Process data from various sources for fault detection"""
    
    def __init__(self, model_path='solar_fault_detection_model.pth', scaler_path='scaler.pkl'):
        """
        Initialize the data processor
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.load_model()
        
        # Database connection
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self._create_tables()
        
        # Processing queue
        self.queue = queue.Queue()
        self.processing_thread = None
        self.stop_event = threading.Event()
    
    def load_model(self):
        """Load the trained model"""
        try:
            from enhanced_model import SolarFaultDetectionSystem
            self.model = SolarFaultDetectionSystem(
                model_path=self.model_path,
                scaler_path=self.scaler_path
            )
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Model not loaded: {str(e)}")
            return False
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        # Create solar_panel_data table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS solar_panel_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                pv_current FLOAT,
                pv_voltage FLOAT,
                pv_fault_1_current FLOAT,
                pv_fault_1_voltage FLOAT,
                pv_fault_2_current FLOAT,
                pv_fault_2_voltage FLOAT,
                pv_fault_3_current FLOAT,
                pv_fault_3_voltage FLOAT,
                pv_fault_4_current FLOAT,
                pv_fault_4_voltage FLOAT,
                prediction VARCHAR(20),
                prediction_label VARCHAR(20),
                confidence FLOAT,
                processed_at DATETIME,
                description VARCHAR(200),
                recommended_action VARCHAR(200),
                irradiance FLOAT,
                temperature FLOAT,
                pv_power FLOAT,
                grid_power FLOAT,
                efficiency FLOAT,
                is_matlab_data BOOLEAN,
                matlab_simulation_id VARCHAR(50),
                simulation_id VARCHAR(50)
            )
        ''')
        
        # Create alerts table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                data_id INTEGER,
                alert_type VARCHAR(50),
                severity VARCHAR(20),
                message VARCHAR(200),
                acknowledged BOOLEAN DEFAULT 0,
                FOREIGN KEY (data_id) REFERENCES solar_panel_data (id)
            )
        ''')
        
        # Create maintenance_logs table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                data_id INTEGER,
                action_taken VARCHAR(200),
                technician VARCHAR(100),
                notes VARCHAR(500),
                FOREIGN KEY (data_id) REFERENCES solar_panel_data (id)
            )
        ''')
        
        # Create performance_metrics table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_type VARCHAR(50),
                value FLOAT,
                unit VARCHAR(20),
                notes VARCHAR(200)
            )
        ''')
        
        # Commit changes
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def start_processing(self, interval=2, batch_size=5):
        """
        Start the continuous data processing thread
        
        Args:
            interval: Interval in seconds between processing cycles
            batch_size: Number of records to process in each cycle
        """
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Processing thread is already running")
            return False
        
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(interval, batch_size),
            daemon=True
        )
        self.processing_thread.start()
        logger.info(f"Started processing thread with interval={interval}s and batch_size={batch_size}")
        return True
    
    def stop_processing(self):
        """Stop the continuous data processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            logger.warning("Processing thread is not running")
            return False
        
        self.stop_event.set()
        self.processing_thread.join(timeout=10)
        logger.info("Stopped processing thread")
        return True
    
    def _processing_loop(self, interval, batch_size):
        """
        Continuous data processing loop
        
        Args:
            interval: Interval in seconds between processing cycles
            batch_size: Number of records to process in each cycle
        """
        while not self.stop_event.is_set():
            try:
                # Get unprocessed data
                self.cursor.execute('''
                    SELECT 
                        id, pv_current, pv_voltage, 
                        pv_fault_1_current, pv_fault_1_voltage,
                        pv_fault_2_current, pv_fault_2_voltage,
                        pv_fault_3_current, pv_fault_3_voltage,
                        pv_fault_4_current, pv_fault_4_voltage,
                        irradiance, temperature, pv_power, grid_power, efficiency
                    FROM 
                        solar_panel_data 
                    WHERE 
                        processed_at IS NULL 
                    ORDER BY 
                        id ASC 
                    LIMIT ?
                ''', (batch_size,))
                
                rows = self.cursor.fetchall()
                
                if rows:
                    # Process data
                    processed_ids = self._process_batch(rows)
                    
                    # Emit update event
                    if processed_ids:
                        sio.emit('data_processed', {'processed_ids': processed_ids})
                        logger.info(f"Processed {len(processed_ids)} records: {processed_ids}")
                
                # Wait for next cycle
                time.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(interval)
    
    def _process_batch(self, rows):
        """
        Process a batch of data
        
        Args:
            rows: List of database rows to process
            
        Returns:
            List of processed record IDs
        """
        if not rows:
            return []
        
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Create DataFrame from rows
            columns = [
                'id', 'pv_current', 'pv_voltage', 
                'pv_fault_1_current', 'pv_fault_1_voltage',
                'pv_fault_2_current', 'pv_fault_2_voltage',
                'pv_fault_3_current', 'pv_fault_3_voltage',
                'pv_fault_4_current', 'pv_fault_4_voltage',
                'irradiance', 'temperature', 'pv_power', 'grid_power', 'efficiency'
            ]
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # Update database with predictions
            processed_ids = []
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for i, row in df.iterrows():
                record_id = row['id']
                prediction = predictions.iloc[i]['prediction']
                confidence = predictions.iloc[i]['confidence']
                
                # Generate description and recommended action
                description, action = self._generate_description_and_action(
                    prediction, confidence, row
                )
                
                # Update database
                self.cursor.execute('''
                    UPDATE solar_panel_data 
                    SET 
                        prediction = ?,
                        confidence = ?,
                        processed_at = ?,
                        description = ?,
                        recommended_action = ?
                    WHERE 
                        id = ?
                ''', (
                    prediction, confidence, current_time, 
                    description, action, record_id
                ))
                
                processed_ids.append(record_id)
                
                # Generate alert if needed
                if prediction != 'Healthy' and confidence > 70:
                    self._generate_alert(record_id, prediction, confidence, description)
            
            # Commit changes
            self.conn.commit()
            
            return processed_ids
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _generate_description_and_action(self, prediction, confidence, data):
        """
        Generate description and recommended action based on prediction
        
        Args:
            prediction: Fault prediction
            confidence: Prediction confidence
            data: Row data
            
        Returns:
            Tuple of (description, recommended_action)
        """
        if prediction == 'Healthy':
            description = "Solar panel is operating normally."
            action = "No action required."
        
        elif prediction == 'Fault_1':
            description = "Line-Line fault detected. Short circuit between conductors."
            action = "Inspect wiring for damaged insulation. Check for water ingress or physical damage."
        
        elif prediction == 'Fault_2':
            description = "Open circuit detected. Current flow is interrupted."
            action = "Check connections, fuses, and circuit breakers. Inspect for broken conductors."
        
        elif prediction == 'Fault_3':
            description = "Partial shading detected. Reduced current with normal voltage."
            action = "Check for physical obstructions. Clean panels if dirty. Reposition if necessary."
        
        elif prediction == 'Fault_4':
            description = "Panel degradation detected. Gradual reduction in both current and voltage."
            action = "Schedule maintenance. Consider panel replacement if efficiency is significantly reduced."
        
        else:
            description = f"Unknown fault condition detected with {confidence:.2f}% confidence."
            action = "Perform general inspection of the system."
        
        return description, action
    
    def _generate_alert(self, data_id, fault_type, confidence, description):
        """
        Generate an alert for a fault
        
        Args:
            data_id: ID of the data record
            fault_type: Type of fault
            confidence: Prediction confidence
            description: Fault description
        """
        # Determine severity
        if confidence > 90:
            severity = 'High'
        elif confidence > 80:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Create alert message
        message = f"{fault_type} detected with {confidence:.2f}% confidence. {description}"
        
        # Insert alert into database
        self.cursor.execute('''
            INSERT INTO alerts (
                timestamp, data_id, alert_type, severity, message, acknowledged
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data_id, fault_type, severity, message, 0
        ))
        
        self.conn.commit()
        
        # Emit alert event
        alert_id = self.cursor.lastrowid
        sio.emit('new_alert', {
            'id': alert_id,
            'data_id': data_id,
            'fault_type': fault_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        logger.info(f"Generated alert: {message}")
    
    def process_matlab_file(self, file_path):
        """
        Process a MATLAB data file
        
        Args:
            file_path: Path to the MATLAB data file
            
        Returns:
            List of inserted record IDs
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Generate simulation ID
            simulation_id = os.path.basename(file_path).split('.')[0]
            
            # Insert data into database
            record_ids = []
            
            for _, row in df.iterrows():
                # Get current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Extract data
                pv_current = row.get('pv_current', 0)
                pv_voltage = row.get('pv_voltage', 0)
                irradiance = row.get('irradiance', 0)
                temperature = row.get('temperature', 0)
                pv_power = row.get('pv_power', pv_current * pv_voltage)
                grid_power = row.get('grid_power', 0)
                efficiency = row.get('efficiency', 0)
                
                # Generate synthetic fault data
                # Fault 1: Line-Line Fault (Lower voltage, higher current)
                pv_fault_1_voltage = pv_voltage * 0.7
                pv_fault_1_current = pv_current * 1.3
                
                # Fault 2: Open Circuit (Higher voltage, near-zero current)
                pv_fault_2_voltage = pv_voltage * 1.17
                pv_fault_2_current = pv_current * 0.05
                
                # Fault 3: Partial Shading (Slightly lower voltage, moderately lower current)
                pv_fault_3_voltage = pv_voltage * 0.95
                pv_fault_3_current = pv_current * 0.95
                
                # Fault 4: Degradation (Higher voltage, negative current)
                pv_fault_4_voltage = pv_voltage * 1.1
                pv_fault_4_current = -abs(pv_current) * 1.2
                
                # Insert into database
                self.cursor.execute('''
                    INSERT INTO solar_panel_data (
                        timestamp, pv_current, pv_voltage, 
                        pv_fault_1_current, pv_fault_1_voltage,
                        pv_fault_2_current, pv_fault_2_voltage,
                        pv_fault_3_current, pv_fault_3_voltage,
                        pv_fault_4_current, pv_fault_4_voltage,
                        irradiance, temperature, pv_power, grid_power, efficiency,
                        is_matlab_data, matlab_simulation_id, simulation_id,
                        processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ''', (
                    timestamp, pv_current, pv_voltage, 
                    pv_fault_1_current, pv_fault_1_voltage,
                    pv_fault_2_current, pv_fault_2_voltage,
                    pv_fault_3_current, pv_fault_3_voltage,
                    pv_fault_4_current, pv_fault_4_voltage,
                    irradiance, temperature, pv_power, grid_power, efficiency,
                    1, simulation_id, simulation_id
                ))
                
                record_ids.append(self.cursor.lastrowid)
            
            # Commit changes
            self.conn.commit()
            
            logger.info(f"Processed MATLAB file {file_path} with {len(record_ids)} records")
            
            return record_ids
        
        except Exception as e:
            logger.error(f"Error processing MATLAB file {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

class MatlabFileHandler(FileSystemEventHandler):
    """Handle MATLAB data file events"""
    
    def __init__(self, processor, file_pattern='*.csv'):
        """
        Initialize the handler
        
        Args:
            processor: DataProcessor instance
            file_pattern: Pattern to match files
        """
        self.processor = processor
        self.file_pattern = file_pattern
    
    def on_created(self, event):
        """
        Handle file creation event
        
        Args:
            event: File event
        """
        if not event.is_directory and event.src_path.endswith(self.file_pattern[1:]):
            logger.info(f"New MATLAB file detected: {event.src_path}")
            self.processor.process_matlab_file(event.src_path)

class MatlabWatcher:
    """Watch for new MATLAB data files"""
    
    def __init__(self, processor, watch_directory, file_pattern='*.csv'):
        """
        Initialize the watcher
        
        Args:
            processor: DataProcessor instance
            watch_directory: Directory to watch
            file_pattern: Pattern to match files
        """
        self.processor = processor
        self.watch_directory = watch_directory
        self.file_pattern = file_pattern
        self.observer = None
    
    def start(self):
        """Start watching for new files"""
        if self.observer is not None:
            logger.warning("Observer is already running")
            return False
        
        # Create watch directory if it doesn't exist
        if not os.path.exists(self.watch_directory):
            os.makedirs(self.watch_directory)
            logger.info(f"Created watch directory: {self.watch_directory}")
        
        # Set up observer
        self.observer = Observer()
        handler = MatlabFileHandler(self.processor, self.file_pattern)
        self.observer.schedule(handler, self.watch_directory, recursive=False)
        self.observer.start()
        
        logger.info(f"Started watching {self.watch_directory} for {self.file_pattern} files")
        return True
    
    def stop(self):
        """Stop watching for new files"""
        if self.observer is None:
            logger.warning("Observer is not running")
            return False
        
        self.observer.stop()
        self.observer.join()
        self.observer = None
        
        logger.info("Stopped watching for new files")
        return True

# API routes
@app.route('/api/start', methods=['POST'])
def start_processing():
    """Start the continuous data processing"""
    data = request.json
    interval = data.get('interval', 2)
    batch_size = data.get('batch_size', 5)
    
    result = processor.start_processing(interval, batch_size)
    
    return jsonify({
        'success': result,
        'message': 'Processing started' if result else 'Failed to start processing'
    })

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    """Stop the continuous data processing"""
    result = processor.stop_processing()
    
    return jsonify({
        'success': result,
        'message': 'Processing stopped' if result else 'Failed to stop processing'
    })

@app.route('/api/matlab/setup_watch', methods=['POST'])
def setup_matlab_watch():
    """Set up the MATLAB data watch"""
    data = request.json
    watch_directory = data.get('watch_directory')
    file_pattern = data.get('file_pattern', '*.csv')
    
    if not watch_directory:
        return jsonify({
            'success': False,
            'message': 'Watch directory is required'
        })
    
    global matlab_watcher
    matlab_watcher = MatlabWatcher(processor, watch_directory, file_pattern)
    result = matlab_watcher.start()
    
    return jsonify({
        'success': result,
        'message': 'MATLAB watch started' if result else 'Failed to start MATLAB watch'
    })

@app.route('/api/matlab/stop_watch', methods=['POST'])
def stop_matlab_watch():
    """Stop the MATLAB data watch"""
    global matlab_watcher
    
    if matlab_watcher is None:
        return jsonify({
            'success': False,
            'message': 'MATLAB watch is not running'
        })
    
    result = matlab_watcher.stop()
    
    return jsonify({
        'success': result,
        'message': 'MATLAB watch stopped' if result else 'Failed to stop MATLAB watch'
    })

@app.route('/api/data/latest', methods=['GET'])
def get_latest_data():
    """Get the latest processed data"""
    limit = request.args.get('limit', 10, type=int)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            id, timestamp, pv_current, pv_voltage, prediction, confidence, description, recommended_action
        FROM 
            solar_panel_data 
        WHERE 
            processed_at IS NOT NULL 
        ORDER BY 
            id DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    data = []
    for row in rows:
        data.append({
            'id': row[0],
            'timestamp': row[1],
            'pv_current': row[2],
            'pv_voltage': row[3],
            'prediction': row[4],
            'confidence': row[5],
            'description': row[6],
            'recommended_action': row[7]
        })
    
    return jsonify(data)

@app.route('/api/alerts/latest', methods=['GET'])
def get_latest_alerts():
    """Get the latest alerts"""
    limit = request.args.get('limit', 10, type=int)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            id, timestamp, data_id, alert_type, severity, message, acknowledged
        FROM 
            alerts 
        ORDER BY 
            id DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append({
            'id': row[0],
            'timestamp': row[1],
            'data_id': row[2],
            'alert_type': row[3],
            'severity': row[4],
            'message': row[5],
            'acknowledged': bool(row[6])
        })
    
    return jsonify(alerts)

@app.route('/api/alerts/acknowledge', methods=['POST'])
def acknowledge_alert():
    """Acknowledge an alert"""
    data = request.json
    alert_id = data.get('alert_id')
    
    if not alert_id:
        return jsonify({
            'success': False,
            'message': 'Alert ID is required'
        })
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE alerts 
        SET acknowledged = 1 
        WHERE id = ?
    ''', (alert_id,))
    
    conn.commit()
    conn.close()
    
    # Emit event
    sio.emit('alert_acknowledged', {'alert_id': alert_id})
    
    return jsonify({
        'success': True,
        'message': f'Alert {alert_id} acknowledged'
    })

@app.route('/api/stats/summary', methods=['GET'])
def get_stats_summary():
    """Get summary statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get prediction counts
    cursor.execute('''
        SELECT 
            prediction, COUNT(*) as count 
        FROM 
            solar_panel_data 
        WHERE 
            processed_at IS NOT NULL 
        GROUP BY 
            prediction
    ''')
    
    prediction_counts = {}
    for row in cursor.fetchall():
        prediction_counts[row[0]] = row[1]
    
    # Get alert counts by severity
    cursor.execute('''
        SELECT 
            severity, COUNT(*) as count 
        FROM 
            alerts 
        GROUP BY 
            severity
    ''')
    
    alert_counts = {}
    for row in cursor.fetchall():
        alert_counts[row[0]] = row[1]
    
    # Get average confidence by prediction
    cursor.execute('''
        SELECT 
            prediction, AVG(confidence) as avg_confidence 
        FROM 
            solar_panel_data 
        WHERE 
            processed_at IS NOT NULL 
        GROUP BY 
            prediction
    ''')
    
    avg_confidence = {}
    for row in cursor.fetchall():
        avg_confidence[row[0]] = row[1]
    
    conn.close()
    
    return jsonify({
        'prediction_counts': prediction_counts,
        'alert_counts': alert_counts,
        'avg_confidence': avg_confidence
    })

@app.route('/api/maintenance/log', methods=['POST'])
def log_maintenance():
    """Log a maintenance action"""
    data = request.json
    data_id = data.get('data_id')
    action_taken = data.get('action_taken')
    technician = data.get('technician')
    notes = data.get('notes', '')
    
    if not data_id or not action_taken or not technician:
        return jsonify({
            'success': False,
            'message': 'Data ID, action taken, and technician are required'
        })
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO maintenance_logs (
            timestamp, data_id, action_taken, technician, notes
        ) VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data_id, action_taken, technician, notes
    ))
    
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    
    # Emit event
    sio.emit('maintenance_logged', {
        'id': log_id,
        'data_id': data_id,
        'action_taken': action_taken,
        'technician': technician,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return jsonify({
        'success': True,
        'message': 'Maintenance logged successfully',
        'id': log_id
    })

# Socket.IO events
@sio.event
def connect(sid, environ):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")

# Global variables
processor = None
matlab_watcher = None

def main():
    """Main function to start the monitoring system"""
    global processor, matlab_watcher
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Start processing
    processor.start_processing(interval=2, batch_size=5)
    
    # Start MATLAB watcher
    matlab_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
    matlab_watcher = MatlabWatcher(processor, matlab_data_dir)
    matlab_watcher.start()
    
    # Start Flask app
    logger.info("Starting Flask app on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == "__main__":
    main()
