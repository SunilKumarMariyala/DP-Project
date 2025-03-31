#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MATLAB Continuous Data Flow Demo

This script demonstrates the continuous data flow from MATLAB to MySQL and then to the Python prediction model.
It sets up a directory watcher to monitor for new MATLAB output files, processes them, and makes predictions.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import our modules
from matlab_interface import MatlabInterface
from database_setup import setup_database, SolarPanelData
from solar_fault_detection import SolarFaultDetectionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("matlab_continuous_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MatlabContinuousDemo")

class MatlabFileHandler(FileSystemEventHandler):
    """
    Handler for MATLAB output files that are created in the watch directory
    """
    def __init__(self, matlab_interface, db_connection_str):
        """
        Initialize the handler
        
        Args:
            matlab_interface: MatlabInterface instance
            db_connection_str: Database connection string
        """
        self.matlab_interface = matlab_interface
        self.db_connection_str = db_connection_str
        self.processed_files = set()
        
        # Initialize the prediction model
        self.detector = SolarFaultDetectionSystem()
    
    def on_created(self, event):
        """
        Handle file creation events
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        # Check if it's a CSV or MAT file
        if not (event.src_path.lower().endswith('.csv') or event.src_path.lower().endswith('.mat')):
            return
        
        # Avoid processing the same file multiple times
        if event.src_path in self.processed_files:
            return
        
        logger.info(f"New MATLAB data file detected: {event.src_path}")
        
        # Wait a moment to ensure the file is completely written
        time.sleep(1)
        
        # Process the file
        try:
            # Process the MATLAB data file and save to database
            result = self.matlab_interface.process_real_time_matlab_data(event.src_path)
            
            if result and result.get('status') == 'success':
                logger.info(f"Successfully processed {result.get('records_added')} records from {event.src_path}")
                
                # Make predictions on the newly added data
                simulation_id = result.get('simulation_id')
                if simulation_id:
                    # Get the session
                    engine, Session = setup_database(self.db_connection_str)
                    session = Session()
                    
                    try:
                        # Get the latest data for this simulation
                        latest_data = session.query(SolarPanelData).filter(
                            SolarPanelData.simulation_id == simulation_id
                        ).all()
                        
                        logger.info(f"Making predictions for {len(latest_data)} data points from simulation {simulation_id}")
                        
                        # Make predictions for each data point
                        for data_point in latest_data:
                            # Create input for prediction model
                            input_data = {
                                'pv_current': data_point.pv_current,
                                'pv_voltage': data_point.pv_voltage,
                                'irradiance': data_point.irradiance,
                                'temperature': data_point.temperature,
                                'pv_fault_1_current': data_point.pv_fault_1_current,
                                'pv_fault_1_voltage': data_point.pv_fault_1_voltage,
                                'pv_fault_2_current': data_point.pv_fault_2_current,
                                'pv_fault_2_voltage': data_point.pv_fault_2_voltage,
                                'pv_fault_3_current': data_point.pv_fault_3_current,
                                'pv_fault_3_voltage': data_point.pv_fault_3_voltage,
                                'pv_fault_4_current': data_point.pv_fault_4_current,
                                'pv_fault_4_voltage': data_point.pv_fault_4_voltage
                            }
                            
                            # Make prediction
                            prediction_result = self.detector.predict(input_data)
                            
                            if prediction_result:
                                # Log prediction
                                logger.info(f"Prediction for data point {data_point.id}: {prediction_result['fault_type']} (Confidence: {prediction_result['confidence']:.2f}%)")
                                
                                # Update database with prediction
                                data_point.prediction = prediction_result.get('fault_type', 'unknown')
                                data_point.prediction_label = prediction_result.get('fault_type', 'unknown')
                                data_point.confidence = prediction_result.get('confidence', 0.0)
                                data_point.processed_at = datetime.now()
                                
                                # Add description and recommended action based on fault type
                                if prediction_result['prediction'] == 0:
                                    data_point.description = "Solar panel is operating normally."
                                    data_point.recommended_action = "No action required."
                                elif prediction_result['prediction'] == 1:
                                    data_point.description = "Line-Line fault detected. Two points in the array are connected."
                                    data_point.recommended_action = "Inspect wiring between panels for short circuits."
                                elif prediction_result['prediction'] == 2:
                                    data_point.description = "Open circuit fault detected. Circuit is broken somewhere in the array."
                                    data_point.recommended_action = "Check for disconnected cables or failed components."
                                elif prediction_result['prediction'] == 3:
                                    data_point.description = "Partial shading detected. Some panels are receiving less sunlight."
                                    data_point.recommended_action = "Check for objects casting shadows or dirt on panels."
                                elif prediction_result['prediction'] == 4:
                                    data_point.description = "Panel degradation detected. Efficiency is lower than expected."
                                    data_point.recommended_action = "Schedule maintenance to inspect panel condition."
                            else:
                                logger.warning(f"No prediction result for data point {data_point.id}")
                        
                        # Commit changes
                        session.commit()
                        logger.info(f"Updated predictions for {len(latest_data)} data points")
                        
                    except Exception as e:
                        logger.error(f"Error making predictions: {e}")
                        session.rollback()
                    finally:
                        session.close()
            else:
                logger.error(f"Failed to process file: {result.get('message') if result else 'Unknown error'}")
            
            # Mark file as processed
            self.processed_files.add(event.src_path)
            
            # If the set gets too large, clear older entries
            if len(self.processed_files) > 100:
                # Keep only the 50 most recent files
                self.processed_files = set(list(self.processed_files)[-50:])
            
        except Exception as e:
            logger.error(f"Error processing file {event.src_path}: {e}")

def main():
    """
    Main function to run the continuous data flow demo
    """
    parser = argparse.ArgumentParser(description='MATLAB Continuous Data Flow Demo')
    parser.add_argument('--watch-dir', type=str, default=os.environ.get('MATLAB_OUTPUT_DIR', './matlab_output'), 
                        help='Directory to watch for MATLAB output files')
    parser.add_argument('--db-host', type=str, default=os.environ.get('DB_HOST', 'localhost'), help='MySQL host')
    parser.add_argument('--db-user', type=str, default=os.environ.get('DB_USER', 'solar_user'), help='MySQL user')
    parser.add_argument('--db-password', type=str, default=os.environ.get('DB_PASSWORD', 'your_secure_password'), help='MySQL password')
    parser.add_argument('--db-name', type=str, default=os.environ.get('DB_NAME', 'solar_panel_db'), help='MySQL database name')
    args = parser.parse_args()
    
    # Create the watch directory if it doesn't exist
    if not os.path.exists(args.watch_dir):
        os.makedirs(args.watch_dir)
        logger.info(f"Created watch directory: {args.watch_dir}")
    
    # Setup database connection
    db_connection_str = f'mysql+pymysql://{args.db_user}:{args.db_password}@{args.db_host}/{args.db_name}'
    try:
        engine, Session = setup_database(db_connection_str)
        logger.info(f"Connected to MySQL database at {args.db_host}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Initialize MATLAB interface
    matlab_interface = MatlabInterface(db_connection_str=db_connection_str)
    
    # Create an observer to watch for new files
    event_handler = MatlabFileHandler(matlab_interface, db_connection_str)
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=False)
    observer.start()
    
    logger.info(f"Started watching directory: {args.watch_dir}")
    logger.info("Waiting for MATLAB output files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping file watcher...")
    
    observer.join()
    logger.info("File watcher stopped. Exiting.")

if __name__ == "__main__":
    main()
