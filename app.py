from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
from torch import nn
import threading
import time
from datetime import datetime
import logging
from database_setup import SolarPanelData, setup_database, get_latest_data as db_get_latest_data
from realtime_prediction import SolarFaultDetector
import traceback
from matlab_interface import MatlabInterface
from matlab_integration import MatlabIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SolarFaultApp")

app = Flask(__name__)

# Initialize the fault detector
detector = None

# Background thread for continuous prediction
prediction_thread = None
running = False

# Initialize MATLAB interface if available
try:
    matlab_interface = MatlabInterface()
    matlab_integration = MatlabIntegration()
    matlab_available = matlab_interface.matlab_available and matlab_interface.eng is not None
    if matlab_available:
        logger.info("MATLAB integration is available")
    else:
        logger.warning("MATLAB integration is not available")
except Exception as e:
    logger.error(f"Error initializing MATLAB integration: {e}")
    matlab_interface = None
    matlab_integration = None
    matlab_available = False

# Initialize the detector at startup
def init_app():
    """
    Initialize the application and detector
    """
    global detector
    
    try:
        # Check if model exists
        model_path = 'solar_fault_detection_model.pth'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Initialize detector
        detector = SolarFaultDetector()
        logger.info("Solar Fault Detector initialized successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}\n{traceback.format_exc()}")
        return False

# Initialize the detector at startup
init_app()

@app.route('/')
def home():
    """
    Render the home page
    """
    # Redirect to advanced dashboard for testing
    return redirect('/advanced_dashboard')
    
    # Original code (commented out for now)
    '''
    # Get latest data
    try:
        # Use the database function directly to get latest data
        latest_data = db_get_latest_data(10)
    except Exception as e:
        logger.error(f"Error getting latest data for home page: {str(e)}")
        latest_data = []  # Provide empty list as fallback
    
    # Get MATLAB availability status
    matlab_status = "Available" if matlab_available else "Not Available"
    
    return render_template('index.html', 
                           latest_data=latest_data, 
                           matlab_status=matlab_status,
                           matlab_available=matlab_available)
    '''

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on input data
    """
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['pv_current', 'pv_voltage']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction, labels, confidence = detector.predict(df)
        
        # Get detailed information
        details = detector.get_prediction_details(prediction[0])
        
        # Return result
        result = {
            'prediction': int(prediction[0]),
            'label': labels[0],
            'confidence': float(confidence[0]),
            'description': details['description'],
            'recommended_action': details['recommended_action'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log the prediction
        logger.info(f"Prediction: {labels[0]} with confidence {confidence[0]:.2f}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest_data', methods=['GET'])
def get_latest_data():
    """
    Get the latest data from the database with enhanced information
    """
    try:
        # Get limit parameter (default 10)
        limit = request.args.get('limit', default=10, type=int)
        
        # Get latest records
        session = detector.Session()
        records = session.query(SolarPanelData).order_by(SolarPanelData.id.desc()).limit(limit).all()
        
        # Convert to list of dictionaries
        result = []
        for record in records:
            result.append({
                'id': record.id,
                'timestamp': record.timestamp.isoformat() if record.timestamp else None,
                'pv_current': record.pv_current,
                'pv_voltage': record.pv_voltage,
                'prediction': record.prediction,
                'prediction_label': record.prediction_label,
                'confidence': record.confidence if hasattr(record, 'confidence') else 0,
                'processed_at': record.processed_at.isoformat() if hasattr(record, 'processed_at') and record.processed_at else None
            })
        
        session.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching latest data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_prediction():
    """
    Start the continuous prediction thread
    """
    global prediction_thread, running
    
    try:
        if running:
            return jsonify({'success': False, 'message': 'Prediction already running', 'status': 'running'})
        
        # Get interval and batch size
        data = request.json or {}
        interval = data.get('interval', 5)
        batch_size = data.get('batch_size', 10)
        
        # Validate parameters
        if interval < 1:
            return jsonify({'success': False, 'error': 'Interval must be at least 1 second'}), 400
        if batch_size < 1:
            return jsonify({'success': False, 'error': 'Batch size must be at least 1'}), 400
        
        # Start prediction thread
        running = True
        detector.running = True
        prediction_thread = threading.Thread(
            target=detector.run_continuous_prediction,
            kwargs={'interval': interval, 'batch_size': batch_size}
        )
        prediction_thread.daemon = True
        prediction_thread.start()
        
        logger.info(f"Started continuous prediction (interval: {interval}s, batch size: {batch_size})")
        
        return jsonify({
            'success': True,
            'message': f'Prediction started (interval: {interval}s, batch size: {batch_size})',
            'status': 'running'
        })
    
    except Exception as e:
        logger.error(f"Error starting prediction: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_prediction():
    """
    Stop the continuous prediction thread
    """
    global running
    
    try:
        if not running:
            return jsonify({'success': False, 'message': 'Prediction not running', 'status': 'stopped'})
        
        # Stop prediction thread
        running = False
        detector.running = False
        
        logger.info("Stopped continuous prediction")
        
        return jsonify({'success': True, 'message': 'Prediction stopped', 'status': 'stopped'})
    
    except Exception as e:
        logger.error(f"Error stopping prediction: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get the status of the prediction thread and system metrics
    """
    try:
        # Get performance metrics
        metrics = detector.get_performance_metrics() if detector else {}
        
        # Get panel statistics
        session = detector.Session()
        total_count = session.query(SolarPanelData).count()
        healthy_count = session.query(SolarPanelData).filter(SolarPanelData.prediction == 0).count()
        fault_count = total_count - healthy_count
        
        # Calculate average response time
        avg_response_time = metrics.get('avg_prediction_time', 0) * 1000  # Convert to ms
        
        status = {
            'running': running,
            'processed_count': metrics.get('processed_count', 0),
            'accuracy': metrics.get('accuracy', 0),
            'total_panels': total_count,
            'healthy_panels': healthy_count,
            'fault_panels': fault_count,
            'response_time': avg_response_time
        }
        
        session.close()
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get comprehensive statistics about the predictions
    """
    try:
        session = detector.Session()
        
        # Get total count
        total_count = session.query(SolarPanelData).count()
        processed_count = session.query(SolarPanelData).filter(SolarPanelData.prediction != None).count()
        
        # Get counts by prediction type
        type_counts = {}
        for i in range(5):  # Assuming 5 classes (0-4)
            count = session.query(SolarPanelData).filter(SolarPanelData.prediction == i).count()
            if count > 0:
                type_counts[str(i)] = count
        
        # Get performance metrics
        metrics = detector.get_performance_metrics() if detector else {}
        
        # Calculate class-specific accuracy
        class_accuracy = {}
        for i in range(5):
            class_accuracy[str(i)] = metrics.get(f'class_{i}_accuracy', 0)
        
        # Get time-based statistics (last 24 hours)
        current_time = datetime.now()
        last_24h = current_time - pd.Timedelta(days=1)
        hourly_counts = []
        
        for hour in range(24):
            hour_start = last_24h + pd.Timedelta(hours=hour)
            hour_end = last_24h + pd.Timedelta(hours=hour+1)
            
            count = session.query(SolarPanelData).filter(
                SolarPanelData.timestamp >= hour_start,
                SolarPanelData.timestamp < hour_end
            ).count()
            
            hourly_counts.append({
                'hour': hour_start.strftime('%H:%M'),
                'count': count
            })
        
        session.close()
        
        return jsonify({
            'total_count': total_count,
            'processed_count': processed_count,
            'unprocessed_count': total_count - processed_count,
            'type_counts': type_counts,
            'class_accuracy': class_accuracy,
            'hourly_counts': hourly_counts,
            'avg_response_time': metrics.get('avg_prediction_time', 0) * 1000,  # Convert to ms
            'overall_accuracy': metrics.get('accuracy', 0)
        })
    
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simple_predict', methods=['POST'])
def simple_predict():
    """
    Make a prediction based on just PV current and voltage
    """
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['pv_current', 'pv_voltage']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract values
        pv_current = float(data['pv_current'])
        pv_voltage = float(data['pv_voltage'])
        
        # Use the detector to make a prediction with simple inputs
        result = detector.predict_with_simple_inputs(pv_current, pv_voltage)
        
        # Log the prediction
        logger.info(f"Simple prediction: {result['prediction_label']} with confidence {result['confidence']:.2f}")
        
        return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in simple prediction: {error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/fault_types', methods=['GET'])
def get_fault_types():
    """
    Get the list of fault types and their descriptions
    """
    try:
        result = {
            'fault_types': detector.fault_types,
            'descriptions': detector.fault_descriptions
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting fault types: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """
    Get information about the model
    """
    try:
        # Get model information
        model_path = 'solar_fault_detection_model.pth'
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        model_modified = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(model_path) else None
        
        # Get feature information
        feature_cols = detector.feature_cols if detector else []
        
        result = {
            'model_file': model_path,
            'model_size_bytes': model_size,
            'model_last_modified': model_modified,
            'feature_count': len(feature_cols),
            'top_features': feature_cols[:10] if feature_cols else []
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve images from the images directory
    """
    return send_from_directory('', filename)

@app.route('/advanced_dashboard')
def advanced_dashboard():
    """
    Render the advanced dashboard page with real-time monitoring
    """
    return render_template('dashboard.html')

@app.route('/api/advanced/start', methods=['POST'])
def start_advanced_monitoring():
    """
    Start the advanced monitoring system
    """
    try:
        from advanced_monitoring import DataProcessor
        
        # Get parameters from request
        data = request.json or {}
        interval = data.get('interval', 2)
        batch_size = data.get('batch_size', 5)
        
        # Initialize processor if not already done
        global advanced_processor
        if 'advanced_processor' not in globals() or advanced_processor is None:
            advanced_processor = DataProcessor()
        
        # Start processing
        result = advanced_processor.start_processing(interval, batch_size)
        
        return jsonify({
            'success': result,
            'message': 'Advanced monitoring started' if result else 'Failed to start advanced monitoring'
        })
    
    except Exception as e:
        logger.error(f"Error starting advanced monitoring: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/advanced/stop', methods=['POST'])
def stop_advanced_monitoring():
    """
    Stop the advanced monitoring system
    """
    try:
        global advanced_processor
        if 'advanced_processor' not in globals() or advanced_processor is None:
            return jsonify({
                'success': False,
                'message': 'Advanced monitoring not running'
            })
        
        # Stop processing
        result = advanced_processor.stop_processing()
        
        return jsonify({
            'success': result,
            'message': 'Advanced monitoring stopped' if result else 'Failed to stop advanced monitoring'
        })
    
    except Exception as e:
        logger.error(f"Error stopping advanced monitoring: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# MATLAB Integration Routes

@app.route('/matlab_dashboard')
def matlab_dashboard():
    """
    Dashboard for MATLAB integration
    """
    # Get MATLAB simulation data
    simulation_data = get_matlab_simulation_data(10) if matlab_available else []
    
    return render_template('matlab_dashboard.html', 
                           simulation_data=simulation_data,
                           matlab_available=matlab_available)

@app.route('/api/matlab/run_simulation', methods=['POST'])
def run_matlab_simulation():
    """
    API endpoint for running a MATLAB simulation
    """
    if not matlab_available:
        return jsonify({'error': 'MATLAB integration is not available'}), 400
    
    try:
        # Get parameters from request
        data = request.json
        irradiance = float(data.get('irradiance', 1000))
        temperature = float(data.get('temperature', 25))
        
        # Run simulation and prediction
        result = matlab_integration.run_simulation_and_predict(irradiance, temperature)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Simulation failed'}), 500
    
    except Exception as e:
        logger.error(f"Error in MATLAB simulation API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/matlab/get_simulations')
def get_matlab_simulations():
    """
    API endpoint for getting MATLAB simulation data
    """
    if not matlab_available:
        return jsonify({'error': 'MATLAB integration is not available'}), 400
    
    try:
        # Get simulation data
        simulation_data = get_matlab_simulation_data(10)
        
        # Convert to list of dicts
        data_list = []
        for record in simulation_data:
            data_dict = {
                'id': record.id,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'pv_current': record.pv_current,
                'pv_voltage': record.pv_voltage,
                'pv_power': record.pv_power,
                'grid_power': record.grid_power,
                'efficiency': record.efficiency,
                'irradiance': record.irradiance,
                'temperature': record.temperature,
                'prediction': record.prediction,
                'prediction_label': record.prediction_label,
                'confidence': record.confidence
            }
            data_list.append(data_dict)
        
        return jsonify(data_list)
    
    except Exception as e:
        logger.error(f"Error in MATLAB simulations API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/matlab/generate_dataset', methods=['POST'])
def generate_matlab_dataset():
    """
    API endpoint for generating a MATLAB simulation dataset
    """
    if not matlab_available:
        return jsonify({'error': 'MATLAB integration is not available'}), 400
    
    try:
        # Get parameters from request
        data = request.json
        num_samples = int(data.get('num_samples', 10))
        
        # Generate dataset
        df = matlab_integration.generate_fault_simulation_dataset(num_samples=num_samples)
        
        if df is not None:
            # Return basic statistics
            stats = {
                'total_samples': len(df),
                'fault_distribution': df['fault_label'].value_counts().to_dict(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return jsonify(stats)
        else:
            return jsonify({'error': 'Dataset generation failed'}), 500
    
    except Exception as e:
        logger.error(f"Error in MATLAB dataset generation API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/matlab/process_realtime_data', methods=['POST'])
def process_realtime_matlab_data():
    """
    API endpoint for processing real-time data from MATLAB
    """
    try:
        # Check if MATLAB is available
        if not matlab_available:
            return jsonify({'error': 'MATLAB integration is not available'}), 503
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'Empty file provided'}), 400
        
        # Check file extension
        allowed_extensions = {'.csv', '.mat'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'}), 400
        
        # Save file to temporary location
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
        
        # Create directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Process the file
        result = matlab_interface.process_real_time_matlab_data(file_path)
        
        if result is None:
            return jsonify({'error': 'Failed to process MATLAB data'}), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing real-time MATLAB data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/matlab/setup_watch', methods=['POST'])
def setup_matlab_watch():
    """
    API endpoint for setting up a watch directory for MATLAB data
    """
    try:
        # Check if MATLAB is available
        if not matlab_available:
            return jsonify({'error': 'MATLAB integration is not available'}), 503
        
        # Get parameters from request
        data = request.json or {}
        watch_dir = data.get('watch_directory')
        file_pattern = data.get('file_pattern', '*.csv')
        check_interval = data.get('check_interval', 5)
        
        # If no watch directory provided, use default
        if not watch_dir:
            watch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
        
        # Setup watch
        matlab_interface.setup_matlab_data_watch(watch_dir, file_pattern, check_interval)
        
        return jsonify({
            'success': True,
            'watch_directory': watch_dir,
            'file_pattern': file_pattern,
            'check_interval': check_interval
        })
    
    except Exception as e:
        logger.error(f"Error setting up MATLAB watch: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Initialize advanced monitoring if available
    try:
        from advanced_monitoring import DataProcessor
        global advanced_processor
        advanced_processor = DataProcessor()
        logger.info("Advanced monitoring system initialized")
    except Exception as e:
        logger.warning(f"Advanced monitoring system not available: {e}")
        advanced_processor = None
    
    # Start the Flask app
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)
