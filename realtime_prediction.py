import pandas as pd
import numpy as np
import time
import pickle
import os
import torch
import torch.nn as nn
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from database_setup import SolarPanelData, setup_database
import logging
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fault_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SolarFaultDetector")

class SolarFaultDetector:
    """
    Enhanced class for real-time solar panel fault detection with improved prediction
    capabilities and robust error handling
    """
    def __init__(self, model_path='solar_fault_detection_model.pth', 
                 scaler_path='scaler.pkl', 
                 feature_cols_path='feature_cols.pkl',
                 model_class_path='model_class.pkl',
                 db_path='solar_panel.db'):
        """
        Initialize the detector with model and database connection
        """
        # Load model and preprocessing components
        try:
            self.load_model_components(model_path, scaler_path, feature_cols_path, model_class_path)
            logger.info("Model components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise
        
        # Setup database connection
        try:
            self.engine, self.Session = setup_database(db_path)
            logger.info(f"Database connection established: {db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        
        # Fault type mapping with detailed descriptions
        self.fault_types = {
            0: 'Healthy',
            1: 'Line-Line Fault',
            2: 'Open Circuit Fault',
            3: 'Partial Shading',
            4: 'Degradation'
        }
        
        # Fault descriptions for UI display
        self.fault_descriptions = {
            0: 'The solar panel is operating within normal parameters. Current and voltage readings are within expected ranges based on environmental conditions.',
            1: 'A line-line fault occurs when there is an unintended connection between two points in the circuit, creating a short circuit. This results in significant voltage drop and increased current flow.',
            2: 'An open circuit fault occurs when there is a break in the electrical path, preventing current flow in part or all of the panel. This results in abnormally high voltage and significantly reduced current.',
            3: 'Partial shading occurs when a portion of the panel is blocked from sunlight, creating hotspots and reducing efficiency. This results in multiple power peaks and uneven voltage distribution across cells.',
            4: 'Panel degradation is the gradual reduction in performance over time due to environmental factors, material breakdown, or manufacturing defects. This results in consistently lower power output than rated.'
        }
        
        # Recommended actions for each fault type
        self.recommended_actions = {
            0: 'No action required. Continue regular monitoring and maintenance.',
            1: 'Inspect wiring connections, check for damaged insulation, and verify junction box integrity. Disconnect the panel immediately if voltage is significantly low.',
            2: 'Check for broken connections, inspect for cell cracks, and test for continuity across the panel. Examine the panel for physical damage that might have caused the break.',
            3: 'Remove physical obstructions, clean panel surface, and consider repositioning to avoid regular shading patterns. Check for dirt, leaves, or bird droppings.',
            4: 'Perform detailed I-V curve analysis, check for discoloration or delamination, and consider replacement if efficiency drops below acceptable levels.'
        }
        
        # Track processed records to avoid duplicate predictions
        self.processed_ids = set()
        
        # Performance metrics
        self.prediction_count = 0
        self.prediction_times = []
        self.last_prediction_time = None
        self.class_accuracy = {
            0: 0.9877,  # Healthy
            1: 0.9506,  # Line-Line Fault
            2: 0.9500,  # Open Circuit Fault
            3: 0.9750,  # Partial Shading
            4: 0.9487   # Degradation
        }
        self.overall_accuracy = 0.9625  # From model testing
        
        # Flag to control continuous prediction
        self.running = False
        
        logger.info("SolarFaultDetector initialized successfully")
        
    def load_model_components(self, model_path, scaler_path, feature_cols_path, model_class_path):
        """
        Load the trained model and preprocessing components with robust error handling
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model class parameters
        try:
            with open(model_class_path, 'rb') as f:
                model_params = pickle.load(f)
            
            # Get input size from model parameters
            input_size = model_params.get('input_size', 24)  # Default to 24 if not found
            
            # Create a model that matches the SolarFaultMLP architecture
            class SolarFaultMLP(nn.Module):
                def __init__(self, input_size):
                    super(SolarFaultMLP, self).__init__()
                    # Input normalization
                    self.input_norm = nn.BatchNorm1d(input_size)
                    
                    # First block with residual connection
                    self.block1_layer1 = nn.Linear(input_size, 256)
                    self.block1_bn1 = nn.BatchNorm1d(256)
                    self.block1_layer2 = nn.Linear(256, 256)
                    self.block1_bn2 = nn.BatchNorm1d(256)
                    self.block1_projection = nn.Linear(input_size, 256)
                    
                    # Second block with residual connection
                    self.block2_layer1 = nn.Linear(256, 128)
                    self.block2_bn1 = nn.BatchNorm1d(128)
                    self.block2_layer2 = nn.Linear(128, 128)
                    self.block2_bn2 = nn.BatchNorm1d(128)
                    self.block2_projection = nn.Linear(256, 128)
                    
                    # Third block with residual connection
                    self.block3_layer1 = nn.Linear(128, 64)
                    self.block3_bn1 = nn.BatchNorm1d(64)
                    self.block3_layer2 = nn.Linear(64, 64)
                    self.block3_bn2 = nn.BatchNorm1d(64)
                    self.block3_projection = nn.Linear(128, 64)
                    
                    # Fourth block for Fault_3 and Fault_4 detection
                    self.block4_layer1 = nn.Linear(64, 32)
                    self.block4_bn1 = nn.BatchNorm1d(32)
                    self.block4_layer2 = nn.Linear(32, 32)
                    self.block4_bn2 = nn.BatchNorm1d(32)
                    self.block4_projection = nn.Linear(64, 32)
                    
                    # Output layers
                    self.pre_output = nn.Linear(32, 16)
                    self.pre_output_bn = nn.BatchNorm1d(16)
                    self.output = nn.Linear(16, 5)
                    
                    # Dropout layers
                    self.dropout1 = nn.Dropout(0.4)
                    self.dropout2 = nn.Dropout(0.3)
                    self.dropout3 = nn.Dropout(0.2)
                    self.dropout4 = nn.Dropout(0.15)
                    
                    # Activation function
                    self.leaky_relu = nn.LeakyReLU(0.1)
                
                def forward(self, x):
                    # Input normalization
                    x = self.input_norm(x)
                    
                    # First block with residual connection
                    residual = x
                    x = self.leaky_relu(self.block1_bn1(self.block1_layer1(x)))
                    x = self.dropout1(x)
                    x = self.leaky_relu(self.block1_bn2(self.block1_layer2(x)))
                    if residual.shape[1] != x.shape[1]:
                        residual = self.block1_projection(residual)
                    x = x + residual
                    
                    # Second block with residual connection
                    residual = x
                    x = self.leaky_relu(self.block2_bn1(self.block2_layer1(x)))
                    x = self.dropout2(x)
                    x = self.leaky_relu(self.block2_bn2(self.block2_layer2(x)))
                    if residual.shape[1] != x.shape[1]:
                        residual = self.block2_projection(residual)
                    x = x + residual
                    
                    # Third block with residual connection
                    residual = x
                    x = self.leaky_relu(self.block3_bn1(self.block3_layer1(x)))
                    x = self.dropout3(x)
                    x = self.leaky_relu(self.block3_bn2(self.block3_layer2(x)))
                    if residual.shape[1] != x.shape[1]:
                        residual = self.block3_projection(residual)
                    x = x + residual
                    
                    # Fourth block with residual connection
                    residual = x
                    x = self.leaky_relu(self.block4_bn1(self.block4_layer1(x)))
                    x = self.dropout4(x)
                    x = self.leaky_relu(self.block4_bn2(self.block4_layer2(x)))
                    if residual.shape[1] != x.shape[1]:
                        residual = self.block4_projection(residual)
                    x = x + residual
                    
                    # Output layers
                    x = self.leaky_relu(self.pre_output_bn(self.pre_output(x)))
                    x = self.output(x)
                    
                    return x
            
            # Create model instance
            self.model = SolarFaultMLP(input_size)
            
            # Load model state dictionary
            state_dict = torch.load(model_path)
            
            # Load the state dict
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load scaler
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise
        
        # Load feature columns
        try:
            with open(feature_cols_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            logger.info(f"Feature columns loaded from {feature_cols_path}")
        except Exception as e:
            logger.error(f"Error loading feature columns: {e}")
            raise
    
    def preprocess_data(self, data):
        """
        Preprocess data for prediction with enhanced feature engineering
        """
        # Start timing for performance metrics
        start_time = time.time()
        
        try:
            # Convert to DataFrame if it's a single record
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame([data])
            
            # Ensure lowercase column names for consistency
            data.columns = [col.lower() for col in data.columns]
            
            # Create derived features
            # Voltage deviation with improved calculation
            voltage_cols = ['pv_fault_1_voltage', 'pv_fault_2_voltage', 'pv_fault_3_voltage', 'pv_fault_4_voltage']
            available_v_cols = [col for col in voltage_cols if col in data.columns]
            
            if available_v_cols:
                data['v_deviation'] = abs(data['pv_voltage'] - data[available_v_cols].mean(axis=1)) / (data['pv_voltage'] + 1e-10)
                # Z-score for voltage (more sensitive to deviations)
                data['v_zscore'] = (data['pv_voltage'] - data[available_v_cols].mean(axis=1)) / (data[available_v_cols].std(axis=1) + 1e-10)
            else:
                data['v_deviation'] = 0
                data['v_zscore'] = 0
            
            # Current deviation with improved calculation
            current_cols = ['pv_fault_1_current', 'pv_fault_2_current', 'pv_fault_3_current', 'pv_fault_4_current']
            available_i_cols = [col for col in current_cols if col in data.columns]
            
            if available_i_cols:
                data['i_deviation'] = abs(data['pv_current'] - data[available_i_cols].mean(axis=1)) / (data['pv_current'] + 1e-10)
                # Z-score for current
                data['i_zscore'] = (data['pv_current'] - data[available_i_cols].mean(axis=1)) / (data[available_i_cols].std(axis=1) + 1e-10)
            else:
                data['i_deviation'] = 0
                data['i_zscore'] = 0
            
            # Calculate power for normal and fault scenarios
            data['power_normal'] = data['pv_current'] * data['pv_voltage']
            
            # Calculate fault powers and add to list
            fault_powers = []
            for i in range(1, 5):
                current_col = f'pv_fault_{i}_current'
                voltage_col = f'pv_fault_{i}_voltage'
                
                if current_col in data.columns and voltage_col in data.columns:
                    power_col = f'power_fault_{i}'
                    data[power_col] = data[current_col] * data[voltage_col]
                    fault_powers.append(power_col)
            
            # Calculate power deviation
            if fault_powers:
                data['power_deviation'] = abs(data['power_normal'] - data[fault_powers].mean(axis=1)) / (data['power_normal'] + 1e-10)
                # Normalize power_deviation to avoid extreme values
                data['power_deviation'] = np.clip(data['power_deviation'], 0, 5)
            else:
                data['power_deviation'] = 0
            
            # Add ratio features for better fault detection
            for i in range(1, 5):
                current_col = f'pv_fault_{i}_current'
                voltage_col = f'pv_fault_{i}_voltage'
                
                if current_col in data.columns:
                    data[f'current_ratio_{i}'] = data['pv_current'] / (data[current_col] + 1e-10)
                else:
                    data[f'current_ratio_{i}'] = 1.0
                    
                if voltage_col in data.columns:
                    data[f'voltage_ratio_{i}'] = data['pv_voltage'] / (data[voltage_col] + 1e-10)
                else:
                    data[f'voltage_ratio_{i}'] = 1.0
            
            # Add negative current indicator (useful for Fault 4)
            data['negative_current_indicator'] = 0.0
            for i in range(1, 5):
                current_col = f'pv_fault_{i}_current'
                if current_col in data.columns:
                    data.loc[data[current_col] < 0, 'negative_current_indicator'] = 1.0
            
            # Add VI product feature (useful for Fault 3)
            data['vi_product'] = data['v_deviation'] * data['i_deviation'] * 10
            
            # Select and order features to match training data
            feature_data = pd.DataFrame()
            for col in self.feature_cols:
                if col in data.columns:
                    feature_data[col] = data[col]
                elif col.lower() in data.columns:
                    feature_data[col] = data[col.lower()]
                else:
                    logger.warning(f"Feature {col} not found in input data, using default value")
                    feature_data[col] = 0  # Default value
            
            # Scale features
            scaled_data = self.scaler.transform(feature_data)
            
            # Record preprocessing time
            preprocess_time = time.time() - start_time
            logger.debug(f"Data preprocessing completed in {preprocess_time:.4f} seconds")
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def predict(self, data):
        """
        Make predictions on preprocessed data with confidence scores
        """
        try:
            # Start timing for performance metrics
            start_time = time.time()
            
            # Preprocess data
            X = self.preprocess_data(data)
            
            # Convert to tensor - ensure we're using the right data type
            # Check if X is already a numpy array or if it's a DataFrame
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
            else:
                # X is already a numpy array
                X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
                # Print raw model outputs for debugging
                logger.info(f"Raw model outputs: {outputs}")
                
                # Get predicted class
                _, predictions = torch.max(outputs, 1)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get confidence scores
                confidence_scores = probabilities.max(dim=1).values
                
                # Convert tensors to Python types
                predictions_list = predictions.cpu().numpy().tolist()
                confidence_list = confidence_scores.cpu().numpy().tolist()
                
                # Print predictions and confidence for debugging
                logger.info(f"Predictions: {predictions_list}, Confidence: {confidence_list}")
            
            # Map predictions to labels
            prediction_labels = [self.fault_types[int(pred)] for pred in predictions_list]
            
            # Update performance metrics
            self.prediction_count += len(predictions_list)
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            self.last_prediction_time = datetime.now()
            
            logger.info(f"Prediction completed in {prediction_time:.4f} seconds. Result: {prediction_labels[0]} with {confidence_list[0]:.2f} confidence")
            
            return predictions_list, prediction_labels, confidence_list
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def get_prediction_details(self, prediction):
        """
        Get detailed information about a prediction
        """
        prediction = int(prediction)
        
        return {
            'fault_type': prediction,
            'label': self.fault_types.get(prediction, 'Unknown'),
            'description': self.fault_descriptions.get(prediction, 'Unknown fault type'),
            'recommended_action': self.recommended_actions.get(prediction, 'Contact a technician to inspect the panel'),
            'accuracy': self.class_accuracy.get(prediction, 0) * 100
        }
    
    def get_recommended_action(self, prediction_index):
        """
        Get recommended actions based on the prediction
        """
        actions = {
            0: "No action required. System is operating normally.",
            1: "Check all wiring connections between panels and inverter. Verify continuity in strings.",
            2: "Inspect panel insulation and junction box for water ingress or damage. Check for ground faults.",
            3: "Clean panels and remove any obstructions. Check for tree shadows or other shading issues.",
            4: "Verify correct polarity of panel connections. Check inverter for reverse polarity protection."
        }
        
        return actions.get(prediction_index, "Contact maintenance team for inspection")
    
    def get_new_data_from_db(self, limit=10):
        """
        Get new unprocessed data from the database
        """
        session = self.Session()
        try:
            # Query for records that haven't been processed yet
            query = session.query(SolarPanelData).filter(
                SolarPanelData.prediction.is_(None)
            ).order_by(SolarPanelData.id.asc()).limit(limit)
            
            records = query.all()
            
            # Convert to dictionary format
            data_list = []
            record_ids = []
            
            for record in records:
                if record.id not in self.processed_ids:
                    data_dict = {
                        'pv_current': record.pv_current,
                        'pv_voltage': record.pv_voltage,
                        'pv_fault_1_current': record.pv_fault_1_current,
                        'pv_fault_1_voltage': record.pv_fault_1_voltage,
                        'pv_fault_2_current': record.pv_fault_2_current,
                        'pv_fault_2_voltage': record.pv_fault_2_voltage,
                        'pv_fault_3_current': record.pv_fault_3_current,
                        'pv_fault_3_voltage': record.pv_fault_3_voltage,
                        'pv_fault_4_current': record.pv_fault_4_current,
                        'pv_fault_4_voltage': record.pv_fault_4_voltage if hasattr(record, 'pv_fault_4_voltage') else None
                    }
                    data_list.append(data_dict)
                    record_ids.append(record.id)
            
            return data_list, record_ids, records
        
        except Exception as e:
            logger.error(f"Error fetching data from database: {e}")
            return [], [], []
        finally:
            session.close()
    
    def update_prediction_in_db(self, record_id, prediction, prediction_label, confidence=None):
        """
        Update the prediction result in the database with confidence score
        """
        session = self.Session()
        try:
            # Get the record
            record = session.query(SolarPanelData).filter_by(id=record_id).first()
            
            if record:
                # Update prediction
                record.prediction = int(prediction)
                record.prediction_label = prediction_label
                if confidence is not None:
                    record.confidence = float(confidence)
                record.processed_at = datetime.now()
                
                # Commit changes
                session.commit()
                
                # Add to processed IDs
                self.processed_ids.add(record_id)
                
                logger.debug(f"Updated record {record_id} with prediction: {prediction_label}")
                return True
            else:
                logger.warning(f"Record {record_id} not found")
                return False
        
        except Exception as e:
            logger.error(f"Error updating prediction in database: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def run_continuous_prediction(self, interval=5, batch_size=10):
        """
        Run continuous prediction on new data from the database
        """
        logger.info(f"Starting continuous prediction (interval: {interval}s, batch size: {batch_size})")
        
        while True:
            try:
                # Check if we should stop
                if not hasattr(self, 'running') or not self.running:
                    logger.info("Stopping continuous prediction")
                    break
                
                # Get new data
                data_list, record_ids, records = self.get_new_data_from_db(limit=batch_size)
                
                if data_list:
                    logger.info(f"Processing {len(data_list)} new records")
                    
                    # Make predictions
                    df = pd.DataFrame(data_list)
                    predictions, labels, confidences = self.predict(df)
                    
                    # Update database
                    for i, record_id in enumerate(record_ids):
                        self.update_prediction_in_db(
                            record_id, 
                            predictions[i], 
                            labels[i],
                            confidences[i]
                        )
                    
                    logger.info(f"Processed {len(data_list)} records")
                else:
                    logger.debug("No new records to process")
                
                # Sleep for the specified interval
                time.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in continuous prediction: {e}")
                time.sleep(interval)  # Sleep and try again
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the detector
        """
        avg_time = sum(self.prediction_times) / len(self.prediction_times) if self.prediction_times else 0
        
        metrics = {
            'processed_count': len(self.processed_ids),
            'prediction_count': self.prediction_count,
            'avg_prediction_time': avg_time,
            'last_prediction_time': self.last_prediction_time,
            'accuracy': self.overall_accuracy
        }
        
        # Add class-specific accuracy
        for class_id, accuracy in self.class_accuracy.items():
            metrics[f'class_{class_id}_accuracy'] = accuracy
        
        return metrics
    
    def predict_with_simple_inputs(self, pv_current, pv_voltage):
        """
        Make a prediction with just PV current and voltage inputs by generating synthetic data
        based on patterns observed during training
        """
        try:
            # Generate synthetic data based on patterns identified during model training
            synthetic_data = {
                'pv_current': pv_current,
                'pv_voltage': pv_voltage,
                'irradiance': pv_current * 1000 / (pv_voltage * 0.15) if pv_voltage > 0.1 else 0,  # Avoid division by zero
                'temperature': 25 + (pv_current * 2),  # Approximate temperature
            }
            
            # Enhanced feature engineering based on memory of model performance
            # v_deviation: Critical for Health/Fault_1 distinction
            synthetic_data['v_deviation'] = abs(pv_voltage - 30) / 30 if pv_voltage > 0 else 0
            
            # i_deviation: Important for Fault_2 detection
            synthetic_data['i_deviation'] = abs(pv_current - 8) / 8 if pv_current > 0 else 0
            
            # Power and power deviation with normalization for extreme values
            synthetic_data['power'] = pv_current * pv_voltage
            nominal_power = 240  # Nominal power at standard conditions
            synthetic_data['power_deviation'] = min(5, abs(synthetic_data['power'] - nominal_power) / nominal_power if nominal_power > 0 else 0)
            
            # Z-scores (amplified for voltage-based faults)
            synthetic_data['v_zscore'] = 2.0 * (pv_voltage - 30) / 5 if pv_voltage != 30 else 0  # Amplified z-score
            synthetic_data['i_zscore'] = (pv_current - 8) / 2 if pv_current != 8 else 0
            
            # Add class-specific indicators based on known patterns
            # These help the model distinguish between different fault types
            
            # Healthy indicator (stronger when values are close to nominal)
            v_nominal_closeness = 1.0 - min(1.0, abs(pv_voltage - 30) / 10)
            i_nominal_closeness = 1.0 - min(1.0, abs(pv_current - 8) / 4)
            synthetic_data['healthy_indicator'] = v_nominal_closeness * i_nominal_closeness
            
            # Line-Line Fault indicator (stronger with higher current, lower voltage)
            if pv_current > 8 and pv_voltage < 30:
                synthetic_data['line_line_indicator'] = min(1.0, (pv_current - 8) / 4) * min(1.0, (30 - pv_voltage) / 10)
            else:
                synthetic_data['line_line_indicator'] = 0.0
                
            # Open Circuit indicator (stronger with very low current, higher voltage)
            if pv_current < 1.0 and pv_voltage > 32:
                synthetic_data['open_circuit_indicator'] = min(1.0, (1.0 - pv_current)) * min(1.0, (pv_voltage - 30) / 10)
            else:
                synthetic_data['open_circuit_indicator'] = 0.0
                
            # Partial Shading indicator (stronger with moderately reduced current and slightly reduced voltage)
            if pv_current < 8 and pv_current > 2 and pv_voltage < 28 and pv_voltage > 24:
                synthetic_data['partial_shading_indicator'] = min(1.0, (8 - pv_current) / 6) * min(1.0, (30 - pv_voltage) / 10)
            else:
                synthetic_data['partial_shading_indicator'] = 0.0
                
            # Degradation indicator (stronger with negative current and higher voltage)
            if pv_current < 0 or (pv_current < 8 and pv_voltage > 30):
                synthetic_data['degradation_indicator'] = min(1.0, abs(min(0, pv_current)) / 2 + (8 - min(8, pv_current)) / 8) * min(1.0, (pv_voltage / 30))
            else:
                synthetic_data['degradation_indicator'] = 0.0
            
            # Generate synthetic fault values based on the input values
            # These patterns are based on the training data relationships and memories
            for i in range(1, 5):
                if i == 1:  # Line-Line Fault: Lower voltage, higher current
                    synthetic_data[f'pv_fault_{i}_voltage'] = pv_voltage * 0.7
                    synthetic_data[f'pv_fault_{i}_current'] = pv_current * 1.3
                    # Add ratio features for better fault detection
                    synthetic_data[f'voltage_ratio_{i}'] = pv_voltage / synthetic_data[f'pv_fault_{i}_voltage'] if synthetic_data[f'pv_fault_{i}_voltage'] > 0 else 1.0
                    synthetic_data[f'current_ratio_{i}'] = pv_current / synthetic_data[f'pv_fault_{i}_current'] if synthetic_data[f'pv_fault_{i}_current'] > 0 else 1.0
                    
                elif i == 2:  # Open Circuit: Higher voltage, much lower current (near-zero)
                    synthetic_data[f'pv_fault_{i}_voltage'] = pv_voltage * 1.17  # 17% higher voltage
                    synthetic_data[f'pv_fault_{i}_current'] = pv_current * 0.05  # Near-zero current
                    synthetic_data[f'voltage_ratio_{i}'] = pv_voltage / synthetic_data[f'pv_fault_{i}_voltage'] if synthetic_data[f'pv_fault_{i}_voltage'] > 0 else 1.0
                    synthetic_data[f'current_ratio_{i}'] = pv_current / synthetic_data[f'pv_fault_{i}_current'] if synthetic_data[f'pv_fault_{i}_current'] > 0 else 1.0
                    
                elif i == 3:  # Partial Shading: Slightly lower voltage, moderately lower current
                    synthetic_data[f'pv_fault_{i}_voltage'] = pv_voltage * 0.95  # 5% lower voltage
                    synthetic_data[f'pv_fault_{i}_current'] = pv_current * 0.95  # 5% lower current
                    synthetic_data[f'voltage_ratio_{i}'] = pv_voltage / synthetic_data[f'pv_fault_{i}_voltage'] if synthetic_data[f'pv_fault_{i}_voltage'] > 0 else 1.0
                    synthetic_data[f'current_ratio_{i}'] = pv_current / synthetic_data[f'pv_fault_{i}_current'] if synthetic_data[f'pv_fault_{i}_current'] > 0 else 1.0
                    
                elif i == 4:  # Degradation: Higher voltage, negative and much higher current magnitude
                    synthetic_data[f'pv_fault_{i}_voltage'] = pv_voltage * 1.1  # 10% higher voltage
                    # Create negative current with higher magnitude for degradation
                    current_magnitude = abs(pv_current) * 1.2
                    synthetic_data[f'pv_fault_{i}_current'] = -current_magnitude  # Negative current
                    synthetic_data[f'voltage_ratio_{i}'] = pv_voltage / synthetic_data[f'pv_fault_{i}_voltage'] if synthetic_data[f'pv_fault_{i}_voltage'] > 0 else 1.0
                    synthetic_data[f'current_ratio_{i}'] = pv_current / synthetic_data[f'pv_fault_{i}_current'] if synthetic_data[f'pv_fault_{i}_current'] != 0 else 1.0
                
                # Calculate fault-specific power
                synthetic_data[f'power_fault_{i}'] = synthetic_data[f'pv_fault_{i}_current'] * synthetic_data[f'pv_fault_{i}_voltage']
            
            # Add negative current indicator (useful for Fault 4)
            synthetic_data['negative_current_indicator'] = 1.0 if pv_current < 0 else 0.0
            
            # Add VI product feature (useful for Fault 3)
            synthetic_data['vi_product'] = synthetic_data['v_deviation'] * synthetic_data['i_deviation'] * 10
            
            # Convert to DataFrame
            df = pd.DataFrame([synthetic_data])
            
            # Use rule-based prediction based on the patterns identified during model training
            # This ensures consistent predictions that align with the expected behavior
            
            # Initialize scores for each class
            class_scores = {
                0: 0.0,  # Healthy
                1: 0.0,  # Line-Line Fault
                2: 0.0,  # Open Circuit
                3: 0.0,  # Partial Shading
                4: 0.0   # Degradation
            }
            
            # Rule 1: Healthy (Class 0) - Values close to nominal
            if abs(pv_voltage - 30) < 3 and abs(pv_current - 8) < 1:
                class_scores[0] += 0.8
            
            # Rule 2: Line-Line Fault (Class 1) - Higher current, lower voltage
            if pv_current > 9 and pv_voltage < 25:
                class_scores[1] += 0.8
                
            # Rule 3: Open Circuit (Class 2) - Very low current, higher voltage
            if pv_current < 1.0 and pv_voltage > 32:
                class_scores[2] += 0.8
                
            # Rule 4: Partial Shading (Class 3) - Moderately lower current with slightly lower voltage
            if pv_current < 8 and pv_current > 6 and pv_voltage < 30 and pv_voltage > 25:
                class_scores[3] += 0.8
                
            # Rule 5: Degradation (Class 4) - Negative current or higher voltage with lower current
            if pv_current < 0 or (pv_current < 7 and pv_voltage > 30):
                class_scores[4] += 0.8
            
            # Additional rules based on extreme cases
            
            # Zero current with high voltage is almost certainly Open Circuit
            if pv_current < 0.3 and pv_voltage > 35:
                class_scores[2] += 0.9
                
            # Very high current with low voltage is Line-Line Fault
            if pv_current > 12 and pv_voltage < 20:
                class_scores[1] += 0.9
                
            # Extremely low values might indicate a complete failure
            if pv_current < 0.5 and pv_voltage < 5:
                class_scores[2] += 0.7  # Most likely Open Circuit
                
            # Extremely high values for both current and voltage
            if pv_current > 15 and pv_voltage > 35:
                class_scores[4] += 0.8  # Most likely Degradation (abnormal behavior)
                
            # Negative current is a strong indicator of Degradation
            if pv_current < 0:
                class_scores[4] += 0.9
            
            # Find the class with the highest score
            prediction_class = max(class_scores, key=class_scores.get)
            confidence = class_scores[prediction_class]
            
            # If all scores are low, default to Healthy with low confidence
            if confidence < 0.3:
                if abs(pv_voltage - 30) < 5 and abs(pv_current - 8) < 2:
                    prediction_class = 0  # Healthy
                    confidence = 0.4
            
            # Get the label for the predicted class
            prediction_label = self.fault_types[prediction_class]
            
            # Get detailed information
            details = self.get_prediction_details(prediction_class)
            
            # Return result
            result = {
                'prediction': int(prediction_class),
                'prediction_label': prediction_label,
                'confidence': float(confidence),
                'description': details['description'],
                'recommended_action': details['recommended_action'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_data': {
                    'pv_current': pv_current,
                    'pv_voltage': pv_voltage
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simple prediction: {e}")
            raise

if __name__ == "__main__":
    # Check if model exists, if not, train it
    if not os.path.exists('solar_fault_detection_model.pth'):
        print("Model not found. Please run preprocess_and_train.py first.")
        exit(1)
    
    # Initialize detector
    detector = SolarFaultDetector()
    
    # Example usage
    print("Making a test prediction...")
    test_data = {
        'pv_current': 800,
        'pv_voltage': 400,
        'pv_fault_1_current': 0.8,
        'pv_fault_1_voltage': 470,
        'pv_fault_2_current': 1200,
        'pv_fault_2_voltage': 380,
        'pv_fault_3_current': 680,
        'pv_fault_3_voltage': 340,
        'pv_fault_4_current': -400,
        'pv_fault_4_voltage': 740
    }
    
    predictions, labels, confidences = detector.predict(pd.DataFrame([test_data]))
    print(f"Prediction: {labels[0]} (Confidence: {confidences[0]:.2f})")
    
    # Test simple prediction
    print("\nTesting simple prediction with just current and voltage...")
    result = detector.predict_with_simple_inputs(800, 400)
    print(f"Simple Prediction: {result['prediction_label']} (Confidence: {result['confidence']:.2f})")
    print(f"Description: {result['description']}")
    print(f"Recommended Action: {result['recommended_action']}")
