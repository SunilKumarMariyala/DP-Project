import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from matlab_interface import MatlabInterface
from realtime_prediction import SolarFaultDetector
from database_setup import setup_database, SolarPanelData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("matlab_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MatlabIntegration")

class MatlabIntegration:
    """
    Integration class to connect MATLAB simulations with the solar fault detection system
    """
    def __init__(self, matlab_path=None, model_path=None, db_path='solar_panel.db'):
        """
        Initialize the integration
        """
        # Initialize MATLAB interface
        try:
            self.matlab_interface = MatlabInterface(matlab_path, model_path, db_path)
            logger.info("MATLAB interface initialized")
        except Exception as e:
            logger.error(f"Error initializing MATLAB interface: {e}")
            self.matlab_interface = None
        
        # Initialize fault detector
        try:
            self.fault_detector = SolarFaultDetector(db_path=db_path)
            logger.info("Fault detector initialized")
        except Exception as e:
            logger.error(f"Error initializing fault detector: {e}")
            self.fault_detector = None
        
        # Setup database connection
        try:
            self.engine, self.Session = setup_database(db_path)
            logger.info(f"Database connection established: {db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def run_simulation_and_predict(self, irradiance=1000, temperature=25):
        """
        Run a MATLAB simulation and make a fault prediction
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            
        Returns:
            Dictionary containing simulation results and fault prediction
        """
        if self.matlab_interface is None or self.fault_detector is None:
            logger.error("MATLAB interface or fault detector not initialized")
            return None
        
        try:
            # Run simulation
            logger.info(f"Running simulation with irradiance={irradiance} W/m², temperature={temperature}°C")
            simulation_result = self.matlab_interface.simulate_fault_conditions(irradiance, temperature)
            
            if simulation_result is None:
                logger.error("Simulation failed")
                return None
            
            # Save to database
            record_id = self.matlab_interface.save_simulation_to_db(simulation_result)
            
            if record_id is None:
                logger.error("Failed to save simulation to database")
                return None
            
            # Get the record from database
            session = self.Session()
            record = session.query(SolarPanelData).filter_by(id=record_id).first()
            
            if record is None:
                logger.error(f"Record {record_id} not found in database")
                session.close()
                return None
            
            # Prepare data for prediction
            data = {
                'pv_current': record.pv_current,
                'pv_voltage': record.pv_voltage,
                'pv_fault_1_current': record.pv_fault_1_current,
                'pv_fault_1_voltage': record.pv_fault_1_voltage,
                'pv_fault_2_current': record.pv_fault_2_current,
                'pv_fault_2_voltage': record.pv_fault_2_voltage,
                'pv_fault_3_current': record.pv_fault_3_current,
                'pv_fault_3_voltage': record.pv_fault_3_voltage,
                'pv_fault_4_current': record.pv_fault_4_current,
                'pv_fault_4_voltage': getattr(record, 'pv_fault_4_voltage', None)
            }
            
            # Make prediction
            logger.info("Making fault prediction")
            predictions, labels, confidences = self.fault_detector.predict(pd.DataFrame([data]))
            
            # Update prediction in database
            self.fault_detector.update_prediction_in_db(
                record_id,
                predictions[0],
                labels[0],
                confidences[0]
            )
            
            # Get prediction details
            prediction_details = self.fault_detector.get_prediction_details(predictions[0])
            
            # Combine results
            result = {
                'simulation': simulation_result,
                'prediction': {
                    'class': int(predictions[0]),
                    'label': labels[0],
                    'confidence': float(confidences[0]),
                    'description': prediction_details['description'],
                    'recommended_action': prediction_details['recommended_action']
                },
                'record_id': record_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Prediction: {labels[0]} with confidence {confidences[0]:.2f}")
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error in simulation and prediction: {e}")
            return None
    
    def run_continuous_simulation_and_prediction(self, interval=60, duration=3600, 
                                                min_irradiance=600, max_irradiance=1200,
                                                min_temperature=15, max_temperature=45):
        """
        Run continuous simulations and predictions for a specified duration
        
        Args:
            interval: Time between simulations in seconds
            duration: Total duration in seconds
            min_irradiance: Minimum irradiance in W/m²
            max_irradiance: Maximum irradiance in W/m²
            min_temperature: Minimum temperature in °C
            max_temperature: Maximum temperature in °C
        """
        if self.matlab_interface is None or self.fault_detector is None:
            logger.error("MATLAB interface or fault detector not initialized")
            return
        
        logger.info(f"Starting continuous simulation and prediction for {duration} seconds")
        logger.info(f"Interval: {interval} seconds")
        logger.info(f"Irradiance range: {min_irradiance}-{max_irradiance} W/m²")
        logger.info(f"Temperature range: {min_temperature}-{max_temperature} °C")
        
        start_time = time.time()
        end_time = start_time + duration
        
        results = []
        
        while time.time() < end_time:
            # Generate random irradiance and temperature
            irradiance = np.random.uniform(min_irradiance, max_irradiance)
            temperature = np.random.uniform(min_temperature, max_temperature)
            
            # Run simulation and prediction
            result = self.run_simulation_and_predict(irradiance, temperature)
            
            if result:
                results.append(result)
                logger.info(f"Completed simulation {len(results)}")
                logger.info(f"Prediction: {result['prediction']['label']} with confidence {result['prediction']['confidence']:.2f}")
            
            # Calculate time to next simulation
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            if remaining <= 0:
                break
            
            # Sleep until next interval
            sleep_time = min(interval, remaining)
            logger.info(f"Sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        logger.info(f"Completed {len(results)} simulations and predictions")
        return results
    
    def generate_fault_simulation_dataset(self, num_samples=50, random_seed=42):
        """
        Generate a dataset of fault simulations for training and testing
        
        Args:
            num_samples: Number of samples per fault type
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame containing the generated dataset
        """
        np.random.seed(random_seed)
        
        if self.matlab_interface is None:
            logger.error("MATLAB interface not initialized")
            return None
        
        logger.info(f"Generating fault simulation dataset with {num_samples} samples per fault type")
        
        # Define fault conditions
        fault_conditions = {
            'healthy': {
                'irradiance_range': (800, 1200),
                'temperature_range': (20, 35)
            },
            'fault_1': {  # Line-Line Fault
                'irradiance_range': (800, 1200),
                'temperature_range': (35, 50)
            },
            'fault_2': {  # Open Circuit
                'irradiance_range': (50, 300),
                'temperature_range': (15, 30)
            },
            'fault_3': {  # Partial Shading
                'irradiance_range': (400, 700),
                'temperature_range': (25, 40)
            },
            'fault_4': {  # Degradation
                'irradiance_range': (600, 900),
                'temperature_range': (30, 45)
            }
        }
        
        all_records = []
        
        # Generate samples for each fault type
        for fault_type, conditions in fault_conditions.items():
            logger.info(f"Generating {num_samples} samples for {fault_type}")
            
            for i in range(num_samples):
                # Generate random irradiance and temperature within the specified range
                irradiance = np.random.uniform(*conditions['irradiance_range'])
                temperature = np.random.uniform(*conditions['temperature_range'])
                
                # Run simulation
                if fault_type == 'healthy':
                    # For healthy samples, run normal simulation
                    simulation_result = self.matlab_interface.run_simulation(irradiance, temperature)
                    
                    if simulation_result:
                        # Create a record with all fault values equal to normal values
                        record = {
                            'pv_current': simulation_result['pv_current'],
                            'pv_voltage': simulation_result['pv_voltage'],
                            'pv_fault_1_current': simulation_result['pv_current'],
                            'pv_fault_1_voltage': simulation_result['pv_voltage'],
                            'pv_fault_2_current': simulation_result['pv_current'],
                            'pv_fault_2_voltage': simulation_result['pv_voltage'],
                            'pv_fault_3_current': simulation_result['pv_current'],
                            'pv_fault_3_voltage': simulation_result['pv_voltage'],
                            'pv_fault_4_current': simulation_result['pv_current'],
                            'pv_fault_4_voltage': simulation_result['pv_voltage'],
                            'fault_type': 0,  # Healthy
                            'fault_label': 'Healthy',
                            'irradiance': irradiance,
                            'temperature': temperature
                        }
                        all_records.append(record)
                else:
                    # For fault samples, run fault simulation
                    simulation_result = self.matlab_interface.simulate_fault_conditions(irradiance, temperature)
                    
                    if simulation_result:
                        # Extract fault index (1-4)
                        fault_index = int(fault_type.split('_')[1])
                        
                        # Create a record
                        record = {
                            'pv_current': simulation_result['normal']['pv_current'],
                            'pv_voltage': simulation_result['normal']['pv_voltage'],
                            'pv_fault_1_current': simulation_result['faults']['fault_1']['pv_current'],
                            'pv_fault_1_voltage': simulation_result['faults']['fault_1']['pv_voltage'],
                            'pv_fault_2_current': simulation_result['faults']['fault_2']['pv_current'],
                            'pv_fault_2_voltage': simulation_result['faults']['fault_2']['pv_voltage'],
                            'pv_fault_3_current': simulation_result['faults']['fault_3']['pv_current'],
                            'pv_fault_3_voltage': simulation_result['faults']['fault_3']['pv_voltage'],
                            'pv_fault_4_current': simulation_result['faults']['fault_4']['pv_current'],
                            'pv_fault_4_voltage': simulation_result['faults']['fault_4']['pv_voltage'],
                            'fault_type': fault_index,
                            'fault_label': simulation_result['faults'][fault_type]['description'],
                            'irradiance': irradiance,
                            'temperature': temperature
                        }
                        all_records.append(record)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} samples for {fault_type}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'matlab_simulation_dataset_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Dataset generated with {len(df)} samples and saved to {csv_path}")
        
        return df

def main():
    """
    Main function to run the MATLAB integration
    """
    parser = argparse.ArgumentParser(description='MATLAB Integration for Solar Fault Detection')
    parser.add_argument('--mode', choices=['single', 'continuous', 'dataset'], default='single',
                        help='Operation mode: single simulation, continuous simulation, or dataset generation')
    parser.add_argument('--irradiance', type=float, default=1000,
                        help='Solar irradiance in W/m² (for single mode)')
    parser.add_argument('--temperature', type=float, default=25,
                        help='Cell temperature in °C (for single mode)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Time between simulations in seconds (for continuous mode)')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Total duration in seconds (for continuous mode)')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples per fault type (for dataset mode)')
    
    args = parser.parse_args()
    
    # Create integration
    integration = MatlabIntegration()
    
    if args.mode == 'single':
        # Run a single simulation and prediction
        logger.info(f"Running single simulation with irradiance={args.irradiance} W/m², temperature={args.temperature}°C")
        result = integration.run_simulation_and_predict(args.irradiance, args.temperature)
        
        if result:
            print("\nSimulation Results:")
            print(f"PV Current: {result['simulation']['normal']['pv_current']:.2f} A")
            print(f"PV Voltage: {result['simulation']['normal']['pv_voltage']:.2f} V")
            print(f"PV Power: {result['simulation']['normal']['pv_power']:.2f} W")
            
            print("\nFault Prediction:")
            print(f"Fault Type: {result['prediction']['label']}")
            print(f"Confidence: {result['prediction']['confidence']:.2f}")
            print(f"Description: {result['prediction']['description']}")
            print(f"Recommended Action: {result['prediction']['recommended_action']}")
            
            print(f"\nRecord ID: {result['record_id']}")
        else:
            print("Simulation and prediction failed")
    
    elif args.mode == 'continuous':
        # Run continuous simulations and predictions
        logger.info(f"Running continuous simulation for {args.duration} seconds with {args.interval} second intervals")
        results = integration.run_continuous_simulation_and_prediction(
            interval=args.interval,
            duration=args.duration
        )
        
        if results:
            print(f"\nCompleted {len(results)} simulations and predictions")
            
            # Count predictions by type
            prediction_counts = {}
            for result in results:
                label = result['prediction']['label']
                prediction_counts[label] = prediction_counts.get(label, 0) + 1
            
            print("\nPrediction Counts:")
            for label, count in prediction_counts.items():
                print(f"{label}: {count}")
        else:
            print("No results from continuous simulation")
    
    elif args.mode == 'dataset':
        # Generate a dataset for training and testing
        logger.info(f"Generating dataset with {args.samples} samples per fault type")
        df = integration.generate_fault_simulation_dataset(num_samples=args.samples)
        
        if df is not None:
            print(f"\nGenerated dataset with {len(df)} samples")
            print("\nSample Distribution:")
            print(df['fault_label'].value_counts())
        else:
            print("Dataset generation failed")

if __name__ == "__main__":
    main()
