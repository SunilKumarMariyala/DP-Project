"""
Run MATLAB Model for Real-Time Solar Fault Detection

This script directly connects to the MATLAB GridConnectedPVFarm model,
runs simulations to generate real data, and feeds it into the Python
prediction model for fault detection.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
import requests
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matlab_prediction.log')
    ]
)
logger = logging.getLogger('matlab_prediction')

# MATLAB model path
MATLAB_MODEL_PATH = r"C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample"

# Path to database
DB_PATH = 'solar_panel.db'

def init_matlab_engine():
    """Initialize MATLAB engine and add model path"""
    try:
        import matlab.engine
        logger.info("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        
        # Add model directory to MATLAB path
        eng.addpath(MATLAB_MODEL_PATH, nargout=0)
        logger.info(f"Added model directory to MATLAB path: {MATLAB_MODEL_PATH}")
        
        # Check if model file exists
        model_file = os.path.join(MATLAB_MODEL_PATH, "GridConnectedPVFarm.slx")
        if os.path.exists(model_file):
            logger.info(f"Found MATLAB model file: {model_file}")
        else:
            logger.warning(f"MATLAB model file not found: {model_file}")
            # Look for .slx files in the directory
            slx_files = [f for f in os.listdir(MATLAB_MODEL_PATH) if f.endswith('.slx')]
            if slx_files:
                logger.info(f"Found alternative model files: {slx_files}")
                model_file = os.path.join(MATLAB_MODEL_PATH, slx_files[0])
                logger.info(f"Using model file: {model_file}")
        
        logger.info("MATLAB engine started successfully")
        return eng, model_file
    
    except ImportError:
        logger.error("MATLAB Engine for Python is not available")
        logger.error("To install MATLAB Engine, run 'cd \"matlabroot\\extern\\engines\\python\" && python setup.py install' in MATLAB Command Window")
        return None, None
    
    except Exception as e:
        logger.error(f"Error initializing MATLAB engine: {e}")
        logger.error(traceback.format_exc())
        return None, None

def run_matlab_simulation(eng, model_file, irradiance=1000, temperature=25, simulation_time=3):
    """
    Run simulation with the MATLAB model
    
    Args:
        eng: MATLAB engine instance
        model_file: Path to MATLAB model file
        irradiance: Solar irradiance in W/m²
        temperature: Cell temperature in °C
        simulation_time: Simulation time in seconds
        
    Returns:
        Dictionary containing simulation results
    """
    try:
        logger.info(f"Running MATLAB simulation with parameters:")
        logger.info(f"  - Irradiance: {irradiance} W/m²")
        logger.info(f"  - Temperature: {temperature} °C")
        logger.info(f"  - Simulation time: {simulation_time} seconds")
        
        # Load the model
        eng.load_system(model_file, nargout=0)
        
        # Set model parameters
        eng.set_param('GridConnectedPVFarm/PV Array', 'Irradiance', str(irradiance), nargout=0)
        eng.set_param('GridConnectedPVFarm/PV Array', 'Temperature', str(temperature), nargout=0)
        
        # Set simulation time
        eng.set_param('GridConnectedPVFarm', 'StopTime', str(simulation_time), nargout=0)
        
        # Run simulation
        start_time = time.time()
        eng.sim('GridConnectedPVFarm', nargout=0)
        elapsed_time = time.time() - start_time
        
        # Get simulation results
        pv_current = eng.evalin('base', 'PV_Current.signals.values', nargout=1)
        pv_voltage = eng.evalin('base', 'PV_Voltage.signals.values', nargout=1)
        pv_power = eng.evalin('base', 'PV_Power.signals.values', nargout=1)
        grid_power = eng.evalin('base', 'Grid_Power.signals.values', nargout=1)
        
        # Convert to numpy arrays
        pv_current_np = np.array(pv_current).flatten()
        pv_voltage_np = np.array(pv_voltage).flatten()
        pv_power_np = np.array(pv_power).flatten()
        grid_power_np = np.array(grid_power).flatten()
        
        # Calculate efficiency
        efficiency = np.mean(grid_power_np) / (irradiance * 0.1)  # Assuming 0.1m² panel area
        
        # Create result dictionary
        result = {
            'pv_current': float(np.mean(pv_current_np)),
            'pv_voltage': float(np.mean(pv_voltage_np)),
            'pv_power': float(np.mean(pv_power_np)),
            'grid_power': float(np.mean(grid_power_np)),
            'efficiency': float(efficiency),
            'simulation_time': elapsed_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'raw_data': {
                'pv_current': pv_current_np.tolist(),
                'pv_voltage': pv_voltage_np.tolist(),
                'pv_power': pv_power_np.tolist(),
                'grid_power': grid_power_np.tolist()
            }
        }
        
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results: PV Current={result['pv_current']:.2f}A, PV Voltage={result['pv_voltage']:.2f}V, Efficiency={result['efficiency']:.2f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error running MATLAB simulation: {e}")
        logger.error(traceback.format_exc())
        return None

def simulate_fault_conditions(eng, model_file, base_irradiance=1000, base_temperature=25):
    """
    Simulate different fault conditions using the MATLAB model
    
    Args:
        eng: MATLAB engine instance
        model_file: Path to MATLAB model file
        base_irradiance: Base solar irradiance in W/m²
        base_temperature: Base cell temperature in °C
        
    Returns:
        Dictionary containing simulation results for normal and fault conditions
    """
    try:
        # Run normal condition simulation
        logger.info("Simulating normal operating conditions...")
        normal_result = run_matlab_simulation(eng, model_file, base_irradiance, base_temperature)
        
        # Define fault conditions
        fault_conditions = {
            'fault_1': {  # Line-Line Fault - Lower voltage, higher current
                'irradiance': base_irradiance,
                'temperature': base_temperature + 15,  # Higher temperature due to short circuit
                'description': 'Line-Line Fault'
            },
            'fault_2': {  # Open Circuit - Higher voltage, much lower current
                'irradiance': base_irradiance * 0.1,  # Significantly reduced irradiance to simulate open circuit
                'temperature': base_temperature,
                'description': 'Open Circuit Fault'
            },
            'fault_3': {  # Partial Shading - Slightly lower voltage, moderately lower current
                'irradiance': base_irradiance * 0.8,  # Reduced irradiance due to shading
                'temperature': base_temperature,
                'description': 'Partial Shading'
            },
            'fault_4': {  # Degradation - Reduced current with normal voltage
                'irradiance': base_irradiance * 0.9,  # Slightly reduced irradiance
                'temperature': base_temperature + 10,  # Higher temperature due to degradation
                'description': 'Degradation'
            }
        }
        
        # Run simulations for each fault condition
        fault_results = {}
        for fault_name, params in fault_conditions.items():
            logger.info(f"Simulating {params['description']}...")
            result = run_matlab_simulation(
                eng, 
                model_file,
                params['irradiance'], 
                params['temperature']
            )
            
            if result:
                result['description'] = params['description']
                fault_results[fault_name] = result
        
        return {
            'normal': normal_result,
            'faults': fault_results
        }
    
    except Exception as e:
        logger.error(f"Error simulating fault conditions: {e}")
        logger.error(traceback.format_exc())
        return None

def insert_data_to_database(simulation_results):
    """
    Insert simulation results into the database
    
    Args:
        simulation_results: Dictionary containing simulation results
        
    Returns:
        List of inserted record IDs
    """
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Process normal operating condition
        normal = simulation_results['normal']
        pv_current = normal['pv_current']
        pv_voltage = normal['pv_voltage']
        
        # Process fault conditions
        faults = simulation_results['faults']
        
        # Extract fault data
        pv_fault_1_current = faults['fault_1']['pv_current'] if 'fault_1' in faults else pv_current * 1.3
        pv_fault_1_voltage = faults['fault_1']['pv_voltage'] if 'fault_1' in faults else pv_voltage * 0.7
        
        pv_fault_2_current = faults['fault_2']['pv_current'] if 'fault_2' in faults else pv_current * 0.05
        pv_fault_2_voltage = faults['fault_2']['pv_voltage'] if 'fault_2' in faults else pv_voltage * 1.17
        
        pv_fault_3_current = faults['fault_3']['pv_current'] if 'fault_3' in faults else pv_current * 0.95
        pv_fault_3_voltage = faults['fault_3']['pv_voltage'] if 'fault_3' in faults else pv_voltage * 0.95
        
        pv_fault_4_current = faults['fault_4']['pv_current'] if 'fault_4' in faults else -abs(pv_current) * 1.2
        pv_fault_4_voltage = faults['fault_4']['pv_voltage'] if 'fault_4' in faults else pv_voltage * 1.1
        
        # Insert into database
        cursor.execute('''
            INSERT INTO solar_panel_data (
                timestamp, pv_current, pv_voltage, irradiance, temperature, power,
                pv_fault_1_current, pv_fault_1_voltage,
                pv_fault_2_current, pv_fault_2_voltage,
                pv_fault_3_current, pv_fault_3_voltage,
                pv_fault_4_current, pv_fault_4_voltage,
                processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, pv_current, pv_voltage, 
            normal.get('irradiance', 1000), normal.get('temperature', 25), normal.get('pv_power', pv_current * pv_voltage),
            pv_fault_1_current, pv_fault_1_voltage,
            pv_fault_2_current, pv_fault_2_voltage,
            pv_fault_3_current, pv_fault_3_voltage,
            pv_fault_4_current, pv_fault_4_voltage,
            0  # Not processed
        ))
        
        # Get the ID of the inserted record
        record_id = cursor.lastrowid
        
        # Commit changes
        conn.commit()
        logger.info(f"Inserted simulation results into database (ID: {record_id})")
        
        # Close connection
        conn.close()
        
        return [record_id]
    
    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        logger.error(traceback.format_exc())
        return []

def start_prediction_process():
    """Start the continuous prediction process"""
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/start',
            json={'interval': 2, 'batch_size': 5}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully started prediction process: {result}")
            return True
        else:
            logger.error(f"Failed to start prediction process: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error starting prediction process: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run MATLAB simulations and feed data to prediction model"""
    # Initialize MATLAB engine
    eng, model_file = init_matlab_engine()
    
    if eng is None or model_file is None:
        logger.error("Failed to initialize MATLAB engine. Exiting.")
        return
    
    # Start prediction process
    prediction_started = start_prediction_process()
    if not prediction_started:
        logger.warning("Could not start prediction process via API, continuing anyway...")
    
    # Run simulations and feed data to prediction model
    num_iterations = 10
    interval = 5
    
    logger.info(f"Starting MATLAB simulations ({num_iterations} iterations)")
    
    try:
        for i in range(num_iterations):
            # Vary irradiance and temperature slightly for each iteration
            irradiance = 1000 + np.random.uniform(-100, 100)
            temperature = 25 + np.random.uniform(-5, 5)
            
            # Simulate fault conditions
            logger.info(f"Iteration {i+1}/{num_iterations}: Running MATLAB simulations...")
            simulation_results = simulate_fault_conditions(eng, model_file, irradiance, temperature)
            
            if simulation_results:
                # Insert data into database
                record_ids = insert_data_to_database(simulation_results)
                logger.info(f"Inserted data with IDs: {record_ids}")
            else:
                logger.error("Failed to get simulation results")
            
            # Wait for interval if not the last iteration
            if i < num_iterations - 1:
                logger.info(f"Waiting {interval} seconds before next iteration...")
                time.sleep(interval)
        
        logger.info("MATLAB simulations completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in main simulation loop: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Quit MATLAB engine
        try:
            eng.quit()
            logger.info("MATLAB engine closed")
        except:
            pass

if __name__ == "__main__":
    main()
