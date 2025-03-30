"""
Simulate MATLAB Data and Setup File Watching for Solar Fault Detection

This script generates simulated data in the format expected from MATLAB,
saves it to files in a watch directory, and configures the system to
monitor and process these files for fault detection.
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
import json
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matlab_simulation.log')
    ]
)
logger = logging.getLogger('matlab_simulation')

# Path to database
DB_PATH = 'solar_panel.db'

def ensure_matlab_data_dir():
    """Create the MATLAB data directory if it doesn't exist"""
    matlab_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
    if not os.path.exists(matlab_data_dir):
        os.makedirs(matlab_data_dir)
        logger.info(f"Created MATLAB data directory: {matlab_data_dir}")
    return matlab_data_dir

def generate_normal_data(num_samples=10, base_irradiance=1000, base_temperature=25, noise_level=0.05):
    """
    Generate normal operating condition data for solar panels
    
    Args:
        num_samples: Number of data points to generate
        base_irradiance: Base solar irradiance in W/m²
        base_temperature: Base cell temperature in °C
        noise_level: Level of random noise to add
        
    Returns:
        DataFrame with generated data
    """
    # Base values for normal operation
    base_voltage = 30.0  # V
    base_current = 8.0   # A
    
    # Generate data with random variations
    data = []
    timestamp = datetime.now()
    
    for i in range(num_samples):
        # Add some random variation
        irradiance = base_irradiance * (1 + np.random.uniform(-noise_level, noise_level))
        temperature = base_temperature * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        
        # Calculate voltage and current based on irradiance and temperature
        # Simplified model: voltage decreases with temperature, current increases with irradiance
        voltage = base_voltage * (1 - 0.004 * (temperature - 25)) * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        current = base_current * (irradiance / 1000) * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        
        # Calculate power and efficiency
        power = voltage * current
        grid_power = power * 0.95  # Assuming 95% inverter efficiency
        efficiency = grid_power / (irradiance * 0.1)  # Assuming 0.1m² panel area
        
        # Add to data
        data.append({
            'timestamp': (timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            'pv_voltage': voltage,
            'pv_current': current,
            'pv_power': power,
            'grid_power': grid_power,
            'irradiance': irradiance,
            'temperature': temperature,
            'efficiency': efficiency
        })
    
    return pd.DataFrame(data)

def generate_fault_data(fault_type, num_samples=10, base_irradiance=1000, base_temperature=25, noise_level=0.05):
    """
    Generate fault condition data for solar panels
    
    Args:
        fault_type: Type of fault (1=Line-Line, 2=Open Circuit, 3=Partial Shading, 4=Degradation)
        num_samples: Number of data points to generate
        base_irradiance: Base solar irradiance in W/m²
        base_temperature: Base cell temperature in °C
        noise_level: Level of random noise to add
        
    Returns:
        DataFrame with generated fault data
    """
    # Get normal data as a starting point
    df = generate_normal_data(num_samples, base_irradiance, base_temperature, noise_level)
    
    # Apply fault-specific modifications
    if fault_type == 1:  # Line-Line Fault
        # Lower voltage, higher current
        df['pv_voltage'] *= 0.7
        df['pv_current'] *= 1.3
        df['temperature'] += 15  # Higher temperature due to short circuit
        
    elif fault_type == 2:  # Open Circuit
        # Higher voltage, near-zero current
        df['pv_voltage'] *= 1.17
        df['pv_current'] *= 0.05
        
    elif fault_type == 3:  # Partial Shading
        # Slightly lower voltage, moderately lower current
        df['pv_voltage'] *= 0.95
        df['pv_current'] *= 0.95
        df['irradiance'] *= 0.8  # Lower irradiance due to shading
        
    elif fault_type == 4:  # Degradation
        # Higher voltage, negative current
        df['pv_voltage'] *= 1.1
        current_magnitude = df['pv_current'].abs() * 1.2
        df['pv_current'] = -current_magnitude
    
    # Recalculate power and efficiency
    df['pv_power'] = df['pv_voltage'] * df['pv_current']
    df['grid_power'] = df['pv_power'] * 0.95
    df['efficiency'] = df['grid_power'] / (df['irradiance'] * 0.1)
    
    return df

def save_matlab_data(df, output_dir, filename=None):
    """
    Save generated data to a CSV file in MATLAB format
    
    Args:
        df: DataFrame with generated data
        output_dir: Directory to save the file
        filename: Optional filename, if None a timestamp-based name will be used
        
    Returns:
        Path to the saved file
    """
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matlab_data_{timestamp}.csv"
    
    # Save to CSV
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved MATLAB data to {file_path}")
    
    return file_path

def insert_data_directly(df):
    """Insert data directly into the database for processing"""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Process each row
        record_ids = []
        for _, row in df.iterrows():
            pv_current = row['pv_current']
            pv_voltage = row['pv_voltage']
            
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
                row.get('irradiance', 1000), row.get('temperature', 25), row.get('pv_power', pv_current * pv_voltage),
                pv_fault_1_current, pv_fault_1_voltage,
                pv_fault_2_current, pv_fault_2_voltage,
                pv_fault_3_current, pv_fault_3_voltage,
                pv_fault_4_current, pv_fault_4_voltage,
                0  # Not processed
            ))
            
            # Get the ID of the inserted record
            record_ids.append(cursor.lastrowid)
        
        # Commit changes
        conn.commit()
        logger.info(f"Inserted {len(df)} rows into database")
        
        # Close connection
        conn.close()
        return record_ids
    
    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        logger.error(traceback.format_exc())
        return []

def setup_matlab_watch(watch_dir, file_pattern='*.csv', check_interval=2):
    """
    Set up the MATLAB data watch using the API
    
    Args:
        watch_dir: Directory to watch for MATLAB data files
        file_pattern: Pattern to match files (default: *.csv)
        check_interval: Interval in seconds to check for new files (default: 2)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/matlab/setup_watch',
            json={
                'watch_directory': watch_dir,
                'file_pattern': file_pattern,
                'check_interval': check_interval
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully set up MATLAB watch: {result}")
            return True
        else:
            logger.error(f"Failed to set up MATLAB watch: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error setting up MATLAB watch: {e}")
        logger.error(traceback.format_exc())
        return False

def start_prediction_process(interval=2, batch_size=5):
    """
    Start the continuous prediction process
    
    Args:
        interval: Interval in seconds between prediction cycles (default: 2)
        batch_size: Number of records to process in each cycle (default: 5)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/start',
            json={
                'interval': interval,
                'batch_size': batch_size
            }
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
    """Main function to generate MATLAB-like data and set up processing"""
    # Create MATLAB data directory
    matlab_data_dir = ensure_matlab_data_dir()
    
    # Start prediction process
    prediction_started = start_prediction_process(interval=2, batch_size=5)
    if not prediction_started:
        logger.warning("Could not start prediction process via API, continuing anyway...")
    
    # Set up MATLAB watch
    watch_setup = setup_matlab_watch(matlab_data_dir, check_interval=2)
    if not watch_setup:
        logger.warning("Could not set up MATLAB watch via API, continuing with direct database insertion...")
    
    # Generate and process data
    num_iterations = 20
    interval = 3
    
    logger.info(f"Starting MATLAB-like data generation ({num_iterations} iterations)")
    
    try:
        for i in range(num_iterations):
            # Generate data with different fault types
            # Cycle through: normal, fault_1, fault_2, fault_3, fault_4
            fault_type = i % 5  
            
            # Add some variation to base values
            base_irradiance = 1000 + random.uniform(-100, 100)
            base_temperature = 25 + random.uniform(-5, 5)
            
            # Generate data
            if fault_type == 0:
                df = generate_normal_data(
                    num_samples=5, 
                    base_irradiance=base_irradiance, 
                    base_temperature=base_temperature
                )
                fault_name = "Normal"
            else:
                df = generate_fault_data(
                    fault_type=fault_type, 
                    num_samples=5, 
                    base_irradiance=base_irradiance, 
                    base_temperature=base_temperature
                )
                fault_names = {1: "Line-Line Fault", 2: "Open Circuit", 3: "Partial Shading", 4: "Degradation"}
                fault_name = fault_names[fault_type]
            
            # Save to file for MATLAB watch to pick up
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"matlab_data_{fault_name.replace(' ', '_').lower()}_{timestamp}.csv"
            file_path = save_matlab_data(df, matlab_data_dir, filename)
            
            # Insert directly into database as backup
            record_ids = insert_data_directly(df)
            
            logger.info(f"Iteration {i+1}/{num_iterations}: Generated {fault_name} data")
            logger.info(f"File: {file_path}, Database records: {record_ids}")
            
            # Wait for interval if not the last iteration
            if i < num_iterations - 1:
                logger.info(f"Waiting {interval} seconds before next iteration...")
                time.sleep(interval)
        
        logger.info("MATLAB-like data generation completed successfully!")
        logger.info(f"Generated files are in: {os.path.abspath(matlab_data_dir)}")
        logger.info("The prediction system should be processing this data in real-time.")
        logger.info("Check the web dashboard to see the results.")
    
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
