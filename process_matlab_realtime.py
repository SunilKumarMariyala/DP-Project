"""
Process MATLAB Real-Time Data for Solar Fault Detection

This script generates MATLAB-like data and feeds it directly into the prediction system.
It creates a directory for MATLAB data files, generates sample data, and ensures
the prediction system processes this data for fault detection.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matlab_realtime.log')
    ]
)
logger = logging.getLogger('matlab_realtime')

# Path to database
DB_PATH = 'solar_panel.db'

def ensure_matlab_data_dir():
    """Create the MATLAB data directory if it doesn't exist"""
    matlab_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
    if not os.path.exists(matlab_data_dir):
        os.makedirs(matlab_data_dir)
        logger.info(f"Created MATLAB data directory: {matlab_data_dir}")
    return matlab_data_dir

def generate_matlab_data(num_samples=10, fault_type=0):
    """
    Generate sample data that mimics MATLAB output
    
    Args:
        num_samples: Number of data points to generate
        fault_type: Type of fault (0=Normal, 1-4=Fault types)
        
    Returns:
        DataFrame with generated data
    """
    # Base values
    base_voltage = 30.0  # V
    base_current = 8.0   # A
    base_irradiance = 1000.0  # W/m²
    base_temperature = 25.0  # °C
    
    # Generate data with random variations
    data = []
    timestamp = datetime.now()
    
    for i in range(num_samples):
        # Add some random variation
        noise_level = 0.05
        irradiance = base_irradiance * (1 + np.random.uniform(-noise_level, noise_level))
        temperature = base_temperature * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        
        # Calculate voltage and current based on irradiance and temperature
        voltage = base_voltage * (1 - 0.004 * (temperature - 25)) * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        current = base_current * (irradiance / 1000) * (1 + np.random.uniform(-noise_level/2, noise_level/2))
        
        # Apply fault-specific modifications
        if fault_type == 1:  # Line-Line Fault
            voltage *= 0.7
            current *= 1.3
            temperature += 15
        elif fault_type == 2:  # Open Circuit
            voltage *= 1.17
            current *= 0.05
        elif fault_type == 3:  # Partial Shading
            voltage *= 0.95
            current *= 0.95
            irradiance *= 0.8
        elif fault_type == 4:  # Degradation
            voltage *= 1.1
            current = -abs(current) * 1.2
        
        # Calculate power and efficiency
        power = voltage * current
        grid_power = power * 0.95
        efficiency = grid_power / (irradiance * 0.1) if irradiance > 0 else 0
        
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

def save_data_to_file(df, output_dir):
    """Save DataFrame to CSV file in the output directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"matlab_data_{timestamp}.csv"
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
        
        # Commit changes
        conn.commit()
        logger.info(f"Inserted {len(df)} rows into database")
        
        # Close connection
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        return False

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
        return False

def main():
    """Main function to generate and process MATLAB data"""
    # Create MATLAB data directory
    matlab_data_dir = ensure_matlab_data_dir()
    
    # Start prediction process
    prediction_started = start_prediction_process()
    if not prediction_started:
        logger.warning("Could not start prediction process via API, continuing anyway...")
    
    # Generate and process data
    num_iterations = 10
    interval = 2
    
    logger.info(f"Starting MATLAB real-time data generation ({num_iterations} iterations)")
    
    for i in range(num_iterations):
        # Generate data with different fault types
        fault_type = i % 5  # Cycle through normal and 4 fault types
        df = generate_matlab_data(num_samples=5, fault_type=fault_type)
        
        # Save to file
        file_path = save_data_to_file(df, matlab_data_dir)
        
        # Insert directly into database
        insert_success = insert_data_directly(df)
        
        logger.info(f"Iteration {i+1}/{num_iterations}: Generated data with fault type {fault_type}")
        logger.info(f"File: {file_path}, Database insert: {'Success' if insert_success else 'Failed'}")
        
        # Wait for interval if not the last iteration
        if i < num_iterations - 1:
            time.sleep(interval)
    
    logger.info("MATLAB real-time data generation completed successfully!")

if __name__ == "__main__":
    main()
