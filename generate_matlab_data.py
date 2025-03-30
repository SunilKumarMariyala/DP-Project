"""
Generate Sample MATLAB Data for Testing

This script generates sample data in CSV format that mimics MATLAB output
from the GridConnectedPVFarm model. It creates realistic solar panel data
with various operating conditions to test the fault detection model.
"""

import os
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matlab_data_generator.log')
    ]
)
logger = logging.getLogger('matlab_data_generator')

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
            'timestamp': (timestamp + timedelta(seconds=i)).strftime('%Y-%m-%d %H:%M:%S.%f'),
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
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matlab_data_{timestamp}.csv"
    
    # Save to CSV
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved MATLAB data to {file_path}")
    
    return file_path

def main():
    """Main function to parse arguments and generate MATLAB data"""
    parser = argparse.ArgumentParser(description='Generate Sample MATLAB Data for Testing')
    
    parser.add_argument('--output-dir', type=str, default='matlab_data',
                        help='Directory to save generated data files')
    parser.add_argument('--num-files', type=int, default=5,
                        help='Number of data files to generate')
    parser.add_argument('--samples-per-file', type=int, default=10,
                        help='Number of data points per file')
    parser.add_argument('--fault-type', type=int, default=0,
                        help='Type of fault to generate (0=Normal, 1=Line-Line, 2=Open Circuit, 3=Partial Shading, 4=Degradation)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Interval in seconds between file generation')
    parser.add_argument('--irradiance', type=float, default=1000.0,
                        help='Base irradiance value in W/m²')
    parser.add_argument('--temperature', type=float, default=25.0,
                        help='Base temperature value in °C')
    
    args = parser.parse_args()
    
    logger.info(f"Generating {args.num_files} MATLAB data files with {args.samples_per_file} samples each")
    logger.info(f"Fault type: {args.fault_type} (0=Normal, 1=Line-Line, 2=Open Circuit, 3=Partial Shading, 4=Degradation)")
    
    import time
    
    for i in range(args.num_files):
        # Generate data based on fault type
        if args.fault_type == 0:
            df = generate_normal_data(args.samples_per_file, args.irradiance, args.temperature)
        else:
            df = generate_fault_data(args.fault_type, args.samples_per_file, args.irradiance, args.temperature)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matlab_data_{timestamp}.csv"
        save_matlab_data(df, args.output_dir, filename)
        
        logger.info(f"Generated file {i+1}/{args.num_files}")
        
        # Wait for interval if not the last file
        if i < args.num_files - 1:
            time.sleep(args.interval)
    
    logger.info("MATLAB data generation completed successfully!")
    logger.info(f"Files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
