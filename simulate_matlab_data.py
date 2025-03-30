"""
Simulate MATLAB data for the Solar Fault Detection System
This script generates synthetic data that mimics what would come from MATLAB simulations
"""
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("matlab_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MatlabSimulation")

class MatlabDataSimulator:
    """
    Simulates MATLAB data for the Solar Fault Detection System
    """
    def __init__(self, output_dir="matlab_data"):
        """
        Initialize the MATLAB data simulator
        
        Args:
            output_dir: Directory to save simulated MATLAB data files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def generate_normal_data(self, num_samples=10, base_irradiance=1000, base_temperature=25):
        """
        Generate data for normal operating conditions
        
        Args:
            num_samples: Number of data points to generate
            base_irradiance: Base irradiance value in W/m²
            base_temperature: Base temperature value in °C
            
        Returns:
            DataFrame containing simulated data
        """
        # Generate time points
        time_points = np.linspace(0, 3, num_samples)
        
        # Generate random variations
        irradiance_variations = np.random.normal(0, 20, num_samples)
        temperature_variations = np.random.normal(0, 1, num_samples)
        
        # Calculate irradiance and temperature
        irradiance = base_irradiance + irradiance_variations
        temperature = base_temperature + temperature_variations
        
        # Calculate PV current and voltage based on simplified model
        # I = Isc * (1 - exp((V - Voc) / Vt))
        pv_current = 8.0 * (irradiance / 1000) * (1 - 0.05 * (temperature - 25) / 100)
        pv_voltage = 30.0 * (1 - 0.004 * (temperature - 25))
        
        # Add small random variations
        pv_current += np.random.normal(0, 0.1, num_samples)
        pv_voltage += np.random.normal(0, 0.2, num_samples)
        
        # Calculate power
        pv_power = pv_current * pv_voltage
        grid_power = pv_power * 0.95  # Assume 95% efficiency
        efficiency = grid_power / pv_power
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_points,
            'irradiance': irradiance,
            'temperature': temperature,
            'pv_current': pv_current,
            'pv_voltage': pv_voltage,
            'pv_power': pv_power,
            'grid_power': grid_power,
            'efficiency': efficiency
        })
        
        return df
    
    def generate_fault_data(self, fault_type, num_samples=10, base_irradiance=1000, base_temperature=25):
        """
        Generate data for fault conditions
        
        Args:
            fault_type: Type of fault (1-4)
            num_samples: Number of data points to generate
            base_irradiance: Base irradiance value in W/m²
            base_temperature: Base temperature value in °C
            
        Returns:
            DataFrame containing simulated fault data
        """
        # Generate normal data first
        df = self.generate_normal_data(num_samples, base_irradiance, base_temperature)
        
        # Modify data based on fault type
        if fault_type == 1:  # Line-Line Fault
            # Higher current, lower voltage (based on memory patterns)
            df['pv_voltage'] *= np.random.uniform(0.6, 0.9, num_samples)  # 10-40% reduction
            df['pv_current'] *= np.random.uniform(1.2, 1.5, num_samples)  # 20-50% increase
            df['temperature'] += np.random.uniform(5, 15, num_samples)    # Higher temperature due to fault
            
        elif fault_type == 2:  # Open Circuit
            # Higher voltage, very low or negative current (based on memory patterns)
            df['pv_voltage'] *= np.random.uniform(1.1, 1.5, num_samples)  # 10-50% increase
            
            # Create negative currents for some samples (strong indicator of Fault_2)
            negative_mask = np.random.choice([True, False], num_samples, p=[0.7, 0.3])
            positive_mask = ~negative_mask
            
            # Apply negative currents (important for Fault_2 detection per memory)
            df.loc[df.index[negative_mask], 'pv_current'] *= np.random.uniform(-1.2, -0.8, sum(negative_mask))
            
            # Very low positive currents for the rest
            df.loc[df.index[positive_mask], 'pv_current'] *= np.random.uniform(0.01, 0.3, sum(positive_mask))
            
        elif fault_type == 3:  # Partial Shading
            # Moderately reduced current and slightly reduced voltage (based on memory patterns)
            df['pv_voltage'] *= np.random.uniform(0.8, 1.0, num_samples)  # 0-20% reduction
            df['pv_current'] *= np.random.uniform(0.5, 0.8, num_samples)  # 20-50% reduction
            df['irradiance'] *= np.random.uniform(0.4, 0.7, num_samples)  # 30-60% reduction in irradiance
            
        elif fault_type == 4:  # Degradation
            # Both reduced current and voltage (based on memory patterns)
            df['pv_voltage'] *= np.random.uniform(0.7, 0.9, num_samples)  # 10-30% reduction
            df['pv_current'] *= np.random.uniform(0.6, 0.9, num_samples)  # 10-40% reduction
            # Add some aging effects
            df['temperature'] += np.random.uniform(2, 8, num_samples)     # Slightly higher temperature
            
        # Recalculate power
        df['pv_power'] = df['pv_current'] * df['pv_voltage']
        df['grid_power'] = df['pv_power'] * 0.95  # Assume 95% efficiency
        df['efficiency'] = df['grid_power'] / df['pv_power'].apply(lambda x: max(abs(x), 0.1))  # Avoid division by zero
        
        return df
    
    def save_data_to_csv(self, df, filename=None):
        """
        Save data to CSV file
        
        Args:
            df: DataFrame containing data to save
            filename: Name of the CSV file (default: auto-generated)
            
        Returns:
            Path to the saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"matlab_data_{timestamp}.csv"
        
        file_path = os.path.join(self.output_dir, filename)
        df.to_csv(file_path, index=False)
        
        logger.info(f"Saved data to {file_path}")
        
        return file_path
    
    def generate_and_save_data(self, fault_type=None, num_samples=10):
        """
        Generate and save data for normal or fault conditions
        
        Args:
            fault_type: Type of fault (1-4) or None for normal conditions
            num_samples: Number of data points to generate
            
        Returns:
            Path to the saved CSV file
        """
        if fault_type is None:
            # Generate normal data
            df = self.generate_normal_data(num_samples)
            file_prefix = "normal"
        else:
            # Generate fault data
            df = self.generate_fault_data(fault_type, num_samples)
            file_prefix = f"fault_{fault_type}"
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{file_prefix}_{timestamp}.csv"
        
        return self.save_data_to_csv(df, filename)
    
    def generate_mixed_dataset(self, num_samples_per_type=5):
        """
        Generate a mixed dataset with normal and fault conditions
        
        Args:
            num_samples_per_type: Number of samples per condition type
            
        Returns:
            Path to the saved CSV file
        """
        # Generate data for each condition
        normal_df = self.generate_normal_data(num_samples_per_type)
        normal_df['condition'] = 'normal'
        normal_df['fault_type'] = 0
        
        fault1_df = self.generate_fault_data(1, num_samples_per_type)
        fault1_df['condition'] = 'fault'
        fault1_df['fault_type'] = 1
        
        fault2_df = self.generate_fault_data(2, num_samples_per_type)
        fault2_df['condition'] = 'fault'
        fault2_df['fault_type'] = 2
        
        fault3_df = self.generate_fault_data(3, num_samples_per_type)
        fault3_df['condition'] = 'fault'
        fault3_df['fault_type'] = 3
        
        fault4_df = self.generate_fault_data(4, num_samples_per_type)
        fault4_df['condition'] = 'fault'
        fault4_df['fault_type'] = 4
        
        # Combine all data
        combined_df = pd.concat([normal_df, fault1_df, fault2_df, fault3_df, fault4_df])
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mixed_dataset_{timestamp}.csv"
        
        return self.save_data_to_csv(combined_df, filename)
    
    def generate_all_data(self, num_samples=20):
        """
        Generate sample data for normal operation and all fault types
        
        Args:
            num_samples: Number of data points to generate for each type
        """
        logger.info("Generating all types of MATLAB simulation data")
        
        # Generate normal data
        logger.info("Generating normal data")
        normal_df = self.generate_normal_data(num_samples)
        normal_filename = f"normal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        normal_path = self.save_data_to_csv(normal_df, normal_filename)
        logger.info(f"Saved normal data to {normal_path}")
        
        # Generate fault data for each fault type
        for fault_type in range(1, 5):
            logger.info(f"Generating Fault_{fault_type} data")
            fault_df = self.generate_fault_data(fault_type, num_samples)
            fault_filename = f"fault_{fault_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            fault_path = self.save_data_to_csv(fault_df, fault_filename)
            logger.info(f"Saved Fault_{fault_type} data to {fault_path}")
            
            # Add a small delay to ensure unique timestamps
            time.sleep(1)
        
        logger.info("Data generation complete")

# If running as a script
if __name__ == "__main__":
    # Create simulator
    simulator = MatlabDataSimulator()
    
    # Generate normal data
    print("Generating normal data...")
    normal_file = simulator.generate_and_save_data(fault_type=None, num_samples=20)
    print(f"Normal data saved to: {normal_file}")
    
    # Generate fault data
    for fault_type in range(1, 5):
        print(f"Generating fault {fault_type} data...")
        fault_file = simulator.generate_and_save_data(fault_type=fault_type, num_samples=20)
        print(f"Fault {fault_type} data saved to: {fault_file}")
    
    # Generate mixed dataset
    print("Generating mixed dataset...")
    mixed_file = simulator.generate_mixed_dataset(num_samples_per_type=10)
    print(f"Mixed dataset saved to: {mixed_file}")
    
    # Generate all data
    print("Generating all data...")
    simulator.generate_all_data(num_samples=20)
    print("All data generation complete!")

    print("Data generation complete!")
