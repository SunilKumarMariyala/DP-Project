"""
Solar Panel Data Generator

This script generates synthetic solar panel data for testing the advanced monitoring system.
It simulates normal operation and various fault conditions.
"""

import os
import sys
import time
import random
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_generator.log')
    ]
)
logger = logging.getLogger('data_generator')

# Database path
DB_PATH = 'solar_panel.db'

class SolarDataGenerator:
    """Generate synthetic solar panel data"""
    
    def __init__(self, db_path=DB_PATH):
        """
        Initialize the data generator
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect_db()
        
        # Base parameters for a typical solar panel
        self.base_voltage = 30.0  # Volts
        self.base_current = 8.0   # Amps
        self.base_irradiance = 1000.0  # W/m²
        self.base_temperature = 25.0  # °C
        
        # Time-based variation parameters
        self.time_of_day_factor = 1.0
        self.day_of_year_factor = 1.0
        
        # Fault probabilities (percentage)
        self.fault_probabilities = {
            'Healthy': 70,
            'Fault_1': 10,  # Line-Line fault
            'Fault_2': 8,   # Open circuit
            'Fault_3': 7,   # Partial shading
            'Fault_4': 5    # Panel degradation
        }
        
        # Initialize random seed
        random.seed()
    
    def connect_db(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Check if the solar_panel_data table exists
            self.cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='solar_panel_data'
            ''')
            
            if not self.cursor.fetchone():
                logger.warning("Table 'solar_panel_data' not found. Creating it...")
                self._create_tables()
            
            logger.info(f"Connected to database: {self.db_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
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
        
        # Commit changes
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def update_time_factors(self, timestamp=None):
        """
        Update time-based variation factors
        
        Args:
            timestamp: Datetime to use for calculations (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Time of day factor (0.1 at night, 1.0 at noon)
        hour = timestamp.hour + timestamp.minute / 60.0
        if 6 <= hour < 18:
            # Daytime: bell curve peaking at noon
            self.time_of_day_factor = 0.1 + 0.9 * (1 - ((hour - 12) / 6) ** 2)
        else:
            # Nighttime: minimal factor
            self.time_of_day_factor = 0.1
        
        # Day of year factor (0.7 in winter, 1.0 in summer for northern hemisphere)
        day_of_year = timestamp.timetuple().tm_yday
        self.day_of_year_factor = 0.7 + 0.3 * (1 - ((day_of_year - 172) / 182.5) ** 2)
    
    def generate_normal_data(self, timestamp=None):
        """
        Generate normal (healthy) solar panel data
        
        Args:
            timestamp: Datetime to use (default: current time)
            
        Returns:
            Dictionary of generated data
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update time factors
        self.update_time_factors(timestamp)
        
        # Calculate base values with time factors
        irradiance = self.base_irradiance * self.time_of_day_factor * self.day_of_year_factor
        
        # Add random variations (±5%)
        irradiance *= random.uniform(0.95, 1.05)
        
        # Temperature varies with irradiance but has more inertia
        temperature = self.base_temperature + (irradiance / self.base_irradiance - 0.5) * 20
        temperature *= random.uniform(0.97, 1.03)  # ±3% variation
        
        # Calculate voltage and current based on irradiance and temperature
        # Voltage decreases slightly with temperature (negative temperature coefficient)
        voltage = self.base_voltage * (1 - 0.004 * (temperature - self.base_temperature))
        # Current is roughly proportional to irradiance
        current = self.base_current * (irradiance / self.base_irradiance)
        
        # Add random noise
        voltage *= random.uniform(0.98, 1.02)  # ±2% variation
        current *= random.uniform(0.97, 1.03)  # ±3% variation
        
        # Calculate power
        power = voltage * current
        
        # Calculate grid power (slightly less than panel power due to losses)
        grid_power = power * random.uniform(0.92, 0.97)  # 3-8% losses
        
        # Calculate efficiency
        efficiency = grid_power / (irradiance * 1.7)  # Assuming 1.7m² panel area
        
        # Generate fault values for comparison
        # Fault 1: Line-Line Fault (Lower voltage, higher current)
        fault_1_voltage = voltage * 0.7 * random.uniform(0.95, 1.05)
        fault_1_current = current * 1.3 * random.uniform(0.95, 1.05)
        
        # Fault 2: Open Circuit (Higher voltage, near-zero current)
        fault_2_voltage = voltage * 1.17 * random.uniform(0.95, 1.05)
        fault_2_current = current * 0.05 * random.uniform(0.8, 1.2)
        
        # Fault 3: Partial Shading (Slightly lower voltage, moderately lower current)
        fault_3_voltage = voltage * 0.95 * random.uniform(0.98, 1.02)
        fault_3_current = current * 0.75 * random.uniform(0.95, 1.05)
        
        # Fault 4: Degradation (Lower voltage, lower current)
        fault_4_voltage = voltage * 0.85 * random.uniform(0.98, 1.02)
        fault_4_current = current * 0.8 * random.uniform(0.95, 1.05)
        
        return {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'pv_current': current,
            'pv_voltage': voltage,
            'pv_fault_1_current': fault_1_current,
            'pv_fault_1_voltage': fault_1_voltage,
            'pv_fault_2_current': fault_2_current,
            'pv_fault_2_voltage': fault_2_voltage,
            'pv_fault_3_current': fault_3_current,
            'pv_fault_3_voltage': fault_3_voltage,
            'pv_fault_4_current': fault_4_current,
            'pv_fault_4_voltage': fault_4_voltage,
            'irradiance': irradiance,
            'temperature': temperature,
            'pv_power': power,
            'grid_power': grid_power,
            'efficiency': efficiency,
            'is_matlab_data': 0,
            'simulation_id': f'gen_{timestamp.strftime("%Y%m%d%H%M%S")}'
        }
    
    def generate_fault_data(self, fault_type, timestamp=None):
        """
        Generate fault data
        
        Args:
            fault_type: Type of fault (1, 2, 3, or 4)
            timestamp: Datetime to use (default: current time)
            
        Returns:
            Dictionary of generated data
        """
        # Generate normal data first
        data = self.generate_normal_data(timestamp)
        
        # Apply fault-specific modifications
        if fault_type == 1:  # Line-Line fault
            # Use the fault 1 values as the actual values
            data['pv_voltage'] = data['pv_fault_1_voltage']
            data['pv_current'] = data['pv_fault_1_current']
        
        elif fault_type == 2:  # Open circuit
            # Use the fault 2 values as the actual values
            data['pv_voltage'] = data['pv_fault_2_voltage']
            data['pv_current'] = data['pv_fault_2_current']
        
        elif fault_type == 3:  # Partial shading
            # Use the fault 3 values as the actual values
            data['pv_voltage'] = data['pv_fault_3_voltage']
            data['pv_current'] = data['pv_fault_3_current']
        
        elif fault_type == 4:  # Panel degradation
            # Use the fault 4 values as the actual values
            data['pv_voltage'] = data['pv_fault_4_voltage']
            data['pv_current'] = data['pv_fault_4_current']
        
        # Recalculate power
        data['pv_power'] = data['pv_voltage'] * data['pv_current']
        
        # Adjust grid power and efficiency based on fault
        if fault_type in [1, 2]:
            # Significant impact on grid power
            data['grid_power'] = data['pv_power'] * random.uniform(0.5, 0.8)
        else:
            # Moderate impact on grid power
            data['grid_power'] = data['pv_power'] * random.uniform(0.8, 0.9)
        
        # Recalculate efficiency
        data['efficiency'] = data['grid_power'] / (data['irradiance'] * 1.7)
        
        return data
    
    def generate_random_data(self, timestamp=None):
        """
        Generate random data based on fault probabilities
        
        Args:
            timestamp: Datetime to use (default: current time)
            
        Returns:
            Dictionary of generated data
        """
        # Determine fault type based on probabilities
        rand = random.randint(1, 100)
        cumulative = 0
        
        for fault, probability in self.fault_probabilities.items():
            cumulative += probability
            if rand <= cumulative:
                selected_fault = fault
                break
        
        # Generate data based on selected fault
        if selected_fault == 'Healthy':
            return self.generate_normal_data(timestamp)
        elif selected_fault == 'Fault_1':
            return self.generate_fault_data(1, timestamp)
        elif selected_fault == 'Fault_2':
            return self.generate_fault_data(2, timestamp)
        elif selected_fault == 'Fault_3':
            return self.generate_fault_data(3, timestamp)
        elif selected_fault == 'Fault_4':
            return self.generate_fault_data(4, timestamp)
    
    def insert_data(self, data):
        """
        Insert data into the database
        
        Args:
            data: Dictionary of data to insert
            
        Returns:
            ID of inserted record
        """
        try:
            # Insert data into database
            self.cursor.execute('''
                INSERT INTO solar_panel_data (
                    timestamp, pv_current, pv_voltage, 
                    pv_fault_1_current, pv_fault_1_voltage,
                    pv_fault_2_current, pv_fault_2_voltage,
                    pv_fault_3_current, pv_fault_3_voltage,
                    pv_fault_4_current, pv_fault_4_voltage,
                    irradiance, temperature, pv_power, grid_power, efficiency,
                    is_matlab_data, simulation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'], data['pv_current'], data['pv_voltage'], 
                data['pv_fault_1_current'], data['pv_fault_1_voltage'],
                data['pv_fault_2_current'], data['pv_fault_2_voltage'],
                data['pv_fault_3_current'], data['pv_fault_3_voltage'],
                data['pv_fault_4_current'], data['pv_fault_4_voltage'],
                data['irradiance'], data['temperature'], data['pv_power'], data['grid_power'], data['efficiency'],
                data['is_matlab_data'], data['simulation_id']
            ))
            
            # Commit changes
            self.conn.commit()
            
            # Get the ID of the inserted record
            record_id = self.cursor.lastrowid
            
            logger.info(f"Inserted data with ID: {record_id}")
            
            return record_id
        
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return None
    
    def generate_and_insert_batch(self, count=1, interval=0, start_time=None):
        """
        Generate and insert a batch of data
        
        Args:
            count: Number of records to generate
            interval: Time interval between records in seconds
            start_time: Starting timestamp (default: current time)
            
        Returns:
            List of inserted record IDs
        """
        if start_time is None:
            start_time = datetime.now()
        
        record_ids = []
        
        for i in range(count):
            # Calculate timestamp
            if interval > 0:
                timestamp = start_time + timedelta(seconds=i * interval)
            else:
                timestamp = start_time
            
            # Generate data
            data = self.generate_random_data(timestamp)
            
            # Insert data
            record_id = self.insert_data(data)
            
            if record_id:
                record_ids.append(record_id)
        
        return record_ids
    
    def generate_continuous(self, interval=5, duration=None, fault_scenario=None):
        """
        Generate data continuously
        
        Args:
            interval: Time interval between records in seconds
            duration: Duration in seconds (None for indefinite)
            fault_scenario: Optional fault scenario to simulate
        """
        start_time = datetime.now()
        count = 0
        
        try:
            logger.info(f"Starting continuous data generation with interval={interval}s")
            
            while True:
                # Check if duration is reached
                if duration is not None:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration:
                        logger.info(f"Duration reached ({duration}s). Stopping.")
                        break
                
                # Generate timestamp
                timestamp = datetime.now()
                
                # Generate data based on scenario
                if fault_scenario is None:
                    data = self.generate_random_data(timestamp)
                elif fault_scenario == 'healthy':
                    data = self.generate_normal_data(timestamp)
                elif fault_scenario.startswith('fault_'):
                    fault_type = int(fault_scenario.split('_')[1])
                    data = self.generate_fault_data(fault_type, timestamp)
                else:
                    logger.warning(f"Unknown fault scenario: {fault_scenario}. Using random data.")
                    data = self.generate_random_data(timestamp)
                
                # Insert data
                record_id = self.insert_data(data)
                
                if record_id:
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Generated {count} records so far")
                
                # Wait for next interval
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping.")
        
        finally:
            logger.info(f"Stopped after generating {count} records")
            return count
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Solar Panel Data Generator')
    parser.add_argument('--mode', choices=['batch', 'continuous'], default='continuous',
                        help='Generation mode: batch or continuous')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of records to generate in batch mode')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Time interval between records in seconds')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds for continuous mode (default: indefinite)')
    parser.add_argument('--scenario', choices=['random', 'healthy', 'fault_1', 'fault_2', 'fault_3', 'fault_4'],
                        default='random', help='Fault scenario to simulate')
    parser.add_argument('--db', default=DB_PATH,
                        help='Path to the SQLite database')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SolarDataGenerator(db_path=args.db)
    
    try:
        if args.mode == 'batch':
            logger.info(f"Generating {args.count} records in batch mode")
            record_ids = generator.generate_and_insert_batch(
                count=args.count,
                interval=args.interval
            )
            logger.info(f"Generated {len(record_ids)} records")
        
        else:  # continuous mode
            scenario = None if args.scenario == 'random' else args.scenario
            generator.generate_continuous(
                interval=args.interval,
                duration=args.duration,
                fault_scenario=scenario
            )
    
    finally:
        generator.close()

if __name__ == "__main__":
    main()
