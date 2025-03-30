import os
import sys
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from database_setup import SolarPanelData, setup_database
# Import our simulator
from simulate_matlab_data import MatlabDataSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("matlab_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MatlabInterface")

class MatlabInterface:
    """
    Interface to connect the MATLAB GridConnectedPVFarm model with the solar fault detection system
    """
    def __init__(self, matlab_path=None, model_path=None, db_connection_str=None):
        """
        Initialize the MATLAB interface
        
        Args:
            matlab_path: Path to MATLAB executable
            model_path: Path to the MATLAB model file
            db_connection_str: Database connection string for MySQL
        """
        self.matlab_path = matlab_path or r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe"
        self.model_path = model_path or r"C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample"
        self.db_connection_str = db_connection_str
        
        # Setup database connection
        try:
            self.engine, self.Session = setup_database(self.db_connection_str)
            logger.info(f"Database connection established: {self.db_connection_str}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        
        # Check if MATLAB Engine for Python is available
        try:
            import matlab.engine
            self.matlab_available = True
            logger.info("MATLAB Engine for Python is available")
            
            # Start MATLAB engine
            try:
                logger.info("Starting MATLAB engine...")
                self.eng = matlab.engine.start_matlab()
                
                # Add model directory to MATLAB path
                self.eng.addpath(self.model_path, nargout=0)
                logger.info(f"Added model directory to MATLAB path: {self.model_path}")
                
                # Initialize model
                model_file = os.path.join(self.model_path, "GridConnectedPVFarm.slx")
                if os.path.exists(model_file):
                    logger.info(f"Found MATLAB model file: {model_file}")
                else:
                    logger.warning(f"MATLAB model file not found: {model_file}")
                    # Look for .slx files in the directory
                    slx_files = [f for f in os.listdir(self.model_path) if f.endswith('.slx')]
                    if slx_files:
                        logger.info(f"Found alternative model files: {slx_files}")
                
                logger.info("MATLAB engine started successfully")
            except Exception as e:
                logger.error(f"Error starting MATLAB engine: {e}")
                self.eng = None
        except ImportError:
            logger.warning("MATLAB Engine for Python is not available")
            self.matlab_available = False
            self.eng = None
        
        # Initialize simulator if MATLAB is not available
        if not self.matlab_available:
            self.simulator = MatlabDataSimulator()
            logger.info("Initialized MATLAB data simulator")

    def __del__(self):
        """
        Clean up MATLAB engine when the object is destroyed
        """
        if hasattr(self, 'eng') and self.eng is not None:
            try:
                self.eng.quit()
                logger.info("MATLAB engine stopped")
            except Exception as e:
                logger.error(f"Error stopping MATLAB engine: {e}")

    def run_simulation(self, irradiance=1000, temperature=25, simulation_time=3):
        """
        Run the MATLAB simulation with the specified parameters
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            simulation_time: Simulation time in seconds
            
        Returns:
            Dictionary containing simulation results
        """
        if self.matlab_available and self.eng is not None:
            try:
                # Start timing
                start_time = time.time()
                
                # Set simulation parameters
                logger.info(f"Setting simulation parameters: Irradiance={irradiance} W/m², Temperature={temperature}°C, Time={simulation_time}s")
                
                # Load the model
                model_file = os.path.join(self.model_path, "GridConnectedPVFarm.slx")
                self.eng.load_system(model_file, nargout=0)
                
                # Set model parameters
                self.eng.set_param('GridConnectedPVFarm/PV Array', 'Irradiance', str(irradiance), nargout=0)
                self.eng.set_param('GridConnectedPVFarm/PV Array', 'Temperature', str(temperature), nargout=0)
                self.eng.set_param('GridConnectedPVFarm', 'StopTime', str(simulation_time), nargout=0)
                
                # Run the simulation
                logger.info("Running MATLAB simulation...")
                self.eng.sim('GridConnectedPVFarm', nargout=0)
                
                # Get simulation results
                logger.info("Retrieving simulation results...")
                pv_current = self.eng.evalin('base', 'PV_I.signals.values(end,:)', nargout=1)
                pv_voltage = self.eng.evalin('base', 'PV_V.signals.values(end,:)', nargout=1)
                pv_power = self.eng.evalin('base', 'PV_P.signals.values(end,:)', nargout=1)
                grid_power = self.eng.evalin('base', 'Grid_P.signals.values(end,:)', nargout=1)
                
                # Convert MATLAB arrays to Python
                pv_current = np.array(pv_current).flatten()
                pv_voltage = np.array(pv_voltage).flatten()
                pv_power = np.array(pv_power).flatten()
                grid_power = np.array(grid_power).flatten()
                
                # Calculate efficiency
                efficiency = np.mean(grid_power) / np.mean(pv_power) if np.mean(pv_power) > 0 else 0
                
                # Record simulation time
                simulation_time = time.time() - start_time
                
                # Create result dictionary
                result = {
                    'pv_current': float(np.mean(pv_current)),
                    'pv_voltage': float(np.mean(pv_voltage)),
                    'pv_power': float(np.mean(pv_power)),
                    'grid_power': float(np.mean(grid_power)),
                    'efficiency': float(efficiency),
                    'simulation_time': simulation_time,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
                logger.info(f"Results: PV Current={result['pv_current']:.2f}A, PV Voltage={result['pv_voltage']:.2f}V, Efficiency={result['efficiency']:.2f}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error running MATLAB simulation: {e}")
                logger.warning("Falling back to simulator")
                return self._run_simulation_with_simulator(irradiance, temperature, simulation_time)
        else:
            # Use simulator instead
            return self._run_simulation_with_simulator(irradiance, temperature, simulation_time)

    def _run_simulation_with_simulator(self, irradiance=1000, temperature=25, simulation_time=3):
        """
        Run a simulation using the simulator instead of MATLAB
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            simulation_time: Simulation time in seconds
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            # Start timing
            start_time = time.time()
            
            logger.info(f"Running simulation with simulator: Irradiance={irradiance} W/m², Temperature={temperature}°C")
            
            # Generate data with simulator
            num_samples = int(simulation_time * 10)  # 10 samples per second
            df = self.simulator.generate_normal_data(num_samples, irradiance, temperature)
            
            # Calculate mean values
            pv_current = df['pv_current'].mean()
            pv_voltage = df['pv_voltage'].mean()
            pv_power = df['pv_power'].mean()
            grid_power = df['grid_power'].mean()
            efficiency = df['efficiency'].mean()
            
            # Record simulation time
            simulation_time = time.time() - start_time
            
            # Create result dictionary
            result = {
                'pv_current': float(pv_current),
                'pv_voltage': float(pv_voltage),
                'pv_power': float(pv_power),
                'grid_power': float(grid_power),
                'efficiency': float(efficiency),
                'simulation_time': simulation_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Simulation (simulator) completed in {simulation_time:.2f} seconds")
            logger.info(f"Results: PV Current={result['pv_current']:.2f}A, PV Voltage={result['pv_voltage']:.2f}V, Efficiency={result['efficiency']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running simulation with simulator: {e}")
            return None

    def simulate_fault_conditions(self, base_irradiance=1000, base_temperature=25):
        """
        Simulate different fault conditions using the MATLAB model
        
        Args:
            base_irradiance: Base solar irradiance in W/m²
            base_temperature: Base cell temperature in °C
            
        Returns:
            Dictionary containing simulation results for normal and fault conditions
        """
        if self.matlab_available and self.eng is not None:
            try:
                # Run normal condition simulation
                logger.info("Simulating normal operating conditions...")
                normal_result = self.run_simulation(base_irradiance, base_temperature)
                
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
                    'fault_3': {  # Partial Shading - Slightly lower voltage, much lower current
                        'irradiance': base_irradiance * 0.6,  # Reduced irradiance to simulate shading
                        'temperature': base_temperature + 5,  # Slightly higher temperature due to hotspots
                        'description': 'Partial Shading'
                    },
                    'fault_4': {  # Degradation - Normal voltage, lower current
                        'irradiance': base_irradiance * 0.8,  # Reduced irradiance to simulate degradation
                        'temperature': base_temperature + 10,  # Higher temperature due to degradation
                        'description': 'Degradation'
                    }
                }
                
                # Simulate each fault condition
                fault_results = {}
                for fault_name, fault_params in fault_conditions.items():
                    logger.info(f"Simulating {fault_params['description']}...")
                    fault_results[fault_name] = self.run_simulation(
                        fault_params['irradiance'], 
                        fault_params['temperature']
                    )
                    fault_results[fault_name]['description'] = fault_params['description']
                
                # Combine results
                combined_result = {
                    'normal': normal_result,
                    'faults': fault_results,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return combined_result
                
            except Exception as e:
                logger.error(f"Error simulating fault conditions: {e}")
                logger.warning("Falling back to simulator")
                return self._simulate_fault_conditions_with_simulator(base_irradiance, base_temperature)
        else:
            # Use simulator instead
            return self._simulate_fault_conditions_with_simulator(base_irradiance, base_temperature)

    def _simulate_fault_conditions_with_simulator(self, base_irradiance=1000, base_temperature=25):
        """
        Simulate different fault conditions using the simulator
        
        Args:
            base_irradiance: Base solar irradiance in W/m²
            base_temperature: Base cell temperature in °C
            
        Returns:
            Dictionary containing simulation results for normal and fault conditions
        """
        try:
            # Run normal condition simulation
            logger.info("Simulating normal operating conditions with simulator...")
            normal_df = self.simulator.generate_normal_data(10, base_irradiance, base_temperature)
            
            normal_result = {
                'pv_current': float(normal_df['pv_current'].mean()),
                'pv_voltage': float(normal_df['pv_voltage'].mean()),
                'pv_power': float(normal_df['pv_power'].mean()),
                'grid_power': float(normal_df['grid_power'].mean()),
                'efficiency': float(normal_df['efficiency'].mean()),
                'simulation_time': 0.1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Simulate fault conditions
            fault_results = {}
            
            # Fault 1: Line-Line Fault
            logger.info("Simulating Line-Line Fault with simulator...")
            fault1_df = self.simulator.generate_fault_data(1, 10, base_irradiance, base_temperature)
            fault_results['fault_1'] = {
                'pv_current': float(fault1_df['pv_current'].mean()),
                'pv_voltage': float(fault1_df['pv_voltage'].mean()),
                'pv_power': float(fault1_df['pv_power'].mean()),
                'grid_power': float(fault1_df['grid_power'].mean()),
                'efficiency': float(fault1_df['efficiency'].mean()),
                'simulation_time': 0.1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Line-Line Fault'
            }
            
            # Fault 2: Open Circuit
            logger.info("Simulating Open Circuit Fault with simulator...")
            fault2_df = self.simulator.generate_fault_data(2, 10, base_irradiance, base_temperature)
            fault_results['fault_2'] = {
                'pv_current': float(fault2_df['pv_current'].mean()),
                'pv_voltage': float(fault2_df['pv_voltage'].mean()),
                'pv_power': float(fault2_df['pv_power'].mean()),
                'grid_power': float(fault2_df['grid_power'].mean()),
                'efficiency': float(fault2_df['efficiency'].mean()),
                'simulation_time': 0.1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Open Circuit Fault'
            }
            
            # Fault 3: Partial Shading
            logger.info("Simulating Partial Shading with simulator...")
            fault3_df = self.simulator.generate_fault_data(3, 10, base_irradiance, base_temperature)
            fault_results['fault_3'] = {
                'pv_current': float(fault3_df['pv_current'].mean()),
                'pv_voltage': float(fault3_df['pv_voltage'].mean()),
                'pv_power': float(fault3_df['pv_power'].mean()),
                'grid_power': float(fault3_df['grid_power'].mean()),
                'efficiency': float(fault3_df['efficiency'].mean()),
                'simulation_time': 0.1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Partial Shading'
            }
            
            # Fault 4: Degradation
            logger.info("Simulating Degradation with simulator...")
            fault4_df = self.simulator.generate_fault_data(4, 10, base_irradiance, base_temperature)
            fault_results['fault_4'] = {
                'pv_current': float(fault4_df['pv_current'].mean()),
                'pv_voltage': float(fault4_df['pv_voltage'].mean()),
                'pv_power': float(fault4_df['pv_power'].mean()),
                'grid_power': float(fault4_df['grid_power'].mean()),
                'efficiency': float(fault4_df['efficiency'].mean()),
                'simulation_time': 0.1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Degradation'
            }
            
            # Combine results
            combined_result = {
                'normal': normal_result,
                'faults': fault_results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error simulating fault conditions with simulator: {e}")
            return None

    def save_simulation_to_db(self, simulation_result):
        """
        Save simulation results to the database
        
        Args:
            simulation_result: Dictionary containing simulation results
            
        Returns:
            ID of the created database record
        """
        if simulation_result is None:
            logger.error("No simulation result to save")
            return None
        
        session = self.Session()
        try:
            # Extract data from simulation result
            normal = simulation_result['normal']
            faults = simulation_result['faults']
            
            # Create database record
            data_entry = SolarPanelData(
                pv_current=normal['pv_current'],
                pv_voltage=normal['pv_voltage'],
                pv_fault_1_current=faults['fault_1']['pv_current'],
                pv_fault_1_voltage=faults['fault_1']['pv_voltage'],
                pv_fault_2_current=faults['fault_2']['pv_current'],
                pv_fault_2_voltage=faults['fault_2']['pv_voltage'],
                pv_fault_3_current=faults['fault_3']['pv_current'],
                pv_fault_3_voltage=faults['fault_3']['pv_voltage'],
                pv_fault_4_current=faults['fault_4']['pv_current'],
                pv_fault_4_voltage=faults['fault_4']['pv_voltage']
            )
            
            session.add(data_entry)
            session.commit()
            
            logger.info(f"Simulation results saved to database with ID {data_entry.id}")
            
            return data_entry.id
            
        except Exception as e:
            logger.error(f"Error saving simulation to database: {e}")
            session.rollback()
            return None
        finally:
            session.close()

    def run_and_save_simulation(self, irradiance=1000, temperature=25):
        """
        Run a simulation and save the results to the database
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            
        Returns:
            ID of the created database record
        """
        # Run simulation
        simulation_result = self.simulate_fault_conditions(irradiance, temperature)
        
        # Save to database
        if simulation_result:
            return self.save_simulation_to_db(simulation_result)
        else:
            return None

    def generate_simulation_data(self, num_samples=10, random_seed=42):
        """
        Generate multiple simulation data points with varying conditions
        
        Args:
            num_samples: Number of simulation samples to generate
            random_seed: Random seed for reproducibility
            
        Returns:
            List of database record IDs
        """
        np.random.seed(random_seed)
        
        record_ids = []
        for i in range(num_samples):
            # Generate random irradiance between 600 and 1200 W/m²
            irradiance = np.random.uniform(600, 1200)
            
            # Generate random temperature between 15 and 45 °C
            temperature = np.random.uniform(15, 45)
            
            logger.info(f"Generating sample {i+1}/{num_samples}: Irradiance={irradiance:.2f} W/m², Temperature={temperature:.2f}°C")
            
            # Run simulation and save to database
            record_id = self.run_and_save_simulation(irradiance, temperature)
            
            if record_id:
                record_ids.append(record_id)
            
            # Sleep to avoid overloading the system
            time.sleep(1)
        
        logger.info(f"Generated {len(record_ids)} simulation samples")
        
        return record_ids

    def process_real_time_matlab_data(self, matlab_data_file):
        """
        Process real-time data from MATLAB and store in database
        
        Args:
            matlab_data_file: Path to MATLAB data file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing real-time MATLAB data from {matlab_data_file}")
        
        try:
            # Check file extension
            file_ext = os.path.splitext(matlab_data_file)[1].lower()
            
            # Load data based on file type
            if file_ext == '.mat':
                # Load .mat file
                if self.matlab_available and self.eng is not None:
                    # Load the .mat file in MATLAB
                    self.eng.load(matlab_data_file, nargout=0)
                    
                    # Get variable names in the workspace
                    var_names = self.eng.eval('who', nargout=1)
                    
                    # Extract data from MATLAB workspace
                    data = {}
                    for var_name in var_names:
                        data[var_name] = np.array(self.eng.workspace[var_name])
                    
                    # Convert to DataFrame
                    if 'pv_current' in data and 'pv_voltage' in data:
                        df = pd.DataFrame({
                            'pv_current': data['pv_current'].flatten(),
                            'pv_voltage': data['pv_voltage'].flatten()
                        })
                        
                        # Add additional columns if available
                        if 'irradiance' in data:
                            df['irradiance'] = data['irradiance'].flatten()
                        if 'temperature' in data:
                            df['temperature'] = data['temperature'].flatten()
                        if 'power' in data:
                            df['power'] = data['power'].flatten()
                        elif 'pv_power' in data:
                            df['power'] = data['pv_power'].flatten()
                    else:
                        logger.error("Required variables not found in MATLAB data file")
                        return None
                else:
                    logger.error("MATLAB engine is not available to load .mat file")
                    return None
                
            elif file_ext == '.csv':
                # Load CSV file
                df = pd.read_csv(matlab_data_file)
                
                # Check required columns
                if 'pv_current' not in df.columns or 'pv_voltage' not in df.columns:
                    logger.error("Required columns not found in CSV file")
                    return None
            
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            # Process the data
            logger.info(f"Loaded data with {len(df)} rows")
            
            # Generate fault data based on the normal data
            # This simulates different fault conditions for each data point
            processed_rows = []
            
            # Create session
            session = self.Session()
            
            # Process each row
            for _, row in df.iterrows():
                pv_current = row['pv_current']
                pv_voltage = row['pv_voltage']
                
                # Calculate additional features
                irradiance = row.get('irradiance', pv_current * 1000 / (pv_voltage * 0.15) if pv_voltage > 0.1 else 0)
                temperature = row.get('temperature', 25 + (pv_current * 2))
                power = row.get('power', pv_current * pv_voltage)
                
                # Generate synthetic fault data for the model
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
                current_magnitude = abs(pv_current) * 1.2
                pv_fault_4_current = -current_magnitude
                
                # Create new record
                record = SolarPanelData(
                    timestamp=datetime.now(),
                    pv_current=pv_current,
                    pv_voltage=pv_voltage,
                    irradiance=irradiance,
                    temperature=temperature,
                    power=power,
                    pv_fault_1_current=pv_fault_1_current,
                    pv_fault_1_voltage=pv_fault_1_voltage,
                    pv_fault_2_current=pv_fault_2_current,
                    pv_fault_2_voltage=pv_fault_2_voltage,
                    pv_fault_3_current=pv_fault_3_current,
                    pv_fault_3_voltage=pv_fault_3_voltage,
                    pv_fault_4_current=pv_fault_4_current,
                    pv_fault_4_voltage=pv_fault_4_voltage,
                    processed=False
                )
                
                # Add to session
                session.add(record)
                processed_rows.append(record)
            
            # Commit to database
            session.commit()
            
            # Get IDs of inserted records
            record_ids = [record.id for record in processed_rows]
            
            # Close session
            session.close()
            
            # Return results
            return {
                'processed_count': len(processed_rows),
                'record_ids': record_ids,
                'file_path': matlab_data_file
            }
            
        except Exception as e:
            logger.error(f"Error processing MATLAB data: {e}")
            return None

    def setup_matlab_data_watch(self, watch_directory, file_pattern='*.csv', check_interval=5):
        """
        Set up a watch for new MATLAB data files in a directory
        
        Args:
            watch_directory: Directory to watch for new files
            file_pattern: Pattern to match files (default: *.csv)
            check_interval: Interval in seconds to check for new files
            
        Returns:
            None
        """
        import glob
        from threading import Thread
        
        logger.info(f"Setting up watch for MATLAB data files in {watch_directory}")
        logger.info(f"File pattern: {file_pattern}, Check interval: {check_interval}s")
        
        # Create directory if it doesn't exist
        if not os.path.exists(watch_directory):
            os.makedirs(watch_directory)
            logger.info(f"Created watch directory: {watch_directory}")
        
        # Get initial list of files
        processed_files = set()
        
        def watch_thread():
            nonlocal processed_files
            
            while True:
                try:
                    # Get list of files matching pattern
                    files = glob.glob(os.path.join(watch_directory, file_pattern))
                    
                    # Process new files
                    for file_path in files:
                        if file_path not in processed_files:
                            logger.info(f"Found new MATLAB data file: {file_path}")
                            
                            # Process the file
                            result = self.process_real_time_matlab_data(file_path)
                            
                            if result:
                                logger.info(f"Successfully processed {result['processed_count']} rows from {file_path}")
                            else:
                                logger.error(f"Failed to process {file_path}")
                            
                            # Add to processed files
                            processed_files.add(file_path)
                    
                    # Sleep for the specified interval
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in watch thread: {e}")
                    time.sleep(check_interval)
        
        # Start the watch thread
        thread = Thread(target=watch_thread, daemon=True)
        thread.start()
        
        logger.info("MATLAB data watch thread started")

    def get_latest_data_from_db(self):
        """
        Retrieve the latest data from the database
        
        Returns:
            Dictionary containing the latest solar panel data
        """
        session = self.Session()
        try:
            # Get the latest record from the database
            latest_data = session.query(SolarPanelData).order_by(desc(SolarPanelData.timestamp)).first()
            
            if latest_data is None:
                logger.warning("No data found in the database")
                return None
            
            # Create result dictionary
            result = {
                'pv_current': float(latest_data.pv_current),
                'pv_voltage': float(latest_data.pv_voltage),
                'pv_power': float(latest_data.pv_current * latest_data.pv_voltage),
                'timestamp': latest_data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if latest_data.timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Retrieved latest data from database: PV Current={result['pv_current']:.2f}A, PV Voltage={result['pv_voltage']:.2f}V")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving data from database: {e}")
            return None
        finally:
            session.close()

# If running as a script
if __name__ == "__main__":
    # Create MATLAB interface
    interface = MatlabInterface(db_connection_str="mysql+pymysql://solar_user:your_secure_password@localhost/solar_panel_db")
    
    # Check if MATLAB is available
    if interface.matlab_available and interface.eng is not None:
        # Run a single simulation
        print("Running a single simulation...")
        result = interface.run_simulation()
        print(f"PV Current: {result['pv_current']:.2f} A")
        print(f"PV Voltage: {result['pv_voltage']:.2f} V")
        print(f"PV Power: {result['pv_power']:.2f} W")
        print(f"Grid Power: {result['grid_power']:.2f} W")
        print(f"Efficiency: {result['efficiency']:.2f}")
        
        # Run fault simulations
        print("\nSimulating fault conditions...")
        fault_results = interface.simulate_fault_conditions()
        
        # Save to database
        print("\nSaving simulation results to database...")
        record_id = interface.save_simulation_to_db(fault_results)
        print(f"Saved to database with ID: {record_id}")
        
        # Generate multiple simulation data points
        print("\nGenerating multiple simulation data points...")
        num_samples = 5  # Adjust as needed
        record_ids = interface.generate_simulation_data(num_samples)
        print(f"Generated {len(record_ids)} simulation samples")
        
        # Set up watch for real-time MATLAB data
        print("\nSetting up watch for real-time MATLAB data...")
        watch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
        interface.setup_matlab_data_watch(watch_dir)
        print(f"Watching directory: {watch_dir}")
        
        # Keep the script running
        print("\nPress Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
    else:
        print("MATLAB is not available. Please install MATLAB Engine for Python.")
        print("To install, run 'cd \"matlabroot\\extern\\engines\\python\" && python setup.py install' in MATLAB Command Window")
