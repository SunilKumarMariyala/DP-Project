import pandas as pd
import numpy as np
import os
import sys
import logging
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create a base class for declarative class definitions
Base = declarative_base()

class SolarPanelData(Base):
    """
    SQLAlchemy model for solar panel data
    """
    __tablename__ = 'solar_panel_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    
    # Regular panel readings
    pv_current = Column(Float)
    pv_voltage = Column(Float)
    
    # Fault 1 readings
    pv_fault_1_current = Column(Float)
    pv_fault_1_voltage = Column(Float)
    
    # Fault 2 readings
    pv_fault_2_current = Column(Float)
    pv_fault_2_voltage = Column(Float)
    
    # Fault 3 readings
    pv_fault_3_current = Column(Float)
    pv_fault_3_voltage = Column(Float)
    
    # Fault 4 readings
    pv_fault_4_current = Column(Float)
    pv_fault_4_voltage = Column(Float)
    
    # Prediction result (if any)
    prediction = Column(String(20), nullable=True)
    prediction_label = Column(String(20), nullable=True)
    confidence = Column(Float, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    description = Column(String(200), nullable=True)
    recommended_action = Column(String(200), nullable=True)
    
    # MATLAB simulation data
    irradiance = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    pv_power = Column(Float, nullable=True)
    grid_power = Column(Float, nullable=True)
    efficiency = Column(Float, nullable=True)
    is_matlab_data = Column(Boolean, default=False)
    matlab_simulation_id = Column(String(50), nullable=True)
    simulation_id = Column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<SolarPanelData(id={self.id}, timestamp={self.timestamp}, prediction={self.prediction_label})>"

def setup_database(db_connection_str=None):
    """
    Setup the database for storing solar panel data
    
    Args:
        db_connection_str: SQLAlchemy connection string for the database
        
    Returns:
        Tuple of (engine, Session)
    """
    try:
        # Get database connection from environment variables if not provided
        if db_connection_str is None:
            DB_HOST = os.environ.get('DB_HOST', 'localhost')
            DB_USER = os.environ.get('DB_USER', 'solar_user')
            DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_secure_password')
            DB_NAME = os.environ.get('DB_NAME', 'solar_panel_db')
            db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
        
        # Create engine with connection pooling
        engine = create_engine(
            db_connection_str,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        
        # Create all tables if they don't exist
        Base.metadata.create_all(engine)
        
        # Create a session factory
        Session = sessionmaker(bind=engine)
        
        return engine, Session
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        raise

def load_sample_data_to_db(excel_path='final_dataset.xlsx', db_connection_str=None):
    """
    Load sample data from Excel to database
    """
    # Connect to database
    engine, Session = setup_database(db_connection_str)
    session = Session()
    
    try:
        # Load Excel data
        df = pd.read_excel(excel_path)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        
        # Fix inconsistent column names
        rename_dict = {}
        for col in df.columns:
            if 'current' in col.lower() and 'Current' not in col:
                rename_dict[col] = col.replace('current', 'Current')
            if 'voltage' in col.lower() and 'Voltage' not in col:
                rename_dict[col] = col.replace('voltage', 'Voltage')
        
        df = df.rename(columns=rename_dict)
        
        # Convert column names to lowercase for consistency with SQLAlchemy model
        df.columns = [col.lower() for col in df.columns]
        
        # Rename columns to match SQLAlchemy model
        column_mapping = {
            'pv_current': 'pv_current',
            'pv_voltage': 'pv_voltage',
            'pv_fault_1_current': 'pv_fault_1_current',
            'pv_fault_1_voltage': 'pv_fault_1_voltage',
            'pv_fault_2_current': 'pv_fault_2_current',
            'pv_fault_2_voltage': 'pv_fault_2_voltage',
            'pv_fault_3_current': 'pv_fault_3_current',
            'pv_fault_3_voltage': 'pv_fault_3_voltage',
            'pv_fault_4_current': 'pv_fault_4_current',
            'pv_fault_4_voltage': 'pv_fault_4_voltage'
        }
        
        # Ensure all required columns exist
        for db_col, excel_col in column_mapping.items():
            if excel_col not in df.columns:
                print(f"Warning: Column {excel_col} not found in Excel. Available columns: {df.columns.tolist()}")
        
        # Insert data into database
        print(f"Loading {len(df)} records into database...")
        
        # Use a batch approach to avoid memory issues with large datasets
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                data_entry = SolarPanelData(
                    pv_current=row.get('pv_current', None),
                    pv_voltage=row.get('pv_voltage', None),
                    pv_fault_1_current=row.get('pv_fault_1_current', None),
                    pv_fault_1_voltage=row.get('pv_fault_1_voltage', None),
                    pv_fault_2_current=row.get('pv_fault_2_current', None),
                    pv_fault_2_voltage=row.get('pv_fault_2_voltage', None),
                    pv_fault_3_current=row.get('pv_fault_3_current', None),
                    pv_fault_3_voltage=row.get('pv_fault_3_voltage', None),
                    pv_fault_4_current=row.get('pv_fault_4_current', None),
                    pv_fault_4_voltage=row.get('pv_fault_4_voltage', None)
                )
                session.add(data_entry)
            
            # Commit batch
            session.commit()
            print(f"Loaded batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        print("Sample data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        session.rollback()
    finally:
        session.close()

def get_latest_data(n=10, db_connection_str=None):
    """
    Get the latest n records from the database
    """
    # Connect to database
    engine, Session = setup_database(db_connection_str)
    session = Session()
    
    try:
        # Query latest records
        latest_data = session.query(SolarPanelData).order_by(SolarPanelData.id.desc()).limit(n).all()
        return latest_data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return []
    finally:
        session.close()

def get_matlab_simulation_data(n=10, db_connection_str=None):
    """
    Get the latest n records from MATLAB simulations
    """
    # Connect to database
    engine, Session = setup_database(db_connection_str)
    session = Session()
    
    try:
        # Query latest MATLAB simulation records
        latest_data = session.query(SolarPanelData).filter(
            SolarPanelData.is_matlab_data == True
        ).order_by(SolarPanelData.id.desc()).limit(n).all()
        return latest_data
    except Exception as e:
        print(f"Error retrieving MATLAB data: {e}")
        return []
    finally:
        session.close()

if __name__ == "__main__":
    # Setup database
    print("Setting up database...")
    engine, Session = setup_database()
    
    # Check if sample data should be loaded
    if len(sys.argv) > 1 and sys.argv[1] == '--load-sample-data':
        excel_path = sys.argv[2] if len(sys.argv) > 2 else 'final_dataset.xlsx'
        load_sample_data_to_db(excel_path)
