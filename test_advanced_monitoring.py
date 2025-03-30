"""
Test script for the advanced monitoring system
"""

import time
from advanced_monitoring import DataProcessor

def main():
    print("Initializing Data Processor...")
    processor = DataProcessor()
    
    print("Starting data processing...")
    processor.start_processing(interval=2, batch_size=5)
    
    print("Processing started. Waiting for 30 seconds...")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    print("Stopping data processing...")
    processor.stop_processing()
    
    print("Processing stopped.")

if __name__ == "__main__":
    main()
