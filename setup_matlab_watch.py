"""
Setup MATLAB Data Watch for Real-Time Predictions

This script sets up a watch directory for MATLAB-generated data files
and configures the system to process them for real-time predictions.
"""

import os
import sys
import requests
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matlab_watch.log')
    ]
)
logger = logging.getLogger('matlab_watch_setup')

def setup_matlab_watch(watch_dir=None, file_pattern='*.csv', check_interval=2):
    """
    Set up the MATLAB data watch directory and start the prediction process
    
    Args:
        watch_dir: Directory to watch for MATLAB data files (default: matlab_data in current directory)
        file_pattern: Pattern to match files (default: *.csv)
        check_interval: Interval in seconds to check for new files (default: 2)
    """
    # Set default watch directory if not provided
    if watch_dir is None:
        watch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
    
    # Create the watch directory if it doesn't exist
    if not os.path.exists(watch_dir):
        os.makedirs(watch_dir)
        logger.info(f"Created MATLAB data watch directory: {watch_dir}")
    
    # Set up the MATLAB watch using the API
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
            logger.info(f"Watching directory: {result.get('watch_directory')}")
            logger.info(f"File pattern: {result.get('file_pattern')}")
            logger.info(f"Check interval: {result.get('check_interval')}s")
            return True
        else:
            logger.error(f"Failed to set up MATLAB watch: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error setting up MATLAB watch: {e}")
        return False

def start_prediction_process(interval=2, batch_size=5):
    """
    Start the continuous prediction process
    
    Args:
        interval: Interval in seconds between prediction cycles (default: 2)
        batch_size: Number of records to process in each cycle (default: 5)
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
            logger.info(f"Status: {result.get('status')}")
            logger.info(f"Interval: {interval}s, Batch size: {batch_size}")
            return True
        else:
            logger.error(f"Failed to start prediction process: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error starting prediction process: {e}")
        return False

def main():
    """Main function to parse arguments and set up the MATLAB watch"""
    parser = argparse.ArgumentParser(description='Setup MATLAB Data Watch for Real-Time Predictions')
    
    parser.add_argument('--watch-dir', type=str, default=None,
                        help='Directory to watch for MATLAB data files')
    parser.add_argument('--file-pattern', type=str, default='*.csv',
                        help='Pattern to match files (default: *.csv)')
    parser.add_argument('--check-interval', type=int, default=2,
                        help='Interval in seconds to check for new files (default: 2)')
    parser.add_argument('--prediction-interval', type=int, default=2,
                        help='Interval in seconds between prediction cycles (default: 2)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of records to process in each prediction cycle (default: 5)')
    
    args = parser.parse_args()
    
    # Setup the MATLAB watch
    watch_success = setup_matlab_watch(
        watch_dir=args.watch_dir,
        file_pattern=args.file_pattern,
        check_interval=args.check_interval
    )
    
    if watch_success:
        # Start the prediction process
        prediction_success = start_prediction_process(
            interval=args.prediction_interval,
            batch_size=args.batch_size
        )
        
        if prediction_success:
            logger.info("MATLAB real-time prediction setup completed successfully!")
            logger.info(f"Place your MATLAB data files in: {args.watch_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matlab_data')}")
            logger.info(f"Files matching pattern '{args.file_pattern}' will be processed automatically")
        else:
            logger.error("Failed to start prediction process")
    else:
        logger.error("Failed to set up MATLAB watch")

if __name__ == "__main__":
    main()
