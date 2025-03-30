import os
import pandas as pd
import numpy as np
import json
from matlab_interface import MatlabInterface, MatlabDataSimulator
from datetime import datetime

def process_specific_fault_file(fault_type):
    """
    Process a specific fault type file and print the prediction results
    
    Args:
        fault_type: String indicating the fault type (normal, fault_1, fault_2, etc.)
    """
    print(f"\nProcessing {fault_type} data...")
    print("-" * 40)
    
    # Initialize the MATLAB interface
    interface = MatlabInterface()
    
    # Get the data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab_data")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # Find the matching files
    matching_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and fault_type.lower() in f.lower()]
    
    if not matching_files:
        print(f"No {fault_type} files found.")
        return
    
    # Sort by creation time (most recent first)
    matching_files.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
    
    file_path = os.path.join(data_dir, matching_files[0])
    print(f"Processing file: {matching_files[0]}")
    
    # Determine the expected fault type
    expected_fault = "Unknown"
    if "normal" in fault_type.lower():
        expected_fault = "Healthy"
    elif "fault_1" in fault_type.lower():
        expected_fault = "Fault_1"
    elif "fault_2" in fault_type.lower():
        expected_fault = "Fault_2"
    elif "fault_3" in fault_type.lower():
        expected_fault = "Fault_3"
    elif "fault_4" in fault_type.lower():
        expected_fault = "Fault_4"
    
    try:
        # Process the file
        result = interface.process_real_time_matlab_data(file_path)
        
        if result:
            print(f"Successfully processed {result['processed_count']} records")
            
            # Calculate accuracy
            if expected_fault != "Unknown":
                correct_predictions = sum(1 for pred in result['results'] if pred['prediction'] == expected_fault)
                accuracy = (correct_predictions / len(result['results'])) * 100 if result['results'] else 0
                print(f"Expected fault type: {expected_fault}")
                print(f"Prediction accuracy: {accuracy:.2f}%")
                
                # Print confusion matrix
                predictions = [pred['prediction'] for pred in result['results']]
                unique_predictions = set(predictions)
                print("\nPrediction Distribution:")
                for pred_type in unique_predictions:
                    count = predictions.count(pred_type)
                    percentage = (count / len(predictions)) * 100
                    print(f"  {pred_type}: {count} records ({percentage:.2f}%)")
            
            # Print sample predictions
            print("\nSample Predictions (first 3):")
            print("-" * 20)
            
            for i, pred in enumerate(result['results'][:3]):
                print(f"Record {i+1}:")
                print(f"  PV Current: {pred['pv_current']:.2f} A")
                print(f"  PV Voltage: {pred['pv_voltage']:.2f} V")
                print(f"  Prediction: {pred['prediction']}")
                print(f"  Confidence: {pred['confidence']:.2f}%")
                if 'description' in pred:
                    print(f"  Description: {pred['description']}")
                if 'recommended_action' in pred:
                    print(f"  Action: {pred['recommended_action']}")
                print("-" * 20)
            
            return {
                'expected': expected_fault,
                'accuracy': accuracy if expected_fault != "Unknown" else 0,
                'count': len(result['results']),
                'predictions': predictions
            }
        else:
            print("Failed to process file")
            return None
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def process_all_fault_types():
    """
    Process all fault types and show a summary of results
    """
    print("Solar Fault Detection System - MATLAB Fault Analysis")
    print("=" * 50)
    
    # Process each fault type
    fault_types = ["normal", "fault_1", "fault_2", "fault_3", "fault_4"]
    results = {}
    
    for fault_type in fault_types:
        result = process_specific_fault_file(fault_type)
        if result:
            results[fault_type] = result
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY OF FAULT DETECTION RESULTS")
    print("=" * 50)
    print(f"{'Fault Type':<15} {'Accuracy':<10} {'Sample Count':<15}")
    print("-" * 50)
    
    overall_correct = 0
    overall_total = 0
    
    for fault_type, result in results.items():
        print(f"{fault_type:<15} {result['accuracy']:.2f}%{' ':5} {result['count']:<15}")
        overall_correct += int((result['accuracy'] / 100) * result['count'])
        overall_total += result['count']
    
    # Calculate overall accuracy
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 50)
    print(f"{'Overall':<15} {overall_accuracy:.2f}%{' ':5} {overall_total:<15}")
    print("=" * 50)

if __name__ == "__main__":
    process_all_fault_types()
