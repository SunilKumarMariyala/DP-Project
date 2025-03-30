"""
Display Solar Fault Detection Prediction Results

This script queries the database for the latest prediction results
and displays them in a formatted table with detailed information.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

def get_latest_predictions(limit=20):
    """
    Get the latest prediction results from the database
    
    Args:
        limit: Number of results to retrieve (default: 20)
        
    Returns:
        DataFrame with prediction results
    """
    try:
        # Connect to database
        conn = sqlite3.connect('solar_panel.db')
        
        # Query for latest predictions
        query = """
        SELECT 
            id, 
            timestamp, 
            pv_current, 
            pv_voltage, 
            pv_fault_1_current,
            pv_fault_1_voltage,
            pv_fault_2_current,
            pv_fault_2_voltage,
            pv_fault_3_current,
            pv_fault_3_voltage,
            pv_fault_4_current,
            pv_fault_4_voltage,
            prediction, 
            confidence, 
            description,
            processed_at
        FROM 
            solar_panel_data 
        WHERE 
            processed_at IS NOT NULL 
        ORDER BY 
            id DESC 
        LIMIT ?
        """
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=(limit,))
        
        # Close connection
        conn.close()
        
        return df
    
    except Exception as e:
        print(f"Error retrieving prediction results: {e}")
        return pd.DataFrame()

def calculate_feature_importance(df):
    """
    Calculate feature importance based on the prediction results
    
    Args:
        df: DataFrame with prediction results
        
    Returns:
        DataFrame with feature importance metrics
    """
    # Calculate deviation features
    df['i_deviation'] = (df['pv_current'] - df['pv_fault_1_current']) / df['pv_fault_1_current'] * 100
    df['v_deviation'] = (df['pv_voltage'] - df['pv_fault_1_voltage']) / df['pv_fault_1_voltage'] * 100
    
    # Calculate power values
    df['pv_power'] = df['pv_current'] * df['pv_voltage']
    df['fault_1_power'] = df['pv_fault_1_current'] * df['pv_fault_1_voltage']
    df['power_deviation'] = (df['pv_power'] - df['fault_1_power']) / df['fault_1_power'] * 100
    
    # Group by prediction and calculate mean feature values
    feature_importance = df.groupby('prediction').agg({
        'i_deviation': 'mean',
        'v_deviation': 'mean',
        'power_deviation': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    return feature_importance

def print_prediction_summary(df):
    """
    Print a summary of prediction results
    
    Args:
        df: DataFrame with prediction results
    """
    # Count predictions by type
    prediction_counts = df['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']
    
    # Calculate average confidence by prediction type
    confidence_by_type = df.groupby('prediction')['confidence'].mean().reset_index()
    confidence_by_type.columns = ['Prediction', 'Avg Confidence']
    
    # Merge counts and confidence
    summary = pd.merge(prediction_counts, confidence_by_type, on='Prediction')
    
    # Print summary
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Total predictions: {len(df)}")
    print("\nPrediction Counts and Confidence:")
    for _, row in summary.iterrows():
        print(f"  {row['Prediction']}: {row['Count']} predictions with {row['Avg Confidence']:.2f}% average confidence")

def print_detailed_results(df):
    """
    Print detailed prediction results
    
    Args:
        df: DataFrame with prediction results
    """
    print("\n=== DETAILED PREDICTION RESULTS ===")
    
    # Format DataFrame for display
    display_df = df[['id', 'timestamp', 'pv_current', 'pv_voltage', 'prediction', 'confidence', 'description']]
    display_df = display_df.rename(columns={
        'id': 'ID',
        'timestamp': 'Timestamp',
        'pv_current': 'PV Current (A)',
        'pv_voltage': 'PV Voltage (V)',
        'prediction': 'Prediction',
        'confidence': 'Confidence (%)',
        'description': 'Description'
    })
    
    # Format numeric columns
    display_df['PV Current (A)'] = display_df['PV Current (A)'].round(2)
    display_df['PV Voltage (V)'] = display_df['PV Voltage (V)'].round(2)
    display_df['Confidence (%)'] = display_df['Confidence (%)'].round(2)
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_colwidth', 50)
    print(display_df)

def print_feature_importance(df):
    """
    Print feature importance analysis
    
    Args:
        df: DataFrame with prediction results
    """
    # Calculate feature importance
    feature_importance = calculate_feature_importance(df)
    
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    print("Average deviation values by prediction type:")
    
    # Format DataFrame for display
    display_df = feature_importance.rename(columns={
        'prediction': 'Prediction',
        'i_deviation': 'Current Deviation (%)',
        'v_deviation': 'Voltage Deviation (%)',
        'power_deviation': 'Power Deviation (%)',
        'confidence': 'Confidence (%)'
    })
    
    # Format numeric columns
    display_df['Current Deviation (%)'] = display_df['Current Deviation (%)'].round(2)
    display_df['Voltage Deviation (%)'] = display_df['Voltage Deviation (%)'].round(2)
    display_df['Power Deviation (%)'] = display_df['Power Deviation (%)'].round(2)
    display_df['Confidence (%)'] = display_df['Confidence (%)'].round(2)
    
    # Print results
    print(display_df)

def main():
    """Main function to display prediction results"""
    print("=" * 80)
    print("SOLAR FAULT DETECTION - PREDICTION RESULTS")
    print("=" * 80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get latest predictions
    df = get_latest_predictions(limit=20)
    
    if df.empty:
        print("\nNo prediction results found in the database.")
        return
    
    # Print prediction summary
    print_prediction_summary(df)
    
    # Print detailed results
    print_detailed_results(df)
    
    # Print feature importance analysis
    print_feature_importance(df)
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

if __name__ == "__main__":
    main()
