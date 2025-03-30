"""
Detailed Test Cases for Solar Fault Detection Model

This script tests the solar fault detection model with various test cases
and provides detailed output about the prediction process.
"""

import pandas as pd
import numpy as np
import requests
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from realtime_prediction import SolarFaultDetector

# Initialize the detector
detector = SolarFaultDetector()

def run_test_case(pv_current, pv_voltage, description):
    """Run a test case and return detailed results"""
    print(f"\n=== Testing {description} ===")
    print(f"PV Current: {pv_current}, PV Voltage: {pv_voltage}")
    
    # Get prediction using the detector directly
    result = detector.predict_with_simple_inputs(pv_current, pv_voltage)
    
    # Print detailed results
    print(f"Prediction: {result['prediction_label']} (Class {result['prediction']})")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Generate synthetic data for this test case to show feature engineering
    synthetic_data = {
        'pv_current': pv_current,
        'pv_voltage': pv_voltage,
        'irradiance': pv_current * 1000 / (pv_voltage * 0.15) if pv_voltage > 0.1 else 0,
    }
    
    # Calculate key indicators
    v_nominal_closeness = 1.0 - min(1.0, abs(pv_voltage - 30) / 10)
    i_nominal_closeness = 1.0 - min(1.0, abs(pv_current - 8) / 4)
    healthy_indicator = v_nominal_closeness * i_nominal_closeness
    
    line_line_indicator = 0.0
    if pv_current > 8 and pv_voltage < 30:
        line_line_indicator = min(1.0, (pv_current - 8) / 4) * min(1.0, (30 - pv_voltage) / 10)
    
    open_circuit_indicator = 0.0
    if pv_current < 1.0 and pv_voltage > 32:
        open_circuit_indicator = min(1.0, (1.0 - pv_current)) * min(1.0, (pv_voltage - 30) / 10)
    
    partial_shading_indicator = 0.0
    if pv_current < 8 and pv_current > 2 and pv_voltage < 28 and pv_voltage > 24:
        partial_shading_indicator = min(1.0, (8 - pv_current) / 6) * min(1.0, (30 - pv_voltage) / 10)
    
    degradation_indicator = 0.0
    if pv_current < 8 and abs(pv_voltage - 30) < 3:
        degradation_indicator = min(1.0, (8 - pv_current) / 4) * (1.0 - min(1.0, abs(pv_voltage - 30) / 3))
    
    # Calculate derived features
    v_deviation = abs(pv_voltage - 30) / 30 if pv_voltage > 0 else 0
    i_deviation = abs(pv_current - 8) / 8 if pv_current > 0 else 0
    power = pv_current * pv_voltage
    nominal_power = 240
    power_deviation = min(5, abs(power - nominal_power) / nominal_power if nominal_power > 0 else 0)
    
    # Print feature engineering details
    print("\nFeature Engineering Details:")
    features_table = [
        ["Power", f"{power:.2f} W", f"{power/nominal_power*100:.1f}% of nominal"],
        ["Voltage Deviation", f"{v_deviation:.2f}", f"{v_deviation*100:.1f}%"],
        ["Current Deviation", f"{i_deviation:.2f}", f"{i_deviation*100:.1f}%"],
        ["Power Deviation", f"{power_deviation:.2f}", f"{power_deviation*100:.1f}%"],
        ["Healthy Indicator", f"{healthy_indicator:.2f}", "Higher is more likely healthy"],
        ["Line-Line Indicator", f"{line_line_indicator:.2f}", "Higher is more likely Line-Line Fault"],
        ["Open Circuit Indicator", f"{open_circuit_indicator:.2f}", "Higher is more likely Open Circuit"],
        ["Partial Shading Indicator", f"{partial_shading_indicator:.2f}", "Higher is more likely Partial Shading"],
        ["Degradation Indicator", f"{degradation_indicator:.2f}", "Higher is more likely Degradation"]
    ]
    
    print(tabulate(features_table, headers=["Feature", "Value", "Notes"], tablefmt="grid"))
    
    # Print decision factors
    print("\nDecision Factors:")
    indicators = {
        "Healthy": healthy_indicator,
        "Line-Line Fault": line_line_indicator,
        "Open Circuit": open_circuit_indicator,
        "Partial Shading": partial_shading_indicator,
        "Degradation": degradation_indicator
    }
    
    # Sort indicators by value in descending order
    sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
    decision_table = [(fault, f"{value:.2f}") for fault, value in sorted_indicators]
    
    print(tabulate(decision_table, headers=["Fault Type", "Indicator Value"], tablefmt="grid"))
    
    # Print recommended action
    print("\nRecommended Action:")
    print(result['recommended_action'])
    
    return result

def main():
    """Run a series of test cases"""
    print("Starting detailed test cases...\n")
    
    # Define test cases
    test_cases = [
        # Healthy cases
        {"pv_current": 8.0, "pv_voltage": 30.0, "description": "Normal Operation"},
        
        # Line-Line Fault cases
        {"pv_current": 12.0, "pv_voltage": 18.0, "description": "Severe Line-Line Fault"},
        
        # Open Circuit cases
        {"pv_current": 0.2, "pv_voltage": 38.0, "description": "Severe Open Circuit"},
        
        # Partial Shading cases
        {"pv_current": 4.8, "pv_voltage": 27.0, "description": "Moderate Partial Shading"},
        
        # Degradation cases
        {"pv_current": 6.4, "pv_voltage": 30.0, "description": "Moderate Degradation"},
    ]
    
    # Run all test cases
    results = []
    for case in test_cases:
        result = run_test_case(case["pv_current"], case["pv_voltage"], case["description"])
        results.append({
            "description": case["description"],
            "pv_current": case["pv_current"],
            "pv_voltage": case["pv_voltage"],
            "prediction": result["prediction_label"],
            "confidence": result["confidence"]
        })
    
    # Print summary table
    print("\n=== Test Results Summary ===")
    summary_table = []
    for result in results:
        summary_table.append([
            result["description"],
            result["pv_current"],
            result["pv_voltage"],
            result["prediction"],
            f"{result['confidence']:.2f}"
        ])
    
    print(tabulate(summary_table, 
                  headers=["Test Case", "Current", "Voltage", "Prediction", "Confidence"],
                  tablefmt="grid"))
    
    # Calculate prediction distribution
    prediction_counts = {}
    for result in results:
        label = result["prediction"]
        prediction_counts[label] = prediction_counts.get(label, 0) + 1
    
    print("\nPrediction Distribution:")
    for label, count in prediction_counts.items():
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
    
    # Create a visualization of the test cases
    plt.figure(figsize=(10, 8))
    
    # Create a color map for the different fault types
    fault_colors = {
        'Healthy': 'green',
        'Line-Line Fault': 'red',
        'Open Circuit Fault': 'orange',
        'Partial Shading': 'purple',
        'Degradation': 'blue'
    }
    
    # Extract data for plotting
    currents = [r["pv_current"] for r in results]
    voltages = [r["pv_voltage"] for r in results]
    predictions = [r["prediction"] for r in results]
    colors = [fault_colors.get(p, 'gray') for p in predictions]
    
    # Create the scatter plot
    plt.scatter(currents, voltages, c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, result in enumerate(results):
        plt.annotate(result["description"], 
                    (currents[i], voltages[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=fault)
                      for fault, color in fault_colors.items()]
    plt.legend(handles=legend_elements, title="Fault Types")
    
    # Set labels and title
    plt.xlabel('PV Current (A)')
    plt.ylabel('PV Voltage (V)')
    plt.title('Solar Panel Fault Detection - Test Cases')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('test_cases_visualization.png')
    print("\nVisualization saved as 'test_cases_visualization.png'")

if __name__ == "__main__":
    main()
