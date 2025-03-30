"""
Comprehensive Test Analysis for Solar Fault Detection Model

This script provides a detailed analysis of how the solar fault detection model
predicts different fault types based on various input parameters.
"""

import pandas as pd
import numpy as np
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
from realtime_prediction import SolarFaultDetector

# Initialize the detector
detector = SolarFaultDetector()

def analyze_test_case(pv_current, pv_voltage, description, fault_type=None):
    """Run a test case and provide detailed analysis of the prediction process"""
    print(f"\n{'='*50}")
    print(f"TEST CASE: {description}")
    print(f"{'='*50}")
    print(f"Input Parameters:")
    print(f"  PV Current: {pv_current} A")
    print(f"  PV Voltage: {pv_voltage} V")
    print(f"  Expected Fault: {fault_type if fault_type else 'Not specified'}")
    
    # Get prediction using the detector
    result = detector.predict_with_simple_inputs(pv_current, pv_voltage)
    
    # Print result
    print(f"\nPREDICTION RESULT:")
    print(f"  Predicted Fault: {result['prediction_label']} (Class {result['prediction']})")
    print(f"  Confidence: {result['confidence']:.2f}")
    
    # Calculate key features
    power = pv_current * pv_voltage
    nominal_power = 240  # Nominal power at standard conditions
    v_deviation = abs(pv_voltage - 30) / 30 if pv_voltage > 0 else 0
    i_deviation = abs(pv_current - 8) / 8 if pv_current > 0 else 0
    power_deviation = min(5, abs(power - nominal_power) / nominal_power if nominal_power > 0 else 0)
    
    # Calculate class-specific indicators
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
    
    # Print key features
    print(f"\nKEY FEATURES:")
    features_table = [
        ["Power Output", f"{power:.2f} W", f"{power/nominal_power*100:.1f}% of nominal"],
        ["Voltage", f"{pv_voltage:.2f} V", f"{pv_voltage/30*100:.1f}% of nominal"],
        ["Current", f"{pv_current:.2f} A", f"{pv_current/8*100:.1f}% of nominal"],
        ["Voltage Deviation", f"{v_deviation:.2f}", f"{v_deviation*100:.1f}%"],
        ["Current Deviation", f"{i_deviation:.2f}", f"{i_deviation*100:.1f}%"],
        ["Power Deviation", f"{power_deviation:.2f}", f"{power_deviation*100:.1f}%"]
    ]
    print(tabulate(features_table, headers=["Feature", "Value", "Relative to Nominal"], tablefmt="grid"))
    
    # Print fault indicators
    print(f"\nFAULT INDICATORS:")
    indicators_table = [
        ["Healthy", f"{healthy_indicator:.2f}", "↑" if healthy_indicator > 0.5 else "↓"],
        ["Line-Line Fault", f"{line_line_indicator:.2f}", "↑" if line_line_indicator > 0.5 else "↓"],
        ["Open Circuit", f"{open_circuit_indicator:.2f}", "↑" if open_circuit_indicator > 0.5 else "↓"],
        ["Partial Shading", f"{partial_shading_indicator:.2f}", "↑" if partial_shading_indicator > 0.5 else "↓"],
        ["Degradation", f"{degradation_indicator:.2f}", "↑" if degradation_indicator > 0.5 else "↓"]
    ]
    print(tabulate(indicators_table, headers=["Fault Type", "Indicator Value", "Trend"], tablefmt="grid"))
    
    # Print decision explanation
    print(f"\nDECISION EXPLANATION:")
    
    # Get the highest indicator
    indicators = {
        "Healthy": healthy_indicator,
        "Line-Line Fault": line_line_indicator,
        "Open Circuit": open_circuit_indicator,
        "Partial Shading": partial_shading_indicator,
        "Degradation": degradation_indicator
    }
    
    highest_indicator = max(indicators.items(), key=lambda x: x[1])
    
    # Explain the decision based on the input parameters
    if result["prediction"] == 0:  # Healthy
        print(f"The panel is predicted to be HEALTHY because:")
        print(f"  - Current ({pv_current:.1f} A) is close to nominal value (8.0 A)")
        print(f"  - Voltage ({pv_voltage:.1f} V) is close to nominal value (30.0 V)")
        print(f"  - Power output ({power:.1f} W) is close to expected value (~240 W)")
        print(f"  - Low voltage deviation ({v_deviation*100:.1f}%) and current deviation ({i_deviation*100:.1f}%)")
        
    elif result["prediction"] == 1:  # Line-Line Fault
        print(f"The panel is predicted to have a LINE-LINE FAULT because:")
        print(f"  - Current ({pv_current:.1f} A) is higher than nominal (8.0 A)")
        print(f"  - Voltage ({pv_voltage:.1f} V) is lower than nominal (30.0 V)")
        print(f"  - This pattern indicates a short circuit condition")
        print(f"  - Line-Line indicator value is {line_line_indicator:.2f}")
        
    elif result["prediction"] == 2:  # Open Circuit
        print(f"The panel is predicted to have an OPEN CIRCUIT FAULT because:")
        print(f"  - Current ({pv_current:.1f} A) is much lower than nominal (8.0 A)")
        print(f"  - Voltage ({pv_voltage:.1f} V) is higher than nominal (30.0 V)")
        print(f"  - This pattern indicates a break in the electrical path")
        print(f"  - Open Circuit indicator value is {open_circuit_indicator:.2f}")
        
    elif result["prediction"] == 3:  # Partial Shading
        print(f"The panel is predicted to have PARTIAL SHADING because:")
        print(f"  - Current ({pv_current:.1f} A) is moderately reduced from nominal (8.0 A)")
        print(f"  - Voltage ({pv_voltage:.1f} V) is slightly reduced from nominal (30.0 V)")
        print(f"  - This pattern is typical of shading on a portion of the panel")
        print(f"  - Partial Shading indicator value is {partial_shading_indicator:.2f}")
        
    elif result["prediction"] == 4:  # Degradation
        print(f"The panel is predicted to have DEGRADATION because:")
        print(f"  - Current ({pv_current:.1f} A) is reduced from nominal (8.0 A)")
        print(f"  - Voltage ({pv_voltage:.1f} V) remains close to nominal (30.0 V)")
        print(f"  - This pattern indicates aging or material breakdown")
        print(f"  - Degradation indicator value is {degradation_indicator:.2f}")
    
    # Print recommended action
    print(f"\nRECOMMENDED ACTION:")
    print(f"  {result['recommended_action']}")
    
    return result

def main():
    """Run comprehensive test analysis"""
    print("SOLAR FAULT DETECTION - COMPREHENSIVE TEST ANALYSIS")
    print("="*60)
    
    # Define representative test cases for each fault type
    test_cases = [
        # Healthy case
        {
            "pv_current": 8.0, 
            "pv_voltage": 30.0, 
            "description": "Normal Operation", 
            "fault_type": "Healthy"
        },
        
        # Line-Line Fault case
        {
            "pv_current": 12.0, 
            "pv_voltage": 18.0, 
            "description": "Severe Line-Line Fault", 
            "fault_type": "Line-Line Fault"
        },
        
        # Open Circuit case
        {
            "pv_current": 0.2, 
            "pv_voltage": 38.0, 
            "description": "Severe Open Circuit", 
            "fault_type": "Open Circuit Fault"
        },
        
        # Partial Shading case
        {
            "pv_current": 4.8, 
            "pv_voltage": 27.0, 
            "description": "Moderate Partial Shading", 
            "fault_type": "Partial Shading"
        },
        
        # Degradation case
        {
            "pv_current": 6.4, 
            "pv_voltage": 30.0, 
            "description": "Moderate Degradation", 
            "fault_type": "Degradation"
        }
    ]
    
    # Run analysis for each test case
    results = []
    for case in test_cases:
        result = analyze_test_case(
            case["pv_current"], 
            case["pv_voltage"], 
            case["description"], 
            case["fault_type"]
        )
        results.append({
            "description": case["description"],
            "expected": case["fault_type"],
            "predicted": result["prediction_label"],
            "confidence": result["confidence"],
            "match": case["fault_type"] == result["prediction_label"]
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF TEST RESULTS")
    print("="*60)
    
    summary_table = []
    for result in results:
        summary_table.append([
            result["description"],
            result["expected"],
            result["predicted"],
            f"{result['confidence']:.2f}",
            "✓" if result["match"] else "✗"
        ])
    
    print(tabulate(summary_table, 
                  headers=["Test Case", "Expected", "Predicted", "Confidence", "Match"],
                  tablefmt="grid"))
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r["match"]) / len(results) * 100
    print(f"\nOverall accuracy on test cases: {accuracy:.1f}%")
    
    print("\nThis analysis demonstrates how the model uses current and voltage patterns")
    print("to identify different fault types in solar panels.")

if __name__ == "__main__":
    main()
