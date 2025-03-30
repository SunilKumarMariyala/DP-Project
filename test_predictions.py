import requests
import json
import time
from tabulate import tabulate

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_simple_prediction(pv_current, pv_voltage, description=""):
    """
    Test the simple prediction endpoint with given PV current and voltage
    """
    data = {
        'pv_current': pv_current,
        'pv_voltage': pv_voltage
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/simple_predict", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction_label']} (Class {result['prediction']})")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Description: {result['description'][:100]}...")
            print(f"Recommended Action: {result['recommended_action'][:100]}...")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
            return None
    except Exception as e:
        print(f"Exception: {str(e)}")
        return None

def run_test_suite():
    """
    Run a comprehensive test suite with various input scenarios
    """
    print("Starting prediction test suite...\n")
    
    # Store successful test results
    successful_tests = []
    
    # Test cases for different scenarios
    test_cases = [
        # Normal operation
        {"pv_current": 8.0, "pv_voltage": 30.0, "description": "Normal Operation"},
        {"pv_current": 7.5, "pv_voltage": 32.0, "description": "Slightly High Voltage"},
        {"pv_current": 8.2, "pv_voltage": 28.5, "description": "Slightly Low Voltage"},
        
        # Fault Type 1 (Line-Line Fault)
        {"pv_current": 10.5, "pv_voltage": 21.0, "description": "Potential Line-Line Fault"},
        {"pv_current": 12.0, "pv_voltage": 18.0, "description": "Severe Line-Line Fault"},
        
        # Fault Type 2 (Open Circuit)
        {"pv_current": 0.8, "pv_voltage": 36.0, "description": "Potential Open Circuit"},
        {"pv_current": 0.2, "pv_voltage": 38.0, "description": "Severe Open Circuit"},
        
        # Fault Type 3 (Partial Shading)
        {"pv_current": 4.8, "pv_voltage": 27.0, "description": "Partial Shading"},
        {"pv_current": 3.5, "pv_voltage": 26.0, "description": "Severe Partial Shading"},
        
        # Fault Type 4 (Degradation)
        {"pv_current": 6.4, "pv_voltage": 30.0, "description": "Panel Degradation"},
        {"pv_current": 5.0, "pv_voltage": 29.5, "description": "Severe Panel Degradation"},
        
        # Edge cases
        {"pv_current": 0.0, "pv_voltage": 40.0, "description": "Zero Current, High Voltage"},
        {"pv_current": 15.0, "pv_voltage": 0.1, "description": "High Current, Near Zero Voltage"},
        {"pv_current": 20.0, "pv_voltage": 40.0, "description": "Extremely High Values"},
        {"pv_current": 0.1, "pv_voltage": 0.1, "description": "Extremely Low Values"}
    ]
    
    # Run each test case
    for test_case in test_cases:
        print(f"\n=== Testing {test_case['description']} ===")
        print(f"PV Current: {test_case['pv_current']}, PV Voltage: {test_case['pv_voltage']}")
        
        result = test_simple_prediction(
            test_case['pv_current'], 
            test_case['pv_voltage'], 
            test_case['description']
        )
        
        if result:
            # Add test case info to result
            result['test_description'] = test_case['description']
            result['pv_current'] = test_case['pv_current']
            result['pv_voltage'] = test_case['pv_voltage']
            successful_tests.append(result)
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Print summary of results
    print("\n\n=== Test Results Summary ===")
    
    if successful_tests:
        print(f"Completed {len(successful_tests)} successful tests")
        
        # Create a table of results
        table_data = []
        for result in successful_tests:
            table_data.append([
                result['test_description'],
                f"{result['pv_current']:.1f}",
                f"{result['pv_voltage']:.1f}",
                result['prediction_label'],
                f"{result['confidence']:.2f}"
            ])
        
        # Print table
        headers = ["Test Case", "Current", "Voltage", "Prediction", "Confidence"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Count predictions by type
        prediction_counts = {}
        for result in successful_tests:
            label = result['prediction_label']
            prediction_counts[label] = prediction_counts.get(label, 0) + 1
        
        print("\nPrediction Distribution:")
        for label, count in prediction_counts.items():
            print(f"  {label}: {count} ({count/len(successful_tests)*100:.1f}%)")
    else:
        print("No successful test results to display")

if __name__ == "__main__":
    run_test_suite()
