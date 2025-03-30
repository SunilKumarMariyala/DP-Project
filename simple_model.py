import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimpleModel")

class SimpleSolarModel:
    """
    A simplified model for solar fault detection based on current and voltage values
    """
    def __init__(self):
        """Initialize the model with default parameters"""
        logger.info("Initializing SimpleSolarModel")
        
        # Define reference values for a healthy panel
        self.ref_current = 8.0  # Reference current in Amperes
        self.ref_voltage = 30.0  # Reference voltage in Volts
        
        # Define thresholds for fault detection
        self.current_threshold = 0.2  # 20% deviation
        self.voltage_threshold = 0.1  # 10% deviation
        
        # Define fault characteristics
        self.fault_characteristics = {
            "Healthy": {
                "current_range": (0.8, 1.2),  # 80-120% of reference
                "voltage_range": (0.9, 1.1),  # 90-110% of reference
                "description": "Solar panel is operating within normal parameters.",
                "recommended_action": "No action required. Continue regular monitoring."
            },
            "Fault_1": {  # Line-Line Fault
                "current_range": (1.2, 1.5),  # 120-150% of reference
                "voltage_range": (0.6, 0.9),  # 60-90% of reference
                "description": "Line-Line fault detected. Abnormal current with reduced voltage.",
                "recommended_action": "Inspect panel connections for short circuits. Check for damaged insulation."
            },
            "Fault_2": {  # Open Circuit
                "current_range": (0.0, 0.3),  # 0-30% of reference (including negative currents)
                "voltage_range": (1.1, 1.5),  # 110-150% of reference
                "description": "Open circuit fault detected. Very low current with elevated voltage.",
                "recommended_action": "Check panel connections for breaks. Inspect junction box and wiring."
            },
            "Fault_3": {  # Partial Shading
                "current_range": (0.5, 0.8),  # 50-80% of reference
                "voltage_range": (0.8, 1.0),  # 80-100% of reference
                "description": "Partial shading detected. Reduced current with slightly reduced voltage.",
                "recommended_action": "Check for objects casting shadows on panels. Consider cleaning if dust/debris is present."
            },
            "Fault_4": {  # Degradation
                "current_range": (0.6, 0.9),  # 60-90% of reference
                "voltage_range": (0.7, 0.9),  # 70-90% of reference
                "description": "Panel degradation detected. Gradual reduction in both current and voltage.",
                "recommended_action": "Monitor performance over time. Consider panel replacement if efficiency continues to decrease."
            }
        }
    
    def predict(self, pv_current, pv_voltage):
        """
        Predict the fault type based on current and voltage values
        
        Args:
            pv_current: PV current in Amperes
            pv_voltage: PV voltage in Volts
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Calculate normalized values
            norm_current = pv_current / self.ref_current
            norm_voltage = pv_voltage / self.ref_voltage
            
            # Handle negative current values (important for Fault_2 detection)
            if pv_current < 0:
                # Negative current is a strong indicator of Fault_2 (Open Circuit)
                return {
                    "prediction": "Fault_2",
                    "confidence": 0.95,
                    "description": self.fault_characteristics["Fault_2"]["description"],
                    "recommended_action": self.fault_characteristics["Fault_2"]["recommended_action"],
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Find the most likely fault type
            best_match = "Healthy"
            best_confidence = 0.0
            
            for fault_type, characteristics in self.fault_characteristics.items():
                # Check if values are within the characteristic ranges
                current_in_range = (
                    characteristics["current_range"][0] <= norm_current <= characteristics["current_range"][1]
                )
                voltage_in_range = (
                    characteristics["voltage_range"][0] <= norm_voltage <= characteristics["voltage_range"][1]
                )
                
                # Calculate confidence based on how close to the center of the range
                if current_in_range and voltage_in_range:
                    current_range = characteristics["current_range"]
                    voltage_range = characteristics["voltage_range"]
                    
                    # Calculate distance from center of range (normalized to 0-1)
                    current_center = (current_range[0] + current_range[1]) / 2
                    voltage_center = (voltage_range[0] + voltage_range[1]) / 2
                    
                    current_distance = 1 - min(abs(norm_current - current_center) / 
                                            (current_range[1] - current_range[0]), 1)
                    voltage_distance = 1 - min(abs(norm_voltage - voltage_center) / 
                                            (voltage_range[1] - voltage_range[0]), 1)
                    
                    # Combined confidence
                    confidence = (current_distance + voltage_distance) / 2
                    
                    if confidence > best_confidence:
                        best_match = fault_type
                        best_confidence = confidence
            
            # If confidence is too low, default to "Unknown"
            if best_confidence < 0.3:
                best_match = "Unknown"
                description = "Unable to determine fault type with confidence."
                recommended_action = "Perform manual inspection of the solar panel system."
                best_confidence = 0.0
            else:
                description = self.fault_characteristics[best_match]["description"]
                recommended_action = self.fault_characteristics[best_match]["recommended_action"]
            
            # Return prediction results
            return {
                "prediction": best_match,
                "confidence": best_confidence,
                "description": description,
                "recommended_action": recommended_action,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "description": f"Error making prediction: {str(e)}",
                "recommended_action": "Check input data and try again.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# For testing
if __name__ == "__main__":
    model = SimpleSolarModel()
    
    # Test with different scenarios
    test_cases = [
        {"current": 8.0, "voltage": 30.0, "expected": "Healthy"},  # Normal
        {"current": 10.0, "voltage": 25.0, "expected": "Fault_1"},  # Line-Line Fault
        {"current": 1.0, "voltage": 35.0, "expected": "Fault_2"},   # Open Circuit
        {"current": 5.0, "voltage": 28.0, "expected": "Fault_3"},   # Partial Shading
        {"current": 6.0, "voltage": 25.0, "expected": "Fault_4"}    # Degradation
    ]
    
    for i, test in enumerate(test_cases):
        result = model.predict(test["current"], test["voltage"])
        print(f"Test {i+1}: Current={test['current']}A, Voltage={test['voltage']}V")
        print(f"  Expected: {test['expected']}, Predicted: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Description: {result['description']}")
        print(f"  Action: {result['recommended_action']}")
        print("-" * 50)
