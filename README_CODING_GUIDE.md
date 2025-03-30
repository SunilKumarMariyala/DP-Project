# Solar Panel Fault Detection - Complete Coding Guide

This guide explains every programming concept used in this project, from the most basic to advanced. If you're new to coding, start from the beginning. If you have some experience, you can jump to the sections that interest you.

## Table of Contents
1. [Basic Programming Concepts](#1-basic-programming-concepts)
2. [Python Fundamentals](#2-python-fundamentals)
3. [Data Handling](#3-data-handling)
4. [Database Systems](#4-database-systems)
5. [Web Development](#5-web-development)
6. [Machine Learning](#6-machine-learning)
7. [MATLAB Integration](#7-matlab-integration)
8. [Project-Specific Concepts](#8-project-specific-concepts)

---

## 1. Basic Programming Concepts

### What is Programming?
Programming is giving instructions to a computer to perform specific tasks. Think of it like writing a recipe - you're telling the computer exactly what to do, step by step.

### Variables
Variables are like labeled containers that store information. For example:
```python
pv_current = 8.0  # This creates a container labeled "pv_current" and puts the value 8.0 in it
```

When we write `pv_current = 8.0`, we're saying "create a container called pv_current and put the value 8.0 in it."

### Data Types
Different kinds of information are stored as different "types":
- **Integers (int)**: Whole numbers like 1, 42, -7
- **Floating-point (float)**: Numbers with decimal points like 3.14, 0.5
- **Strings (str)**: Text, written inside quotes like "Hello" or 'world'
- **Booleans (bool)**: True or False values
- **Lists**: Collections of items in order, like [1, 2, 3]
- **Dictionaries**: Collections of key-value pairs, like {"name": "John", "age": 30}

### Functions
Functions are reusable blocks of code that perform specific tasks. They're like mini-programs within your program:

```python
def calculate_power(current, voltage):
    """Calculate electrical power"""
    return current * voltage

# Using the function
power = calculate_power(8.0, 30.0)  # power will be 240.0
```

The parts of a function:
- `def`: Tells Python we're defining a function
- `calculate_power`: The name of our function
- `(current, voltage)`: The inputs (parameters) our function needs
- `return current * voltage`: What the function gives back (returns)

### Control Flow
Control flow determines which parts of your code run based on certain conditions:

#### If-Else Statements
```python
if pv_current < 0.5:
    print("Low current detected!")
elif pv_current > 10.0:
    print("High current detected!")
else:
    print("Current is normal")
```

#### Loops
Loops repeat code multiple times:

```python
# For loop - repeat for each item in a collection
for reading in solar_readings:
    process_reading(reading)

# While loop - repeat as long as a condition is true
while monitoring_active:
    take_reading()
    time.sleep(1)  # Wait 1 second
```

## 2. Python Fundamentals

### Importing Libraries
Libraries are collections of pre-written code that we can use. We import them like this:

```python
import pandas as pd  # Import pandas and nickname it 'pd'
from flask import Flask  # Import just the Flask part from the flask library
```

### Classes and Objects
Classes are like blueprints for creating objects. Objects are things that have properties (attributes) and can do things (methods):

```python
class SolarPanel:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.is_active = True
    
    def generate_power(self, sunlight_intensity):
        if not self.is_active:
            return 0
        return self.capacity * sunlight_intensity * 0.85  # 85% efficiency

# Create a solar panel object
my_panel = SolarPanel(id="Panel-A", capacity=300)  # 300W panel
power_output = my_panel.generate_power(sunlight_intensity=0.7)
```

## 3. Data Handling

### Pandas for Data Analysis
We use a library called Pandas to work with data in tables (DataFrames):

```python
import pandas as pd

# Create a DataFrame
data = {
    'timestamp': ['2023-01-01 10:00', '2023-01-01 10:05'],
    'pv_current': [5.2, 5.4],
    'pv_voltage': [30.1, 29.8]
}
df = pd.DataFrame(data)

# Calculate power
df['pv_power'] = df['pv_current'] * df['pv_voltage']

# Filter data
high_power = df[df['pv_power'] > 150]
```

### NumPy for Numerical Operations
NumPy helps us work with numbers efficiently:

```python
import numpy as np

# Create an array of values
currents = np.array([5.2, 5.4, 5.1, 5.3, 5.0])

# Calculate statistics
average_current = np.mean(currents)  # 5.2
max_current = np.max(currents)  # 5.4
```

## 4. Database Systems

### What is a Database?
A database is like a super-organized filing cabinet for your data. It stores information in a structured way and makes it easy to find and update.

### MySQL
Our project uses MySQL as its database system. Here's how we interact with it:

```python
import mysql.connector

# Connect to the database
cnx = mysql.connector.connect(
    user='username',
    password='password',
    host='127.0.0.1',
    database='solar_panel_db'
)

# Create a cursor object
cursor = cnx.cursor()

# Execute a query
query = ("SELECT * FROM solar_panel_data")
cursor.execute(query)

# Fetch the results
results = cursor.fetchall()

# Close the cursor and connection
cursor.close()
cnx.close()
```

### MySQL and Python
We use the `mysql-connector-python` library to connect to our MySQL database from Python.

### MySQL vs SQLite
Our project originally used SQLite but now uses MySQL. Here's the difference:

**SQLite**:
- Like a simple notebook - good for one person to use
- Stored in a single file on your computer
- Easy to set up, no separate server needed
- Not good for multiple users at once

**MySQL**:
- Like a professional filing system - good for many people to use
- Runs as a separate server program
- More powerful and faster with large amounts of data
- Better for web applications and multiple users

## 5. Web Development

### Flask Web Framework
We use Flask to create a web interface for our system:

```python
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    # Get the latest readings from the database
    latest_data = get_latest_readings()
    return jsonify(latest_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
```

### HTML, CSS, and JavaScript
- **HTML**: Creates the structure of web pages
- **CSS**: Makes the pages look nice
- **JavaScript**: Makes the pages interactive

## 6. Machine Learning

### What is Machine Learning?
Machine learning is teaching computers to learn patterns from data, rather than explicitly programming them with rules.

### Our Neural Network Model
We use a neural network to detect solar panel faults:

```python
import torch
import torch.nn as nn

class SolarFaultMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=5):
        super(SolarFaultMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
```

### How Our Model Works
1. We feed in features like current, voltage, power, and their deviations
2. The model processes these through its layers
3. It outputs a prediction (0-4) for the panel's condition:
   - 0: Healthy
   - 1: Line-Line Fault
   - 2: Open Circuit
   - 3: Partial Shading
   - 4: Panel Degradation

## 7. MATLAB Integration

### What is MATLAB?
MATLAB is a special software for scientific computing, especially good at simulating physical systems like solar panels.

### How We Connect to MATLAB
We use the MATLAB Engine for Python to connect our system with MATLAB:

```python
import matlab.engine

# Start MATLAB
eng = matlab.engine.start_matlab()

# Run a MATLAB function
result = eng.run_simulation(nargout=1)

# Convert MATLAB result to Python
pv_current = float(result['pv_current'])
pv_voltage = float(result['pv_voltage'])
```

### MySQL and MATLAB Integration
Our new MySQL database makes it easier to exchange data with MATLAB:

```python
# In MATLAB:
conn = database('solar_panel_db', 'username', 'password', ...
    'Vendor', 'MySQL', ...
    'Server', 'localhost');

data = select(conn, 'SELECT * FROM solar_panel_data LIMIT 10');
```

## 8. Project-Specific Concepts

### Solar Panel Basics
- **PV Current**: The flow of electricity from the panel (measured in Amperes)
- **PV Voltage**: The electrical pressure from the panel (measured in Volts)
- **PV Power**: How much energy the panel produces (measured in Watts)
  - Power = Current × Voltage

### Fault Detection Process
1. **Data Collection**: Get readings from sensors or MATLAB simulation
2. **Feature Engineering**: Calculate derived features like power and deviations
3. **Normalization**: Scale features to be in a similar range
4. **Prediction**: Run the data through our neural network
5. **Interpretation**: Convert the model's output to a human-readable result
6. **Storage**: Save the readings and predictions to the MySQL database
7. **Alerting**: Notify users if a fault is detected

### Common Test Cases
- **Healthy Panel**: PV Current: 8.0A, PV Voltage: 30.0V
  - Power: 240W, Prediction: Healthy (Class 0)
- **Line-Line Fault**: PV Current: 12.0A, PV Voltage: 15.0V
  - Power: 180W, Prediction: Line-Line Fault (Class 1)
- **Open Circuit**: PV Current: 0.0A, PV Voltage: 40.0V
  - Power: 0W, Prediction: Open Circuit (Class 2)
- **Partial Shading**: PV Current: 4.0A, PV Voltage: 28.0V
  - Power: 112W, Prediction: Partial Shading (Class 3)
- **Panel Degradation**: PV Current: 6.0A, PV Voltage: 25.0V
  - Power: 150W, Prediction: Panel Degradation (Class 4)

## Conclusion

This guide covered the main programming concepts used in our Solar Panel Fault Detection System. If you're interested in learning more:

1. Look at the actual code files to see these concepts in action
2. Try modifying small parts of the code to see what happens
3. Read the other README files for more specific information

Remember, programming is learned by doing - so don't be afraid to experiment!
