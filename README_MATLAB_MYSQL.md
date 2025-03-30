# MATLAB with MySQL Integration Guide

This guide explains how to set up and use MATLAB with MySQL for the Solar Fault Detection System.

## Prerequisites

- MySQL Server installed and configured (see README_MYSQL_SETUP.md)
- MATLAB R2019b or later with MATLAB Engine for Python
- Python 3.7+ with required packages

## Setting Up MATLAB Database Connection

### 1. Install MATLAB Database Toolbox

If not already installed, you'll need the MATLAB Database Toolbox:

1. Open MATLAB
2. Click on the "Add-Ons" button in the top toolbar
3. Search for "Database Toolbox"
4. Click "Install"

### 2. Install MySQL Connector for MATLAB

1. Download the MySQL Connector/J JDBC driver from [MySQL Downloads](https://dev.mysql.com/downloads/connector/j/)
2. Extract the .jar file to a location on your computer
3. In MATLAB, add the .jar file to the Java class path:
   ```matlab
   javaaddpath('path/to/mysql-connector-java-8.0.xx.jar');
   ```
4. To make this permanent, edit the `javaclasspath.txt` file:
   ```matlab
   edit(fullfile(prefdir,'javaclasspath.txt'))
   ```
   Add the full path to the .jar file and save

## Connecting to MySQL from MATLAB

### Direct MATLAB Connection

```matlab
% Create connection to database
conn = database('solar_panel_db', 'username', 'password', ...
    'Vendor', 'MySQL', ...
    'Server', 'localhost', ...
    'PortNumber', 3306);

% Test connection
if isconnection(conn)
    disp('Connected to MySQL database');
else
    disp('Failed to connect to MySQL database');
end

% Example query
data = select(conn, 'SELECT * FROM solar_panel_data LIMIT 10');

% Close connection when done
close(conn);
```

### Using MATLAB with Our Python Interface

Our system uses Python to interface between MATLAB and MySQL. The updated code now supports MySQL connections through SQLAlchemy.

## Configuration Steps

### 1. Update Database Configuration

Edit the database connection parameters in `solar_fault_detection.py`:

```python
# Database configuration
DB_HOST = 'localhost'  # Your MySQL server hostname
DB_USER = 'your_username'  # Your MySQL username
DB_PASSWORD = 'your_password'  # Your MySQL password
DB_NAME = 'solar_panel_db'  # Your database name
```

### 2. Running MATLAB Simulations with MySQL

The `MatlabInterface` class has been updated to work with MySQL. When initializing:

```python
from matlab_interface import MatlabInterface

# Initialize with MySQL connection string
interface = MatlabInterface(
    db_connection_str=f'mysql+pymysql://username:password@localhost/solar_panel_db'
)

# Run a simulation
result = interface.run_simulation(irradiance=800, temperature=30)
```

## Accessing MySQL Data from MATLAB Scripts

If you need to access the MySQL database directly from MATLAB scripts:

```matlab
% Create database connection
conn = database('solar_panel_db', 'username', 'password', ...
    'Vendor', 'MySQL', ...
    'Server', 'localhost');

% Query solar panel data
query = ['SELECT timestamp, pv_current, pv_voltage, temperature, irradiance ', ...
         'FROM solar_panel_data ', ...
         'ORDER BY timestamp DESC LIMIT 100'];
data = select(conn, query);

% Plot the data
figure;
plot(data.timestamp, data.pv_voltage, 'b-');
hold on;
plot(data.timestamp, data.pv_current, 'r-');
title('Solar Panel Voltage and Current');
legend('Voltage (V)', 'Current (A)');
xlabel('Time');
grid on;

% Close connection
close(conn);
```

## Exporting Data from MATLAB to MySQL

To save simulation results from MATLAB to MySQL:

```matlab
% Run simulation in MATLAB
% ... (your simulation code) ...

% Prepare results for database
timestamp = datetime('now');
voltage = 48.5;  % Example simulation result
current = 9.8;   % Example simulation result
power = voltage * current;
temperature = 28.3;
irradiance = 950;

% Create insert query
insertQuery = ['INSERT INTO solar_panel_data ', ...
               '(timestamp, pv_voltage, pv_current, pv_power, temperature, irradiance, is_matlab_data) ', ...
               'VALUES (''', char(timestamp), ''', ', ...
               num2str(voltage), ', ', ...
               num2str(current), ', ', ...
               num2str(power), ', ', ...
               num2str(temperature), ', ', ...
               num2str(irradiance), ', ', ...
               '1)'];

% Execute query
exec(conn, insertQuery);

% Close connection
close(conn);
```

## Troubleshooting

### Connection Issues

If you encounter connection issues from MATLAB:

1. Verify MySQL server is running
2. Check username and password
3. Ensure the database exists
4. Test connection with MySQL Workbench or command line client
5. Check firewall settings if connecting to a remote server

### JDBC Driver Issues

If MATLAB cannot find the JDBC driver:

1. Verify the .jar file is in the correct location
2. Restart MATLAB after adding to the Java path
3. Try using the absolute path to the .jar file

### Data Type Compatibility

MATLAB and MySQL have different data types. Use appropriate conversions:

- MATLAB datetime to MySQL DATETIME: Use `char(datetime)` in MATLAB
- MATLAB arrays to MySQL: Convert to string or individual values before inserting

## Example: Complete MATLAB Script for MySQL Integration

```matlab
function export_to_mysql(simulation_results)
    % Connect to MySQL database
    conn = database('solar_panel_db', 'username', 'password', ...
        'Vendor', 'MySQL', ...
        'Server', 'localhost');
    
    % Check connection
    if ~isconnection(conn)
        error('Failed to connect to MySQL database');
    end
    
    % Process simulation results
    timestamp = datetime('now');
    
    % Insert each row of simulation results
    for i = 1:size(simulation_results, 1)
        voltage = simulation_results(i, 1);
        current = simulation_results(i, 2);
        power = voltage * current;
        temperature = simulation_results(i, 3);
        irradiance = simulation_results(i, 4);
        
        % Create insert query
        insertQuery = ['INSERT INTO solar_panel_data ', ...
                       '(timestamp, pv_voltage, pv_current, pv_power, temperature, irradiance, is_matlab_data) ', ...
                       'VALUES (''', char(timestamp), ''', ', ...
                       num2str(voltage), ', ', ...
                       num2str(current), ', ', ...
                       num2str(power), ', ', ...
                       num2str(temperature), ', ', ...
                       num2str(irradiance), ', ', ...
                       '1)'];
        
        % Execute query
        exec(conn, insertQuery);
    end
    
    disp(['Exported ', num2str(size(simulation_results, 1)), ' rows to MySQL']);
    
    % Close connection
    close(conn);
end
```

## Further Resources

- [MATLAB Database Toolbox Documentation](https://www.mathworks.com/help/database/)
- [MySQL Connector/J Documentation](https://dev.mysql.com/doc/connector-j/en/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
