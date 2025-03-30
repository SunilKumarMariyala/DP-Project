% MATLAB script to run the GridConnectedPVFarm model and save data to SQLite database
% This script should be run on the computer with MATLAB and the solar panel model

% Configuration
modelPath = 'C:\Users\Sunil Kumar\OneDrive\Documents\MATLAB\GridConnectedPVFarmExample\GridConnectedPVFarmExample';
modelFile = 'GridConnectedPVFarm.slx';
dbPath = '\\SHARED_LOCATION\solar_panel.db'; % Update this to your shared database location
samplingInterval = 5; % seconds between data samples

% Add model to path
addpath(modelPath);

% Check if model exists
if ~exist(fullfile(modelPath, modelFile), 'file')
    error('Model file not found: %s', fullfile(modelPath, modelFile));
end

% Load the model
disp('Loading model...');
load_system(fullfile(modelPath, modelFile));

% Setup SQLite connection
% Note: Requires Database Toolbox
try
    % Create connection to SQLite database
    conn = sqlite(dbPath, 'connect');
    disp('Connected to database');
catch e
    error('Error connecting to database: %s', e.message);
end

% Create table if it doesn't exist
try
    % Check if table exists
    tableExists = ~isempty(fetch(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='solar_panel_data'"));
    
    if ~tableExists
        % Create table
        exec(conn, ['CREATE TABLE solar_panel_data (' ...
                   'id INTEGER PRIMARY KEY AUTOINCREMENT, ' ...
                   'pv_current REAL, ' ...
                   'pv_voltage REAL, ' ...
                   'pv_power REAL, ' ...
                   'grid_power REAL, ' ...
                   'efficiency REAL, ' ...
                   'timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)']);
        disp('Created table: solar_panel_data');
    end
catch e
    error('Error creating table: %s', e.message);
end

% Main loop to collect data and save to database
disp('Starting data collection...');
disp('Press Ctrl+C to stop');

try
    while true
        % Set simulation parameters
        irradiance = 1000; % Default irradiance in W/m²
        temperature = 25;  % Default temperature in °C
        
        % You can modify these parameters based on real sensor data if available
        % For example:
        % irradiance = read_from_sensor('irradiance_sensor');
        % temperature = read_from_sensor('temperature_sensor');
        
        % Set model parameters
        set_param('GridConnectedPVFarm/PV Array', 'Irradiance', num2str(irradiance));
        set_param('GridConnectedPVFarm/PV Array', 'Temperature', num2str(temperature));
        
        % Run simulation
        disp('Running simulation...');
        sim('GridConnectedPVFarm');
        
        % Get simulation results
        % Note: These variable names must match the output signals in your model
        pv_current = PV_I.signals.values(end,:);
        pv_voltage = PV_V.signals.values(end,:);
        pv_power = PV_P.signals.values(end,:);
        grid_power = Grid_P.signals.values(end,:);
        
        % Calculate efficiency
        efficiency = mean(grid_power) / mean(pv_power);
        
        % Insert data into database
        insert(conn, 'solar_panel_data', {'pv_current', 'pv_voltage', 'pv_power', 'grid_power', 'efficiency'}, ...
               {mean(pv_current), mean(pv_voltage), mean(pv_power), mean(grid_power), efficiency});
        
        % Display current values
        fprintf('Data saved: PV Current=%.2fA, PV Voltage=%.2fV, Power=%.2fW, Efficiency=%.2f\n', ...
                mean(pv_current), mean(pv_voltage), mean(pv_power), efficiency);
        
        % Wait for next sample
        pause(samplingInterval);
    end
catch e
    if strcmp(e.identifier, 'MATLAB:quit')
        % User pressed Ctrl+C
        disp('Data collection stopped by user');
    else
        % Other error
        disp(['Error: ' e.message]);
    end
end

% Close database connection
close(conn);
disp('Database connection closed');

% Close model
close_system(modelFile, 0);
disp('Model closed');
