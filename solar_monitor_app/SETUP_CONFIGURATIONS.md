# Mobile App Setup Configurations

This guide explains how to configure the Solar Panel Monitor mobile app to work with different system setups.

## System Configuration Options

The Solar Panel Fault Detection System can be set up in two main configurations, and the mobile app works with both:

### 1. Single Computer Setup

In this configuration, everything runs on one machine:
- MATLAB model generates data
- Data is saved to the local database
- Python application reads from the same database and makes predictions
- Mobile app connects to the server running on this computer

#### Setup Instructions:

1. **Run the server on your computer**:
   ```bash
   python solar_fault_detection.py --host 0.0.0.0 --port 8080 --matlab
   ```
   Note: Using `0.0.0.0` instead of `127.0.0.1` makes the server accessible from other devices on the network.

2. **Find your computer's IP address**:
   - Windows: Open Command Prompt and type `ipconfig`
   - macOS/Linux: Open Terminal and type `ifconfig` or `ip addr`
   - Look for the IPv4 address (e.g., 192.168.1.100)

3. **Configure the mobile app**:
   - Enter your computer's IP address in the app's server URL field
   - Example: `http://192.168.1.100:8080`
   - Make sure your mobile device is on the same Wi-Fi network as your computer

### 2. Multi-Computer Setup

In this configuration:
- MATLAB model runs on one computer and generates data
- Data is saved to a shared database (accessible by both computers)
- Python prediction model runs on another computer
- Mobile app connects to the server running on the second computer

The system supports three approaches for multi-computer setup:

#### a) Shared Database Approach (Recommended)

1. **Configure the server**:
   - Set up the shared database accessible by both computers
   - Run the server on the second computer:
     ```bash
     python solar_fault_detection.py --host 0.0.0.0 --port 8080 --db-path /path/to/shared/database/solar_panel.db
     ```

2. **Configure the mobile app**:
   - Enter the second computer's IP address in the app's server URL field
   - Example: `http://192.168.1.101:8080`

#### b) Network Share Approach

1. **Configure the server**:
   - Set up a network share on the first computer
   - Run the server on the second computer pointing to the network share:
     ```bash
     python solar_fault_detection.py --host 0.0.0.0 --port 8080 --db-path \\server\share\solar_panel.db
     ```

2. **Configure the mobile app**:
   - Enter the second computer's IP address in the app's server URL field
   - Example: `http://192.168.1.101:8080`

#### c) REST API Approach

1. **Configure the first computer (MATLAB)**:
   - Run a simple API server on the first computer to expose the data
   - Example: `python matlab_api_server.py --host 0.0.0.0 --port 8081`

2. **Configure the second computer (Prediction)**:
   - Run the main server pointing to the first computer's API:
     ```bash
     python solar_fault_detection.py --host 0.0.0.0 --port 8080 --api-url http://192.168.1.100:8081
     ```

3. **Configure the mobile app**:
   - Enter the second computer's IP address in the app's server URL field
   - Example: `http://192.168.1.101:8080`

## Troubleshooting Network Connectivity

If you're having trouble connecting the mobile app to the server:

1. **Check firewall settings**:
   - Make sure port 8080 is open on the server computer
   - Temporarily disable the firewall to test if that's the issue

2. **Verify network connectivity**:
   - From your mobile device, try opening the server URL in a web browser
   - If the web interface loads, the network connection is working

3. **Check server logs**:
   - Look for connection attempts in the server logs
   - Make sure the server is binding to the correct IP address

4. **Test with localhost first**:
   - Before trying to connect from a mobile device, test the server locally
   - Open a browser on the server computer and navigate to `http://localhost:8080`

## Advanced Configuration

### Using Over the Internet

To access your solar panel monitoring system from anywhere (not just your local network):

1. **Set up port forwarding** on your router to forward port 8080 to your server computer
2. **Use a dynamic DNS service** if you don't have a static IP address
3. **Enable HTTPS** for secure communication
4. **Implement proper authentication** to secure your system

### Multiple Mobile Devices

The system supports multiple mobile devices connecting simultaneously:

- Each device can run the mobile app independently
- All devices will receive the same real-time updates
- No special configuration is needed beyond setting the correct server URL on each device
