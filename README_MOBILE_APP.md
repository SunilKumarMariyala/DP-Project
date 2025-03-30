# Solar Panel Monitor Mobile App Guide

## What is the Solar Panel Monitor App?

The Solar Panel Monitor App is a mobile application that lets you check on your solar panels from your phone or tablet. It's like having a window into your solar energy system that you can carry in your pocket!

## What Can the App Do?

1. **Real-time Monitoring**: See what your solar panels are doing right now - voltage, current, power, temperature, and sunlight levels.

2. **Fault Detection**: The app can tell you if something might be wrong with your solar panels, just like a doctor can tell you if you're sick.

3. **Alerts**: Get notifications when something needs your attention, like when a panel isn't working properly.

4. **Historical Data**: Look back at how your panels performed in the past.

## Getting Started

### Installation

1. Make sure you have Flutter installed on your computer.
2. Download the app code from the `solar_monitor_app` folder.
3. Connect your phone or use an emulator.
4. Run `flutter pub get` to install all the necessary packages.
5. Run `flutter run` to start the app.

### Connecting to Your Solar Panel System

When you first open the app, you'll need to tell it where to find your solar panel system:

1. Tap on the **Settings** icon in the top right corner.
2. Enter the **Server URL** where your solar panel monitoring system is running.
   - This is usually something like `http://192.168.1.100:8080` if it's on your home network.
3. Expand the **MySQL Database Settings** section to configure database connection:
   - Enter your database username (default is "root")
   - Enter your database password
   - Enter your database name (default is "solar_panel_db")
4. Tap **Save Settings**.
5. You can tap **Test Connection** to make sure everything is working.

## Using the App

### Home Screen

The home screen shows you everything important at a glance:

- **Connection Status**: Shows if the app is connected to your server and database
- **Current Status**: Shows the latest readings from your solar panels
- **Realtime Chart**: Shows how voltage, current, and power are changing over time
- **Fault Detection**: Shows if any problems have been detected
- **Recent Alerts**: Shows recent warnings or notifications

### Starting and Stopping Monitoring

- Tap the **Play** button (bottom right) to start monitoring your solar panels.
- Tap the **Stop** button to pause monitoring.

### Checking Alerts

Alerts appear in the "Recent Alerts" section. Each alert shows:
- The alert message
- When it happened
- How serious it is (color-coded)
- A button to mark it as "acknowledged" (meaning you've seen it)

## Understanding the Data

### Solar Panel Readings

- **Voltage (V)**: How much electrical pressure your panels are producing (like water pressure in a pipe)
- **Current (A)**: How much electricity is flowing (like water flowing through a pipe)
- **Power (W)**: How much energy your panels are generating (voltage × current)
- **Temperature (°C)**: How hot your panels are
- **Irradiance (W/m²)**: How much sunlight is hitting your panels

### Fault Detection

The app uses machine learning to detect if something might be wrong with your solar panels. It will show:

- **Fault Type**: What kind of problem it might be
- **Confidence**: How sure the system is about the problem
- **Description**: What this problem means in simple terms
- **Recommended Action**: What you should do about it

## Troubleshooting

### App Won't Connect to Server

1. Check that your server is running
2. Make sure you entered the correct URL in settings
3. Check that your phone is on the same network as the server
4. Try tapping "Test Connection" in the settings

### No Data Appearing

1. Make sure monitoring is started (play button)
2. Check your database connection in settings
3. Verify that your solar panel system is generating data

### App Crashes or Freezes

1. Close and reopen the app
2. Check for app updates
3. Make sure your phone has enough free memory

## For Beginners: How the App Works

The app connects to your solar panel monitoring system through the internet. Think of it like this:

1. Your solar panels send information to a computer (server)
2. The computer stores this information in a filing cabinet (database)
3. The app on your phone asks the computer for information
4. The computer looks in the filing cabinet and sends back the information
5. The app shows this information to you in a nice, easy-to-understand way

The app uses something called "websockets" to get updates instantly, like a phone call rather than sending letters back and forth.

## For Parents and Teachers

This app is a great educational tool to teach children about:
- Renewable energy
- Electricity concepts (voltage, current, power)
- Data visualization
- Problem-solving when things go wrong

Consider using the app as a teaching aid to explain how solar energy works and how we can monitor technology to make sure it's working properly.
