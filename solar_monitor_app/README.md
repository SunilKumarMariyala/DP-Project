# Solar Panel Monitor Mobile App

A cross-platform mobile application for monitoring the Solar Panel Fault Detection System in real-time. This app provides a convenient way to monitor your solar panels from anywhere using your smartphone.

## Features

- **Real-time Monitoring**: 
  - View live data from your solar panels including voltage, current, power, and temperature
  - Interactive charts that update in real-time
  - Color-coded indicators for quick status assessment
  - Automatic data refresh with configurable intervals

- **Fault Detection**: 
  - Receive instant notifications when faults are detected
  - Detailed fault information with confidence scores
  - Support for all fault types (Line-Line Fault, Open Circuit, Partial Shading, Degradation)
  - Historical fault tracking and analysis

- **Historical Data**: 
  - View historical performance data and trends
  - Filter data by date range and parameters
  - Export data for offline analysis
  - Comparative analysis between different time periods

- **Alerts Management**: 
  - View and acknowledge system alerts
  - Prioritized by severity (high, medium, low)
  - Detailed timestamp and message for each alert
  - Push notifications for critical alerts

- **Customizable Settings**: 
  - Configure server connection and notification preferences
  - Dark/light theme options
  - Adjustable update frequency
  - Personalized dashboard layout

## Prerequisites

To build and run this application, you need:

1. **Flutter SDK** (version 2.17.0 or higher)
   - Follow the official Flutter installation guide: https://flutter.dev/docs/get-started/install
   - Verify installation with `flutter doctor` command

2. **Android Studio** or **Xcode** (for iOS development)
   - Android Studio (for Android development): https://developer.android.com/studio
   - Xcode (for iOS development): Available on Mac App Store

3. **Running Solar Panel Fault Detection System** 
   - The backend server must be running and accessible from your mobile device
   - Default server address is http://192.168.1.100:8080
   - Both devices must be on the same network or the server must be accessible via the internet

4. **Development Environment**
   - A code editor (VS Code recommended with Flutter extension)
   - Git for version control
   - USB cable for device testing (optional)

## Setup Instructions

### 1. Install Flutter

1. **Download Flutter SDK**:
   - Windows: https://flutter.dev/docs/get-started/install/windows
   - macOS: https://flutter.dev/docs/get-started/install/macos
   - Linux: https://flutter.dev/docs/get-started/install/linux

2. **Add Flutter to your path**:
   - Windows: Add `flutter\bin` to your PATH environment variable
   - macOS/Linux: Add `export PATH="$PATH:[PATH_TO_FLUTTER_GIT_DIRECTORY]/flutter/bin"` to your shell profile

3. **Verify installation**:
   ```bash
   flutter doctor
   ```
   - Fix any issues reported by the doctor command

### 2. Clone the Repository

```bash
# Navigate to your desired directory
cd your/projects/folder

# Clone the repository
git clone https://github.com/yourusername/solar-panel-monitor.git

# Navigate to the project directory
cd solar-panel-monitor
```

### 3. Install Dependencies

```bash
# Install all required packages
flutter pub get
```

### 4. Configure Server Connection

The app is pre-configured to connect to `http://192.168.1.100:8080` by default. You can change this in several ways:

1. **In the app settings** (after installation)
   - Go to Settings screen
   - Enter your server URL in the "Server URL" field
   - Save changes

2. **During development** (before building)
   - Open `lib/main.dart`
   - Modify the default server URL:
     ```dart
     final serverUrl = prefs.getString('server_url') ?? 'http://your-server-ip:8080';
     ```

3. **At first login**
   - Enter your server URL in the login screen

### 5. Run the Application in Development Mode

```bash
# Connect your device or start an emulator
flutter devices

# Run the app on your selected device
flutter run -d [DEVICE_ID]
```

## Building for Production

### Android

1. **Generate a keystore** (if you don't have one):
   ```bash
   keytool -genkey -v -keystore ~/key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias key
   ```

2. **Create `android/key.properties` file**:
   ```
   storePassword=<password>
   keyPassword=<password>
   keyAlias=key
   storeFile=<path-to-keystore>
   ```

3. **Build the APK**:
   ```bash
   flutter build apk --release
   ```
   - The APK file will be located at `build/app/outputs/flutter-apk/app-release.apk`

4. **Build App Bundle** (for Google Play):
   ```bash
   flutter build appbundle --release
   ```
   - The bundle will be at `build/app/outputs/bundle/release/app-release.aab`

### iOS

1. **Update iOS deployment info**:
   - Open `ios/Runner.xcworkspace` in Xcode
   - Set your Team and Bundle Identifier

2. **Build for release**:
   ```bash
   flutter build ios --release
   ```

3. **Archive in Xcode**:
   - Open Xcode with the workspace
   - Select Product > Archive
   - Follow the distribution steps in the Archives organizer

## Detailed Usage Guide

### 1. Login Screen

- **Server Configuration**: Enter the URL of your Solar Panel Fault Detection System server
- **Authentication**: Enter your username and password (if configured on the server)
- **Demo Mode**: For testing, you can use the "Skip Login" option

### 2. Home Screen

- **Connection Status**: Shows if you're connected to the server
- **Current Metrics**: Displays current voltage, current, power, and temperature
- **Fault Status**: Shows the current fault prediction with confidence level
- **Real-time Chart**: Visualizes voltage and current trends
- **Recent Alerts**: Lists recent system alerts with severity indicators

### 3. Monitoring Controls

- **Start/Stop**: Use the floating action button to start or stop monitoring
  - Blue play button: Start monitoring
  - Red stop button: Stop monitoring
- **Manual Refresh**: Pull down to refresh data manually
- **Auto-refresh**: Data updates automatically based on your settings

### 4. Settings Screen

- **Server URL**: Configure the connection to your server
- **Dark Mode**: Toggle between light and dark themes
- **Notifications**: Enable/disable alert notifications
- **Update Interval**: Set how frequently the app fetches new data
- **About**: View app version and information

### 5. Alert Management

- **View Alerts**: Scroll through recent alerts on the home screen
- **Acknowledge**: Tap the "Acknowledge" button to mark alerts as seen
- **Filter**: Filter alerts by severity or time period
- **Actions**: Some alerts may have recommended actions to take

## Integration with the Solar Panel Fault Detection System

This mobile app connects to the same backend server as the web interface. It uses:

- **HTTP REST API** for data retrieval and control commands:
  - `/api/start` - Start monitoring
  - `/api/stop` - Stop monitoring
  - `/api/data/latest` - Get latest data
  - `/api/alerts/latest` - Get recent alerts

- **Socket.IO** for real-time data updates:
  - `data_update` event - Receive new sensor readings
  - `prediction_update` event - Receive new fault predictions
  - `alert` event - Receive new system alerts

The server must be running and accessible from your mobile device's network.

## Advanced Configuration

### Custom API Endpoints

If you've modified the backend API, you may need to update the endpoints in:
- `lib/providers/solar_data_provider.dart`

### Theming

To customize the app's appearance:
- Modify `lib/utils/app_theme.dart`

### Adding New Features

The app is structured to make adding new features straightforward:
- **Providers**: Add data models and API calls in the providers directory
- **Screens**: Create new screens in the screens directory
- **Widgets**: Add reusable components in the widgets directory

## Troubleshooting

### Connection Issues

If you cannot connect to the server:

1. **Verify server status**:
   ```bash
   # On the server machine
   curl http://localhost:8080/api/status
   ```

2. **Check network connectivity**:
   - Ensure both devices are on the same network
   - Try pinging the server from your mobile device
   - Check if any firewall is blocking the connection

3. **Server URL format**:
   - Must include protocol (http:// or https://)
   - Must include port number if not using standard ports
   - Example: `http://192.168.1.100:8080`

### App Crashes

If the app crashes:

1. **Check Flutter version**:
   ```bash
   flutter --version
   ```
   - Update if necessary: `flutter upgrade`

2. **Reinstall dependencies**:
   ```bash
   flutter clean
   flutter pub get
   ```

3. **Check logs**:
   - Connect device to computer
   - Run `flutter logs` while reproducing the issue

### Data Not Updating

If real-time data is not updating:

1. **Verify Socket.IO connection**:
   - Check the connection status indicator
   - Restart the app and server

2. **Check server logs**:
   - Look for Socket.IO connection errors
   - Verify that data is being emitted by the server

## Performance Optimization

For best performance:

1. **Limit chart data points** to improve rendering speed
2. **Use appropriate update intervals** (5-10 seconds recommended)
3. **Close the app** when not in use to conserve battery

## Security Considerations

1. **Secure your server** with HTTPS if accessible over the internet
2. **Implement proper authentication** if deploying in production
3. **Do not hardcode sensitive credentials** in the app code

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact and Support

For issues, feature requests, or contributions:
- Submit an issue on GitHub
- Contact the development team at support@example.com

## Acknowledgments

- Flutter team for the amazing cross-platform framework
- Socket.IO for real-time communication capabilities
- FL Chart for the interactive charting library
