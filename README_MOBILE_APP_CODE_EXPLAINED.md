# Solar Panel Monitoring Mobile App - Code Explained

This document explains the code structure and functionality of the Solar Panel Monitoring mobile app built with Flutter.

## Prerequisites

Before you begin working with the mobile app code, ensure you have:

1. **Flutter SDK** (version 2.17.0 or higher)
   - Follow the official Flutter installation guide: https://flutter.dev/docs/get-started/install
   - Verify installation with `flutter doctor` command

2. **Android Studio** or **Xcode** (for iOS development)
   - Android Studio (for Android development): https://developer.android.com/studio
   - Xcode (for iOS development): Available on Mac App Store

3. **Development Environment**
   - A code editor (VS Code recommended with Flutter extension)
   - Git for version control
   - USB cable for device testing (optional)

4. **Backend Server**
   - The Solar Panel Fault Detection System backend must be running
   - The backend uses MySQL for data storage
   - Default server address is http://127.0.0.1:8080

## Flutter Setup Guide

### Windows

1. **Download Flutter SDK**:
   - Download from: https://flutter.dev/docs/get-started/install/windows
   - Extract the ZIP file to a desired location (avoid paths with spaces)

2. **Add Flutter to PATH**:
   - Add the `flutter\bin` directory to your PATH environment variable
   - Open Command Prompt and run `flutter doctor` to verify

3. **Install Android Studio**:
   - Download and install Android Studio
   - Run Android Studio and go through the setup wizard
   - Install Flutter and Dart plugins:
     - Go to File > Settings > Plugins
     - Search for "Flutter" and install
     - The Dart plugin will be installed automatically

4. **Set up Android Emulator**:
   - In Android Studio, go to Tools > AVD Manager
   - Create a new virtual device
   - Choose a device definition and system image
   - Start the emulator

### macOS

1. **Download Flutter SDK**:
   - Download from: https://flutter.dev/docs/get-started/install/macos
   - Extract to a desired location

2. **Add Flutter to PATH**:
   - Add the following to your `~/.zshrc` or `~/.bash_profile`:
     ```bash
     export PATH="$PATH:[PATH_TO_FLUTTER_DIRECTORY]/flutter/bin"
     ```
   - Run `source ~/.zshrc` or `source ~/.bash_profile`
   - Run `flutter doctor` to verify

3. **Install Xcode** (for iOS development):
   - Install Xcode from the Mac App Store
   - Install the Xcode command-line tools:
     ```bash
     sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
     sudo xcodebuild -runFirstLaunch
     ```
   - Accept the license:
     ```bash
     sudo xcodebuild -license
     ```

4. **Install Android Studio** (for Android development):
   - Download and install Android Studio
   - Install Flutter and Dart plugins
   - Set up an Android emulator

### Linux

1. **Download Flutter SDK**:
   - Download from: https://flutter.dev/docs/get-started/install/linux
   - Extract to a desired location

2. **Add Flutter to PATH**:
   - Add the following to your `~/.bashrc`:
     ```bash
     export PATH="$PATH:[PATH_TO_FLUTTER_DIRECTORY]/flutter/bin"
     ```
   - Run `source ~/.bashrc`
   - Run `flutter doctor` to verify

3. **Install required dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y curl git unzip xz-utils zip libglu1-mesa
   ```

4. **Install Android Studio**:
   - Download and install Android Studio
   - Install Flutter and Dart plugins
   - Set up an Android emulator

## Project Structure

The mobile app code is located in the `solar_monitor_app` directory. Here's the structure:

```
solar_monitor_app/
├── lib/                    # Main source code
│   ├── main.dart           # Entry point
│   ├── screens/            # UI screens
│   ├── models/             # Data models
│   ├── services/           # API and backend services
│   ├── widgets/            # Reusable UI components
│   └── utils/              # Utility functions
├── assets/                 # Images, fonts, etc.
├── pubspec.yaml            # Dependencies and configuration
└── README.md               # Documentation
```

## Key Components Explained

### App Structure

Our app is organized like a tree with different folders:

```
solar_monitor_app/
├── lib/                  # This is where all our code lives
│   ├── main.dart         # The starting point of our app
│   ├── providers/        # Code that gets data from the server
│   ├── screens/          # Different pages in our app
│   ├── utils/            # Helper tools
│   └── widgets/          # Reusable pieces of the interface
├── pubspec.yaml          # List of ingredients our app needs
└── README.md             # Instructions for developers
```

### The Main App File

Let's look at `main.dart`, which is like the "on" button for our app:

```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final serverUrl = prefs.getString('server_url') ?? 'http://192.168.1.100:8080';
  
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (_) => SolarDataProvider(serverUrl: serverUrl),
        ),
      ],
      child: const SolarMonitorApp(),
    ),
  );
}
```

What this does:
1. `void main()` - This is where our app starts, like the first page of a book
2. `WidgetsFlutterBinding.ensureInitialized()` - Gets Flutter ready to run
3. `SharedPreferences.getInstance()` - Opens a small storage box on your phone
4. `prefs.getString('server_url')` - Looks for the server address in the storage box
5. `?? 'http://192.168.1.100:8080'` - If it can't find one, it uses this default address
6. `runApp(...)` - Starts the app with all the necessary pieces

### The Data Provider

The `SolarDataProvider` in `providers/solar_data_provider.dart` is like a messenger that talks to the server:

```dart
class SolarDataProvider with ChangeNotifier {
  String _serverUrl;
  io.Socket? _socket;
  bool _isConnected = false;
  bool _isMonitoring = false;
  
  // Real-time data
  List<SolarPanelData> _realtimeData = [];
  PredictionData? _currentPrediction;
  List<Alert> _alerts = [];
```

What this does:
1. `_serverUrl` - Remembers where to find the server
2. `_socket` - Creates a direct line to the server (like a phone call)
3. `_isConnected` - Keeps track of whether we're connected
4. `_isMonitoring` - Remembers if we're actively watching the panels
5. `_realtimeData` - Stores the latest readings from the solar panels
6. `_currentPrediction` - Stores information about any problems detected
7. `_alerts` - Keeps a list of important notifications

The provider has several important functions:

### Connecting to the Server

```dart
void _initSocket() {
  try {
    _socket = io.io(_serverUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': true,
    });
    
    _socket!.onConnect((_) {
      _isConnected = true;
      notifyListeners();
    });
```

This is like dialing a phone number to call the server. When the server answers, we mark ourselves as "connected" and tell the app to update the screen.

### Receiving Data

```dart
_socket!.on('data_update', (data) {
  final newData = SolarPanelData.fromJson(data);
  _realtimeData.add(newData);
  if (_realtimeData.length > 20) {
    _realtimeData.removeAt(0);
  }
  notifyListeners();
});
```

This is like listening for messages from the server. When new solar panel data arrives:
1. We convert it into a format our app understands
2. We add it to our list of data
3. If we have too many readings (more than 20), we remove the oldest one
4. We tell the app to update the screen with the new information

### Starting and Stopping Monitoring

```dart
Future<void> startMonitoring() async {
  try {
    final response = await http.post(
      Uri.parse('$_serverUrl/api/start'),
      headers: {'Content-Type': 'application/json'},
    );
    
    if (response.statusCode == 200) {
      _isMonitoring = true;
      notifyListeners();
    }
```

This is like pressing a button on the server to start collecting data. If the server says "OK" (status code 200), we mark ourselves as "monitoring" and update the screen.

### The Screens

Our app has three main screens:

### Home Screen

The `HomeScreen` in `screens/home_screen.dart` is what you see when you open the app:

```dart
Widget build(BuildContext context) {
  return Scaffold(
    appBar: AppBar(
      title: const Text('Solar Panel Monitor'),
      actions: [
        IconButton(
          icon: const Icon(Icons.settings),
          onPressed: () => Navigator.pushNamed(context, '/settings'),
        ),
      ],
    ),
    body: Consumer<SolarDataProvider>(
      builder: (context, provider, child) {
        return RefreshIndicator(
          onRefresh: () async {
            // Check database connection on refresh
            _databaseConnected = await provider.checkDatabaseConnection();
            setState(() {});
            
            await provider.fetchLatestData();
            await provider.fetchAlerts();
          },
```

This creates the main page with:
1. A title bar with "Solar Panel Monitor"
2. A settings button in the top-right corner
3. A pull-to-refresh feature that checks the database connection and gets the latest data
4. Various sections for showing different types of information

The home screen has several sections:

#### Connection Status

```dart
Widget _buildConnectionStatus(SolarDataProvider provider) {
  return Container(
    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
    decoration: BoxDecoration(
      color: provider.isConnected ? Colors.green.shade100 : Colors.red.shade100,
      borderRadius: BorderRadius.circular(8),
    ),
    child: Row(
      children: [
        Icon(
          provider.isConnected ? Icons.check_circle : Icons.error,
          color: provider.isConnected ? Colors.green.shade800 : Colors.red.shade800,
        ),
```

This shows whether we're connected to the server:
1. If connected, it shows a green background with a checkmark
2. If disconnected, it shows a red background with an error icon

#### Current Status

```dart
Widget _buildCurrentStatus(SolarDataProvider provider) {
  final latestData = provider.realtimeData.isNotEmpty
      ? provider.realtimeData.last
      : null;

  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      const Text(
        'Current Status',
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
```

This shows the most recent readings from the solar panels:
1. It looks for the latest data point
2. If there is data, it shows voltage, current, power, temperature, and irradiance
3. If there's no data, it shows "No data available"

#### Realtime Chart

```dart
Widget _buildRealtimeChart(SolarDataProvider provider) {
  if (provider.realtimeData.isEmpty) {
    return const SizedBox.shrink();
  }

  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      const Text(
        'Realtime Monitoring',
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
```

This creates a chart that shows how voltage, current, and power change over time:
1. It only appears if we have data
2. It uses different colors for each line (blue for voltage, amber for current, red for power)
3. It scales the values so they all fit on the same chart

### Settings Screen

The `SettingsScreen` in `screens/settings_screen.dart` lets you configure the app:

```dart
Widget build(BuildContext context) {
  return Scaffold(
    appBar: AppBar(
      title: const Text('Settings'),
    ),
    body: SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Connection Settings',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
```

This creates a page with several settings sections:
1. Connection Settings - where to find the server
2. MySQL Database Settings - username, password, and database name
3. App Settings - dark mode, notifications, and update interval

### The Widgets

We have special reusable components called "widgets" that appear in multiple places:

### Status Card

The `StatusCard` in `widgets/status_card.dart` shows a single reading:

```dart
class StatusCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;
  final Color color;

  const StatusCard({
    Key? key,
    required this.title,
    required this.value,
    required this.icon,
    required this.color,
  }) : super(key: key);
```

This creates a small card that shows:
1. A title (like "Voltage" or "Temperature")
2. A value (like "24.5 V" or "42.1 °C")
3. An icon (like a lightning bolt for voltage)
4. A color to make it visually distinct

### Alert Card

The `AlertCard` in `widgets/alert_card.dart` shows a notification:

```dart
class AlertCard extends StatelessWidget {
  final Alert alert;
  final VoidCallback onAcknowledge;

  const AlertCard({
    Key? key,
    required this.alert,
    required this.onAcknowledge,
  }) : super(key: key);
```

This creates a card that shows:
1. The alert message
2. When it happened
3. How serious it is (color-coded)
4. A button to mark it as "acknowledged"

## How It All Works Together

Here's how all these pieces work together:

1. When you open the app, `main.dart` starts everything up
2. The `SolarDataProvider` connects to the server and starts listening for data
3. The `HomeScreen` shows the latest information from the provider
4. When new data arrives, the provider tells the screen to update
5. When you tap the settings icon, the app shows the `SettingsScreen`
6. When you change settings, they're saved to your phone and used the next time you open the app

It's like a well-organized team where each member has a specific job, and they all communicate to make sure you see the latest information about your solar panels!

## MySQL Integration

Our app has been updated to work with MySQL databases instead of SQLite. Here's what changed:

1. We added database connection settings (username, password, database name)
2. We updated the data models to match the MySQL table structure
3. We added a database connection status indicator
4. We added a test connection button to verify database settings

These changes make the app more powerful and able to work with larger amounts of data!
