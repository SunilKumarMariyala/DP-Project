# Solar Fault Detection System - Mobile App Development Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Development Environment Setup](#development-environment-setup)
5. [App Structure and Components](#app-structure-and-components)
6. [Backend Integration](#backend-integration)
7. [Real-time Data Visualization](#real-time-data-visualization)
8. [Fault Detection Alerts](#fault-detection-alerts)
9. [PDF Report Generation](#pdf-report-generation)
10. [Offline Mode](#offline-mode)
11. [Testing and Deployment](#testing-and-deployment)
12. [Troubleshooting](#troubleshooting)

## Introduction

The Solar Fault Detection System Mobile App provides real-time monitoring and alerts for solar panel installations. This guide explains how to develop the mobile application from scratch, covering all aspects from setup to deployment.

### Key Features

- **Real-time Monitoring**: View live voltage, current, power, temperature, and irradiance data
- **Fault Detection**: Receive instant alerts when the system detects potential faults
- **Historical Data**: Access and analyze past performance data with interactive charts
- **PDF Reports**: Generate detailed reports for sharing and analysis
- **Offline Mode**: Access critical information even without internet connectivity
- **Cross-platform**: Works on both Android and iOS devices

## System Architecture

The mobile app is part of a larger system:

```
Solar Panels → Data Collection Server → MySQL Database → API Server → Mobile App
```

- **Data Flow**: Sensor data flows from solar panels to the server, which processes and stores it in MySQL
- **API Server**: A Flask-based server exposes endpoints for the mobile app
- **Mobile App**: Built with Flutter for cross-platform compatibility

## Prerequisites

Before starting development, ensure you have:

- **Development Tools**:
  - Flutter SDK (latest stable version)
  - Android Studio or VS Code
  - Git for version control
  - A physical Android/iOS device or emulator

- **Accounts**:
  - Google Play Developer Account (for Android deployment)
  - Apple Developer Account (for iOS deployment)
  - Firebase account (for push notifications)

- **Backend**:
  - Running instance of the Solar Fault Detection System backend
  - MySQL database properly configured

## Development Environment Setup

### Installing Flutter

1. **Download Flutter SDK**:
   - Go to [flutter.dev](https://flutter.dev/docs/get-started/install)
   - Download the latest stable version for your operating system

2. **Extract the ZIP file**:
   - Extract to a location without spaces in the path (e.g., `C:\flutter` on Windows)
   - Avoid folders that require administrator privileges

3. **Add Flutter to your PATH**:
   
   **For Windows**:
   ```
   setx PATH "%PATH%;C:\flutter\bin"
   ```
   
   **For macOS/Linux**:
   ```
   export PATH="$PATH:/path/to/flutter/bin"
   ```

4. **Verify installation**:
   ```
   flutter doctor
   ```
   
5. **Fix any issues** reported by Flutter Doctor

### Setting Up Android Development

1. **Install Android Studio**:
   - Download from [developer.android.com](https://developer.android.com/studio)
   - Complete the installation wizard

2. **Install Flutter and Dart plugins**:
   - Open Android Studio
   - Go to File → Settings → Plugins
   - Search for "Flutter" and install
   - The Dart plugin will be installed automatically

3. **Configure Android SDK**:
   - In Android Studio, go to Tools → SDK Manager
   - Install Android SDK Platform-Tools
   - Install at least one Android SDK Platform (Android 11/API level 30 recommended)

4. **Set up Android emulator**:
   - Go to Tools → AVD Manager
   - Click "Create Virtual Device"
   - Select a device (e.g., Pixel 4) and click Next
   - Select a system image (e.g., Android 11) and click Next
   - Name your emulator and click Finish

### Setting Up iOS Development (Mac only)

1. **Install Xcode**:
   - Download from the Mac App Store
   - Open Xcode and accept the license agreement
   - Install additional components if prompted

2. **Configure Xcode command-line tools**:
   ```
   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
   sudo xcodebuild -runFirstLaunch
   ```

3. **Install CocoaPods**:
   ```
   sudo gem install cocoapods
   ```

4. **Set up iOS simulator**:
   - Open Xcode
   - Go to Xcode → Preferences → Components
   - Select a simulator to download

## App Structure and Components

### Creating a New Flutter Project

1. **Create project**:
   ```
   flutter create solar_fault_detection_app
   ```

2. **Navigate to project directory**:
   ```
   cd solar_fault_detection_app
   ```

3. **Run the app to verify setup**:
   ```
   flutter run
   ```

### Project Structure

Organize your project with this structure:

```
lib/
├── main.dart                 # Entry point
├── config/                   # Configuration files
├── models/                   # Data models
├── screens/                  # App screens
├── services/                 # API, database services
├── widgets/                  # Reusable UI components
└── utils/                    # Utility functions
```

### Essential Dependencies

Add these to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5            # For API requests
  provider: ^6.0.5         # For state management
  sqflite: ^2.2.8          # For local database
  shared_preferences: ^2.1.0 # For local storage
  fl_chart: ^0.62.0        # For data visualization
  pdf: ^3.9.0              # For PDF generation
  path_provider: ^2.0.14   # For file system access
  firebase_messaging: ^14.4.0 # For push notifications
  connectivity_plus: ^3.0.3 # For network connectivity
  web_socket_channel: ^2.3.0 # For real-time data
```

Run `flutter pub get` to install dependencies.

## Backend Integration

### API Service

Create a service to communicate with your backend:

```dart
// lib/services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/solar_data.dart';

class ApiService {
  final String baseUrl;
  
  ApiService({required this.baseUrl});
  
  Future<List<SolarData>> getLatestData() async {
    final response = await http.get(Uri.parse('$baseUrl/api/solar_data/latest'));
    
    if (response.statusCode == 200) {
      List<dynamic> jsonData = json.decode(response.body);
      return jsonData.map((data) => SolarData.fromJson(data)).toList();
    } else {
      throw Exception('Failed to load data');
    }
  }
  
  Future<bool> acknowledgeAlert(int alertId) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/alerts/$alertId/acknowledge'),
      headers: {'Content-Type': 'application/json'},
    );
    
    return response.statusCode == 200;
  }
  
  // Add more API methods as needed
}
```

### Data Models

Create models to represent your data:

```dart
// lib/models/solar_data.dart
class SolarData {
  final int id;
  final DateTime timestamp;
  final double voltage;
  final double current;
  final double power;
  final double temperature;
  final double irradiance;
  final int? prediction;
  
  SolarData({
    required this.id,
    required this.timestamp,
    required this.voltage,
    required this.current,
    required this.power,
    required this.temperature,
    required this.irradiance,
    this.prediction,
  });
  
  factory SolarData.fromJson(Map<String, dynamic> json) {
    return SolarData(
      id: json['id'],
      timestamp: DateTime.parse(json['timestamp']),
      voltage: json['pv_voltage'],
      current: json['pv_current'],
      power: json['power'],
      temperature: json['temperature'],
      irradiance: json['irradiance'],
      prediction: json['prediction'],
    );
  }
}
```

### Local Database

Set up a local database for offline access:

```dart
// lib/services/database_service.dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/solar_data.dart';

class DatabaseService {
  static Database? _database;
  
  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }
  
  Future<Database> _initDatabase() async {
    String path = join(await getDatabasesPath(), 'solar_monitor.db');
    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE solar_data(
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            voltage REAL,
            current REAL,
            power REAL,
            temperature REAL,
            irradiance REAL,
            prediction INTEGER
          )
        ''');
        
        await db.execute('''
          CREATE TABLE alerts(
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            fault_type INTEGER,
            message TEXT,
            acknowledged INTEGER
          )
        ''');
      },
    );
  }
  
  // Add methods to insert, query, and update data
}
```

## Real-time Data Visualization

### Creating Charts

Use `fl_chart` to create interactive charts:

```dart
// lib/widgets/line_chart_widget.dart
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../models/solar_data.dart';

class LineChartWidget extends StatelessWidget {
  final List<SolarData> data;
  final String title;
  
  const LineChartWidget({
    Key? key,
    required this.data,
    required this.title,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(show: true),
                  titlesData: FlTitlesData(show: true),
                  borderData: FlBorderData(show: true),
                  lineBarsData: [
                    LineChartBarData(
                      spots: _createSpots(),
                      isCurved: true,
                      color: Colors.blue,
                      barWidth: 3,
                      dotData: FlDotData(show: false),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  List<FlSpot> _createSpots() {
    return data.asMap().entries.map((entry) {
      // Choose which data to display (voltage, current, power, etc.)
      return FlSpot(entry.key.toDouble(), entry.value.voltage);
    }).toList();
  }
}
```

### Real-time Updates with WebSockets

Set up WebSocket connection for real-time data:

```dart
// lib/services/websocket_service.dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/solar_data.dart';

class WebSocketService {
  WebSocketChannel? _channel;
  Function(SolarData)? onDataReceived;
  
  bool get isConnected => _channel != null;
  
  void connect(String url) {
    _channel = WebSocketChannel.connect(Uri.parse(url));
    
    _channel!.stream.listen(
      (message) {
        final data = json.decode(message);
        if (data['type'] == 'solar_data') {
          final solarData = SolarData.fromJson(data['data']);
          onDataReceived?.call(solarData);
        }
      },
      onDone: () {
        _channel = null;
      },
      onError: (error) {
        _channel = null;
      },
    );
  }
  
  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
}
```

## Fault Detection Alerts

### Alert Model

Create a model for fault alerts:

```dart
// lib/models/alert.dart
class Alert {
  final int id;
  final DateTime timestamp;
  final int faultType;
  final String message;
  final bool acknowledged;
  
  Alert({
    required this.id,
    required this.timestamp,
    required this.faultType,
    required this.message,
    required this.acknowledged,
  });
  
  factory Alert.fromJson(Map<String, dynamic> json) {
    return Alert(
      id: json['id'],
      timestamp: DateTime.parse(json['timestamp']),
      faultType: json['fault_type'],
      message: json['message'],
      acknowledged: json['acknowledged'] == 1,
    );
  }
  
  String get faultTypeString {
    switch (faultType) {
      case 1: return 'Open Circuit';
      case 2: return 'Short Circuit';
      case 3: return 'Degradation';
      case 4: return 'Shading';
      default: return 'Unknown';
    }
  }
  
  Color get severityColor {
    switch (faultType) {
      case 1: return Colors.orange;
      case 2: return Colors.red;
      case 3: return Colors.yellow;
      case 4: return Colors.blue;
      default: return Colors.grey;
    }
  }
}
```

### Alert List Widget

Create a widget to display alerts:

```dart
// lib/widgets/alert_list.dart
import 'package:flutter/material.dart';
import '../models/alert.dart';
import '../services/api_service.dart';

class AlertList extends StatelessWidget {
  final List<Alert> alerts;
  final ApiService apiService;
  final Function onAcknowledge;
  
  const AlertList({
    Key? key,
    required this.alerts,
    required this.apiService,
    required this.onAcknowledge,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    if (alerts.isEmpty) {
      return const Center(
        child: Text('No alerts to display'),
      );
    }
    
    return ListView.builder(
      itemCount: alerts.length,
      itemBuilder: (context, index) {
        final alert = alerts[index];
        return Card(
          margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
          child: ListTile(
            leading: Icon(
              Icons.warning,
              color: alert.severityColor,
              size: 36,
            ),
            title: Text(alert.faultTypeString),
            subtitle: Text(
              '${alert.message}\n${_formatDate(alert.timestamp)}',
            ),
            trailing: alert.acknowledged
                ? const Icon(Icons.check_circle, color: Colors.green)
                : ElevatedButton(
                    onPressed: () async {
                      final success = await apiService.acknowledgeAlert(alert.id);
                      if (success) {
                        onAcknowledge();
                      }
                    },
                    child: const Text('Acknowledge'),
                  ),
          ),
        );
      },
    );
  }
  
  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year} ${date.hour}:${date.minute}';
  }
}
```

## PDF Report Generation

### Setting Up PDF Generation

Add the PDF generation functionality:

```dart
// lib/services/pdf_service.dart
import 'dart:io';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import '../models/solar_data.dart';
import '../models/alert.dart';

class PdfService {
  Future<File> generateDailyReport(List<SolarData> data, DateTime date) async {
    final pdf = pw.Document();
    
    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        build: (pw.Context context) {
          return [
            _buildHeader(date),
            _buildSummary(data),
            _buildChart(data),
            _buildDataTable(data),
          ];
        },
      ),
    );
    
    return _saveDocument(pdf, 'daily_report_${_formatDateForFilename(date)}.pdf');
  }
  
  Future<File> generateFaultReport(List<Alert> alerts, DateTime startDate, DateTime endDate) async {
    final pdf = pw.Document();
    
    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        build: (pw.Context context) {
          return [
            _buildFaultReportHeader(startDate, endDate),
            _buildFaultSummary(alerts),
            _buildFaultTable(alerts),
          ];
        },
      ),
    );
    
    return _saveDocument(pdf, 'fault_report_${_formatDateForFilename(startDate)}_${_formatDateForFilename(endDate)}.pdf');
  }
  
  pw.Widget _buildHeader(DateTime date) {
    return pw.Header(
      level: 0,
      child: pw.Text(
        'Solar Panel Performance Report - ${_formatDate(date)}',
        style: pw.TextStyle(
          fontSize: 24,
          fontWeight: pw.FontWeight.bold,
        ),
      ),
    );
  }
  
  pw.Widget _buildSummary(List<SolarData> data) {
    // Calculate summary statistics
    double avgVoltage = _calculateAverage(data.map((d) => d.voltage).toList());
    double avgCurrent = _calculateAverage(data.map((d) => d.current).toList());
    double avgPower = _calculateAverage(data.map((d) => d.power).toList());
    double maxPower = data.map((d) => d.power).reduce((a, b) => a > b ? a : b);
    
    return pw.Container(
      margin: const pw.EdgeInsets.only(top: 20),
      child: pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        children: [
          pw.Text(
            'Performance Summary',
            style: pw.TextStyle(
              fontSize: 18,
              fontWeight: pw.FontWeight.bold,
            ),
          ),
          pw.SizedBox(height: 10),
          pw.Row(
            mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
            children: [
              _buildSummaryItem('Avg Voltage', '${avgVoltage.toStringAsFixed(2)} V'),
              _buildSummaryItem('Avg Current', '${avgCurrent.toStringAsFixed(2)} A'),
              _buildSummaryItem('Avg Power', '${avgPower.toStringAsFixed(2)} W'),
              _buildSummaryItem('Max Power', '${maxPower.toStringAsFixed(2)} W'),
            ],
          ),
        ],
      ),
    );
  }
  
  // Add more helper methods for building the PDF
  
  Future<File> _saveDocument(pw.Document pdf, String filename) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$filename');
    await file.writeAsBytes(await pdf.save());
    return file;
  }
  
  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }
  
  String _formatDateForFilename(DateTime date) {
    return '${date.year}${date.month}${date.day}';
  }
  
  double _calculateAverage(List<double> values) {
    if (values.isEmpty) return 0;
    return values.reduce((a, b) => a + b) / values.length;
  }
}
```

### PDF Report Screen

Create a screen to generate reports:

```dart
// lib/screens/report_screen.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../services/pdf_service.dart';
import '../models/solar_data.dart';
import '../models/alert.dart';

class ReportScreen extends StatefulWidget {
  final ApiService apiService;
  
  const ReportScreen({
    Key? key,
    required this.apiService,
  }) : super(key: key);
  
  @override
  _ReportScreenState createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  final PdfService _pdfService = PdfService();
  DateTime _selectedDate = DateTime.now();
  bool _isLoading = false;
  String? _errorMessage;
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Generate Reports'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Select Report Type',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            _buildReportTypeCard(
              'Daily Performance Report',
              'View detailed performance data for a specific day',
              Icons.insert_chart,
              () => _generateDailyReport(),
            ),
            _buildReportTypeCard(
              'Fault Analysis Report',
              'View all detected faults and their details',
              Icons.warning,
              () => _generateFaultReport(),
            ),
            if (_isLoading)
              const Center(
                child: CircularProgressIndicator(),
              ),
            if (_errorMessage != null)
              Center(
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.red),
                ),
              ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildReportTypeCard(
    String title,
    String description,
    IconData icon,
    VoidCallback onTap,
  ) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              Icon(icon, size: 48, color: Theme.of(context).primaryColor),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(description),
                  ],
                ),
              ),
              const Icon(Icons.arrow_forward_ios),
            ],
          ),
        ),
      ),
    );
  }
  
  Future<void> _generateDailyReport() async {
    final DateTime? pickedDate = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
    );
    
    if (pickedDate != null) {
      setState(() {
        _selectedDate = pickedDate;
        _isLoading = true;
        _errorMessage = null;
      });
      
      try {
        // Fetch data for the selected date
        final data = await widget.apiService.getDailyData(_selectedDate);
        
        // Generate PDF
        final file = await _pdfService.generateDailyReport(data, _selectedDate);
        
        // Show success and open file
        _showReportSuccess(file.path);
      } catch (e) {
        setState(() {
          _errorMessage = 'Failed to generate report: $e';
        });
      } finally {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }
  
  // Add more methods for other report types
  
  void _showReportSuccess(String filePath) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Report Generated'),
        content: Text('Report saved to: $filePath'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
          ElevatedButton(
            onPressed: () {
              // Open the PDF file
              // You'll need a plugin like open_file to implement this
              Navigator.pop(context);
            },
            child: const Text('Open Report'),
          ),
        ],
      ),
    );
  }
}
```

## Offline Mode

### Implementing Offline Capability

Set up offline mode with local database:

```dart
// lib/services/sync_service.dart
import 'package:connectivity_plus/connectivity_plus.dart';
import '../services/api_service.dart';
import '../services/database_service.dart';
import '../models/solar_data.dart';

class SyncService {
  final ApiService apiService;
  final DatabaseService databaseService;
  
  SyncService({
    required this.apiService,
    required this.databaseService,
  });
  
  Future<bool> isConnected() async {
    final result = await Connectivity().checkConnectivity();
    return result != ConnectivityResult.none;
  }
  
  Future<List<SolarData>> getLatestData() async {
    if (await isConnected()) {
      try {
        // Try to get data from API
        final data = await apiService.getLatestData();
        
        // Save to local database
        for (var item in data) {
          await databaseService.insertSolarData(item);
        }
        
        return data;
      } catch (e) {
        // If API fails, fall back to local data
        return databaseService.getLatestSolarData(20);
      }
    } else {
      // If offline, use local data
      return databaseService.getLatestSolarData(20);
    }
  }
  
  // Add more sync methods for alerts and other data
}
```

## Testing and Deployment

### Testing Your App

1. **Run tests**:
   ```
   flutter test
   ```

2. **Test on real devices** before deploying

### Building the App for Release

1. **Android**:
   ```
   flutter build apk --release
   ```
   
   The APK will be at `build/app/outputs/flutter-apk/app-release.apk`

2. **iOS**:
   ```
   flutter build ios --release
   ```
   
   Then use Xcode to archive and distribute the app

### Publishing to App Stores

1. **Google Play Store**:
   - Create a developer account
   - Create a new application
   - Upload your APK
   - Fill in store listing details
   - Set up pricing and distribution
   - Submit for review

2. **Apple App Store**:
   - Create an App Store Connect account
   - Create a new iOS app
   - Upload your build using Xcode or Transporter
   - Fill in App Store information
   - Submit for review

## Troubleshooting

### Common Issues and Solutions

1. **App crashes on startup**:
   - Check if all dependencies are properly installed
   - Verify that the API URL is correct
   - Check for null safety issues in your code

2. **No data appears**:
   - Verify backend connection
   - Check if API endpoints are working
   - Test database connection

3. **Charts not displaying correctly**:
   - Ensure data format is correct
   - Check if there's enough data to display
   - Verify chart configuration

4. **Slow performance**:
   - Optimize database queries
   - Implement pagination for large datasets
   - Use caching where appropriate

5. **Push notifications not working**:
   - Verify Firebase configuration
   - Check device notification permissions
   - Test notification service on backend

### Getting Help

If you encounter issues not covered in this guide:

1. Check Flutter documentation: [flutter.dev/docs](https://flutter.dev/docs)
2. Search Stack Overflow for similar issues
3. Join Flutter community on Discord or Slack
4. File issues on the project GitHub repository
