import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:socket_io_client/socket_io_client.dart' as io;

class SolarPanelData {
  final double voltage;
  final double current;
  final double power;
  final double temperature;
  final double irradiance;
  final String timestamp;

  SolarPanelData({
    required this.voltage,
    required this.current,
    required this.power,
    required this.temperature,
    required this.irradiance,
    required this.timestamp,
  });

  factory SolarPanelData.fromJson(Map<String, dynamic> json) {
    return SolarPanelData(
      voltage: json['voltage']?.toDouble() ?? 0.0,
      current: json['current']?.toDouble() ?? 0.0,
      power: json['power']?.toDouble() ?? 0.0,
      temperature: json['temperature']?.toDouble() ?? 0.0,
      irradiance: json['irradiance']?.toDouble() ?? 0.0,
      timestamp: json['timestamp'] ?? DateTime.now().toString(),
    );
  }
}

class PredictionData {
  final int prediction;
  final String faultType;
  final double confidence;
  final List<double> probabilities;

  PredictionData({
    required this.prediction,
    required this.faultType,
    required this.confidence,
    required this.probabilities,
  });

  factory PredictionData.fromJson(Map<String, dynamic> json) {
    return PredictionData(
      prediction: json['prediction'] ?? 0,
      faultType: json['fault_type'] ?? 'Unknown',
      confidence: json['confidence']?.toDouble() ?? 0.0,
      probabilities: (json['probabilities'] as List<dynamic>?)
          ?.map((e) => e.toDouble())
          .toList() ??
          [],
    );
  }
}

class Alert {
  final int id;
  final String message;
  final String severity;
  final String timestamp;
  final bool acknowledged;

  Alert({
    required this.id,
    required this.message,
    required this.severity,
    required this.timestamp,
    required this.acknowledged,
  });

  factory Alert.fromJson(Map<String, dynamic> json) {
    return Alert(
      id: json['id'] ?? 0,
      message: json['message'] ?? '',
      severity: json['severity'] ?? 'low',
      timestamp: json['timestamp'] ?? DateTime.now().toString(),
      acknowledged: json['acknowledged'] ?? false,
    );
  }
}

class SolarDataProvider with ChangeNotifier {
  String _serverUrl;
  io.Socket? _socket;
  bool _isConnected = false;
  bool _isMonitoring = false;
  
  // Real-time data
  List<SolarPanelData> _realtimeData = [];
  PredictionData? _currentPrediction;
  List<Alert> _alerts = [];
  
  // Historical data
  List<SolarPanelData> _historicalData = [];
  
  // Getters
  String get serverUrl => _serverUrl;
  bool get isConnected => _isConnected;
  bool get isMonitoring => _isMonitoring;
  List<SolarPanelData> get realtimeData => _realtimeData;
  PredictionData? get currentPrediction => _currentPrediction;
  List<Alert> get alerts => _alerts;
  List<SolarPanelData> get historicalData => _historicalData;
  
  SolarDataProvider({required String serverUrl}) : _serverUrl = serverUrl {
    _initSocket();
  }
  
  void _initSocket() {
    try {
      _socket = io.io(_serverUrl, <String, dynamic>{
        'transports': ['websocket'],
        'autoConnect': true,
      });
      
      _socket!.onConnect((_) {
        _isConnected = true;
        notifyListeners();
        print('Connected to server: $_serverUrl');
      });
      
      _socket!.onDisconnect((_) {
        _isConnected = false;
        notifyListeners();
        print('Disconnected from server');
      });
      
      _socket!.on('data_update', (data) {
        final newData = SolarPanelData.fromJson(data);
        _realtimeData.add(newData);
        if (_realtimeData.length > 20) {
          _realtimeData.removeAt(0);
        }
        notifyListeners();
      });
      
      _socket!.on('prediction_update', (data) {
        _currentPrediction = PredictionData.fromJson(data);
        notifyListeners();
      });
      
      _socket!.on('alert', (data) {
        final newAlert = Alert.fromJson(data);
        _alerts.insert(0, newAlert);
        if (_alerts.length > 10) {
          _alerts.removeLast();
        }
        notifyListeners();
      });
      
      _socket!.connect();
    } catch (e) {
      print('Error initializing socket: $e');
    }
  }
  
  void updateServerUrl(String newUrl) {
    if (_serverUrl != newUrl) {
      _serverUrl = newUrl;
      
      // Disconnect old socket
      if (_socket != null) {
        _socket!.disconnect();
        _socket!.dispose();
      }
      
      // Initialize new socket
      _initSocket();
      notifyListeners();
    }
  }
  
  Future<void> startMonitoring() async {
    try {
      final response = await http.post(
        Uri.parse('$_serverUrl/api/start'),
        headers: {'Content-Type': 'application/json'},
      );
      
      if (response.statusCode == 200) {
        _isMonitoring = true;
        notifyListeners();
      } else {
        throw Exception('Failed to start monitoring');
      }
    } catch (e) {
      print('Error starting monitoring: $e');
      rethrow;
    }
  }
  
  Future<void> stopMonitoring() async {
    try {
      final response = await http.post(
        Uri.parse('$_serverUrl/api/stop'),
        headers: {'Content-Type': 'application/json'},
      );
      
      if (response.statusCode == 200) {
        _isMonitoring = false;
        notifyListeners();
      } else {
        throw Exception('Failed to stop monitoring');
      }
    } catch (e) {
      print('Error stopping monitoring: $e');
      rethrow;
    }
  }
  
  Future<void> fetchLatestData() async {
    try {
      final response = await http.get(
        Uri.parse('$_serverUrl/api/data/latest?limit=10'),
      );
      
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        _historicalData = data.map((item) => SolarPanelData.fromJson(item)).toList();
        notifyListeners();
      } else {
        throw Exception('Failed to load latest data');
      }
    } catch (e) {
      print('Error fetching latest data: $e');
      rethrow;
    }
  }
  
  Future<void> fetchAlerts() async {
    try {
      final response = await http.get(
        Uri.parse('$_serverUrl/api/alerts/latest?limit=5'),
      );
      
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        _alerts = data.map((item) => Alert.fromJson(item)).toList();
        notifyListeners();
      } else {
        throw Exception('Failed to load alerts');
      }
    } catch (e) {
      print('Error fetching alerts: $e');
      rethrow;
    }
  }
  
  Future<void> acknowledgeAlert(int alertId) async {
    try {
      final response = await http.post(
        Uri.parse('$_serverUrl/api/alerts/$alertId/acknowledge'),
        headers: {'Content-Type': 'application/json'},
      );
      
      if (response.statusCode == 200) {
        // Update local alert status
        final index = _alerts.indexWhere((alert) => alert.id == alertId);
        if (index != -1) {
          final updatedAlert = Alert(
            id: _alerts[index].id,
            message: _alerts[index].message,
            severity: _alerts[index].severity,
            timestamp: _alerts[index].timestamp,
            acknowledged: true,
          );
          _alerts[index] = updatedAlert;
          notifyListeners();
        }
      } else {
        throw Exception('Failed to acknowledge alert');
      }
    } catch (e) {
      print('Error acknowledging alert: $e');
      rethrow;
    }
  }
  
  @override
  void dispose() {
    if (_socket != null) {
      _socket!.disconnect();
      _socket!.dispose();
    }
    super.dispose();
  }
}
