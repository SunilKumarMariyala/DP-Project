import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import 'package:solar_monitor_app/providers/solar_data_provider.dart';
import 'package:solar_monitor_app/widgets/status_card.dart';
import 'package:solar_monitor_app/widgets/alert_card.dart';
import 'package:solar_monitor_app/utils/app_theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _databaseConnected = false;

  @override
  void initState() {
    super.initState();
    // Fetch initial data when screen loads
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final provider = Provider.of<SolarDataProvider>(context, listen: false);
      // Check database connection
      _databaseConnected = await provider.checkDatabaseConnection();
      setState(() {});
      
      provider.fetchLatestData();
      provider.fetchAlerts();
    });
  }

  @override
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
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildConnectionStatus(provider),
                    const SizedBox(height: 8),
                    _buildDatabaseStatus(),
                    const SizedBox(height: 16),
                    _buildCurrentStatus(provider),
                    const SizedBox(height: 24),
                    _buildRealtimeChart(provider),
                    const SizedBox(height: 24),
                    _buildPredictionInfo(provider),
                    const SizedBox(height: 24),
                    _buildAlerts(provider),
                  ],
                ),
              ),
            ),
          );
        },
      ),
      floatingActionButton: Consumer<SolarDataProvider>(
        builder: (context, provider, child) {
          return FloatingActionButton(
            onPressed: () {
              if (provider.isMonitoring) {
                provider.stopMonitoring();
              } else {
                provider.startMonitoring();
              }
            },
            backgroundColor: provider.isMonitoring
                ? Colors.red
                : AppTheme.primaryColor,
            child: Icon(
              provider.isMonitoring ? Icons.stop : Icons.play_arrow,
            ),
          );
        },
      ),
    );
  }

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
          const SizedBox(width: 8),
          Text(
            provider.isConnected
                ? 'Connected to server'
                : 'Disconnected from server',
            style: TextStyle(
              color: provider.isConnected ? Colors.green.shade800 : Colors.red.shade800,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDatabaseStatus() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      decoration: BoxDecoration(
        color: _databaseConnected ? Colors.green.shade100 : Colors.orange.shade100,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(
            _databaseConnected ? Icons.storage : Icons.storage_outlined,
            color: _databaseConnected ? Colors.green.shade800 : Colors.orange.shade800,
          ),
          const SizedBox(width: 8),
          Text(
            _databaseConnected
                ? 'MySQL Database Connected'
                : 'Database Connection Issue',
            style: TextStyle(
              color: _databaseConnected ? Colors.green.shade800 : Colors.orange.shade800,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

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
        const SizedBox(height: 8),
        if (latestData == null)
          const Text('No data available')
        else
          Row(
            children: [
              Expanded(
                child: StatusCard(
                  title: 'Voltage',
                  value: '${latestData.voltage.toStringAsFixed(2)} V',
                  icon: Icons.bolt,
                  color: Colors.blue,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: StatusCard(
                  title: 'Current',
                  value: '${latestData.current.toStringAsFixed(2)} A',
                  icon: Icons.electric_bolt,
                  color: Colors.amber,
                ),
              ),
            ],
          ),
        const SizedBox(height: 8),
        if (latestData != null)
          Row(
            children: [
              Expanded(
                child: StatusCard(
                  title: 'Power',
                  value: '${latestData.power.toStringAsFixed(2)} W',
                  icon: Icons.power,
                  color: Colors.red,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: StatusCard(
                  title: 'Temperature',
                  value: '${latestData.temperature.toStringAsFixed(1)} °C',
                  icon: Icons.thermostat,
                  color: Colors.orange,
                ),
              ),
            ],
          ),
        const SizedBox(height: 8),
        if (latestData != null)
          Row(
            children: [
              Expanded(
                child: StatusCard(
                  title: 'Irradiance',
                  value: '${latestData.irradiance.toStringAsFixed(1)} W/m²',
                  icon: Icons.wb_sunny,
                  color: Colors.amber,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: StatusCard(
                  title: 'Last Update',
                  value: _formatTimestamp(latestData.timestamp),
                  icon: Icons.access_time,
                  color: Colors.teal,
                ),
              ),
            ],
          ),
      ],
    );
  }

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
        const SizedBox(height: 8),
        Container(
          height: 200,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.1),
                spreadRadius: 1,
                blurRadius: 4,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          padding: const EdgeInsets.all(16),
          child: LineChart(
            LineChartData(
              gridData: FlGridData(
                show: true,
                drawVerticalLine: true,
                horizontalInterval: 1,
                verticalInterval: 1,
              ),
              titlesData: FlTitlesData(
                show: true,
                rightTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                topTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: false,
                  ),
                ),
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 40,
                  ),
                ),
              ),
              borderData: FlBorderData(
                show: true,
                border: Border.all(color: const Color(0xff37434d), width: 1),
              ),
              minX: 0,
              maxX: (provider.realtimeData.length - 1).toDouble(),
              minY: 0,
              maxY: _getMaxValue(provider),
              lineBarsData: [
                // Voltage line
                LineChartBarData(
                  spots: _getSpots(provider, (data) => data.voltage),
                  isCurved: true,
                  color: Colors.blue,
                  barWidth: 2,
                  isStrokeCapRound: true,
                  dotData: FlDotData(show: false),
                  belowBarData: BarAreaData(show: false),
                ),
                // Current line
                LineChartBarData(
                  spots: _getSpots(provider, (data) => data.current * 10), // Scale for visibility
                  isCurved: true,
                  color: Colors.amber,
                  barWidth: 2,
                  isStrokeCapRound: true,
                  dotData: FlDotData(show: false),
                  belowBarData: BarAreaData(show: false),
                ),
                // Power line
                LineChartBarData(
                  spots: _getSpots(provider, (data) => data.power / 10), // Scale for visibility
                  isCurved: true,
                  color: Colors.red,
                  barWidth: 2,
                  isStrokeCapRound: true,
                  dotData: FlDotData(show: false),
                  belowBarData: BarAreaData(show: false),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildChartLegend('Voltage (V)', Colors.blue),
            const SizedBox(width: 16),
            _buildChartLegend('Current (A × 10)', Colors.amber),
            const SizedBox(width: 16),
            _buildChartLegend('Power (W ÷ 10)', Colors.red),
          ],
        ),
      ],
    );
  }

  Widget _buildPredictionInfo(SolarDataProvider provider) {
    final prediction = provider.currentPrediction;
    
    if (prediction == null) {
      return const SizedBox.shrink();
    }

    Color statusColor;
    IconData statusIcon;
    
    switch (prediction.faultType.toLowerCase()) {
      case 'normal':
      case 'no fault':
        statusColor = Colors.green;
        statusIcon = Icons.check_circle;
        break;
      case 'warning':
        statusColor = Colors.orange;
        statusIcon = Icons.warning;
        break;
      default:
        statusColor = Colors.red;
        statusIcon = Icons.error;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Fault Detection',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: statusColor.withOpacity(0.1),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: statusColor.withOpacity(0.5)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(statusIcon, color: statusColor, size: 24),
                  const SizedBox(width: 8),
                  Text(
                    prediction.faultType,
                    style: TextStyle(
                      color: statusColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    'Confidence: ${(prediction.confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: statusColor,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
              if (prediction.description.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(
                  'Description:',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.grey.shade800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(prediction.description),
              ],
              if (prediction.recommendedAction.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(
                  'Recommended Action:',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.grey.shade800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(prediction.recommendedAction),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildAlerts(SolarDataProvider provider) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Recent Alerts',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        if (provider.alerts.isEmpty)
          const Text('No recent alerts')
        else
          ListView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: provider.alerts.length,
            itemBuilder: (context, index) {
              final alert = provider.alerts[index];
              return AlertCard(
                alert: alert,
                onAcknowledge: () {
                  provider.acknowledgeAlert(alert.id);
                },
              );
            },
          ),
      ],
    );
  }

  Widget _buildChartLegend(String label, Color color) {
    return Row(
      children: [
        Container(
          width: 12,
          height: 12,
          color: color,
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }

  List<FlSpot> _getSpots(
    SolarDataProvider provider,
    double Function(SolarPanelData) getValue,
  ) {
    return List.generate(
      provider.realtimeData.length,
      (index) => FlSpot(
        index.toDouble(),
        getValue(provider.realtimeData[index]),
      ),
    );
  }

  double _getMaxValue(SolarDataProvider provider) {
    double maxVoltage = 0;
    double maxScaledCurrent = 0;
    double maxScaledPower = 0;

    for (final data in provider.realtimeData) {
      if (data.voltage > maxVoltage) {
        maxVoltage = data.voltage;
      }
      if (data.current * 10 > maxScaledCurrent) {
        maxScaledCurrent = data.current * 10;
      }
      if (data.power / 10 > maxScaledPower) {
        maxScaledPower = data.power / 10;
      }
    }

    return [maxVoltage, maxScaledCurrent, maxScaledPower].reduce(
      (a, b) => a > b ? a : b,
    );
  }

  String _formatTimestamp(String timestamp) {
    try {
      final dateTime = DateTime.parse(timestamp);
      return DateFormat('HH:mm:ss').format(dateTime);
    } catch (e) {
      return timestamp;
    }
  }
}
