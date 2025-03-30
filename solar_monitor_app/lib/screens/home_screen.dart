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
  @override
  void initState() {
    super.initState();
    // Fetch initial data when screen loads
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final provider = Provider.of<SolarDataProvider>(context, listen: false);
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
                    const SizedBox(height: 16),
                    _buildCurrentStatus(provider),
                    const SizedBox(height: 24),
                    _buildRealtimeChart(provider),
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
            color: provider.isConnected ? Colors.green : Colors.red,
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

  Widget _buildCurrentStatus(SolarDataProvider provider) {
    final prediction = provider.currentPrediction;
    final latestData = provider.realtimeData.isNotEmpty
        ? provider.realtimeData.last
        : provider.historicalData.isNotEmpty
            ? provider.historicalData.first
            : null;

    if (latestData == null) {
      return const Center(
        child: Text('No data available'),
      );
    }

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
        const SizedBox(height: 16),
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
            const SizedBox(width: 16),
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
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: StatusCard(
                title: 'Power',
                value: '${latestData.power.toStringAsFixed(2)} W',
                icon: Icons.power,
                color: Colors.green,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: StatusCard(
                title: 'Temperature',
                value: '${latestData.temperature.toStringAsFixed(1)} °C',
                icon: Icons.thermostat,
                color: Colors.red,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        if (prediction != null)
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: prediction.prediction == 0
                  ? AppTheme.healthyColor.withOpacity(0.1)
                  : AppTheme.dangerColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: prediction.prediction == 0
                    ? AppTheme.healthyColor
                    : AppTheme.dangerColor,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      prediction.prediction == 0
                          ? Icons.check_circle
                          : Icons.warning,
                      color: prediction.prediction == 0
                          ? AppTheme.healthyColor
                          : AppTheme.dangerColor,
                      size: 24,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      prediction.faultType,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: prediction.prediction == 0
                            ? AppTheme.healthyColor
                            : AppTheme.dangerColor,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        '${prediction.confidence.toStringAsFixed(1)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  prediction.prediction == 0
                      ? 'The solar panel is operating normally.'
                      : 'Fault detected! Maintenance may be required.',
                  style: const TextStyle(fontSize: 16),
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildRealtimeChart(SolarDataProvider provider) {
    final data = provider.realtimeData.isEmpty
        ? provider.historicalData
        : provider.realtimeData;

    if (data.isEmpty) {
      return const SizedBox(
        height: 200,
        child: Center(
          child: Text('No data available for chart'),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Real-time Monitoring',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          height: 250,
          child: LineChart(
            LineChartData(
              gridData: FlGridData(
                show: true,
                drawVerticalLine: true,
                horizontalInterval: 10,
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
                    showTitles: true,
                    reservedSize: 30,
                    interval: 1,
                    getTitlesWidget: (value, meta) {
                      if (value.toInt() % 5 != 0) {
                        return const SizedBox();
                      }
                      return Text(
                        value.toInt().toString(),
                        style: const TextStyle(
                          color: Colors.grey,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      );
                    },
                  ),
                ),
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    interval: 10,
                    getTitlesWidget: (value, meta) {
                      return Text(
                        value.toInt().toString(),
                        style: const TextStyle(
                          color: Colors.grey,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      );
                    },
                    reservedSize: 42,
                  ),
                ),
              ),
              borderData: FlBorderData(
                show: true,
                border: Border.all(color: const Color(0xff37434d), width: 1),
              ),
              minX: 0,
              maxX: data.length.toDouble() - 1,
              minY: 0,
              maxY: 60,
              lineBarsData: [
                // Voltage Line
                LineChartBarData(
                  spots: List.generate(data.length, (index) {
                    return FlSpot(
                      index.toDouble(),
                      data[index].voltage,
                    );
                  }),
                  isCurved: true,
                  color: Colors.blue,
                  barWidth: 3,
                  isStrokeCapRound: true,
                  dotData: FlDotData(show: false),
                  belowBarData: BarAreaData(
                    show: true,
                    color: Colors.blue.withOpacity(0.1),
                  ),
                ),
                // Current Line
                LineChartBarData(
                  spots: List.generate(data.length, (index) {
                    return FlSpot(
                      index.toDouble(),
                      data[index].current,
                    );
                  }),
                  isCurved: true,
                  color: Colors.amber,
                  barWidth: 3,
                  isStrokeCapRound: true,
                  dotData: FlDotData(show: false),
                  belowBarData: BarAreaData(
                    show: true,
                    color: Colors.amber.withOpacity(0.1),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildChartLegend(Colors.blue, 'Voltage (V)'),
            const SizedBox(width: 24),
            _buildChartLegend(Colors.amber, 'Current (A)'),
          ],
        ),
      ],
    );
  }

  Widget _buildChartLegend(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(width: 8),
        Text(label),
      ],
    );
  }

  Widget _buildAlerts(SolarDataProvider provider) {
    final alerts = provider.alerts;

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
        const SizedBox(height: 16),
        if (alerts.isEmpty)
          const Center(
            child: Padding(
              padding: EdgeInsets.all(16.0),
              child: Text('No alerts'),
            ),
          )
        else
          ListView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: alerts.length,
            itemBuilder: (context, index) {
              final alert = alerts[index];
              return AlertCard(
                alert: alert,
                onAcknowledge: () => provider.acknowledgeAlert(alert.id),
              );
            },
          ),
      ],
    );
  }
}
