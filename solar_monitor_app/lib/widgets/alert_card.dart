import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:solar_monitor_app/providers/solar_data_provider.dart';
import 'package:solar_monitor_app/utils/app_theme.dart';

class AlertCard extends StatelessWidget {
  final Alert alert;
  final VoidCallback onAcknowledge;

  const AlertCard({
    Key? key,
    required this.alert,
    required this.onAcknowledge,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Color severityColor;
    IconData severityIcon;

    switch (alert.severity.toLowerCase()) {
      case 'high':
        severityColor = AppTheme.dangerColor;
        severityIcon = Icons.error;
        break;
      case 'medium':
        severityColor = AppTheme.warningColor;
        severityIcon = Icons.warning;
        break;
      default:
        severityColor = AppTheme.accentColor;
        severityIcon = Icons.info;
        break;
    }

    // Format the timestamp
    DateTime timestamp;
    try {
      timestamp = DateTime.parse(alert.timestamp);
    } catch (e) {
      timestamp = DateTime.now();
    }
    final formattedTime = DateFormat('MMM d, h:mm a').format(timestamp);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: severityColor.withOpacity(0.5),
          width: 1,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  severityIcon,
                  color: severityColor,
                  size: 24,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    alert.message,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  formattedTime,
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 14,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: severityColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: severityColor.withOpacity(0.5),
                    ),
                  ),
                  child: Text(
                    alert.severity.toUpperCase(),
                    style: TextStyle(
                      color: severityColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            if (!alert.acknowledged)
              Padding(
                padding: const EdgeInsets.only(top: 12.0),
                child: Align(
                  alignment: Alignment.centerRight,
                  child: OutlinedButton(
                    onPressed: onAcknowledge,
                    style: OutlinedButton.styleFrom(
                      side: BorderSide(color: AppTheme.primaryColor),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text('Acknowledge'),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
