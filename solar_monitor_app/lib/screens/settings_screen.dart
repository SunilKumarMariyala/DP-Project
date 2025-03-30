import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:solar_monitor_app/providers/solar_data_provider.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _serverUrlController = TextEditingController();
  final _databaseUsernameController = TextEditingController();
  final _databasePasswordController = TextEditingController();
  final _databaseNameController = TextEditingController();
  
  bool _isDarkMode = false;
  bool _notificationsEnabled = true;
  String _updateInterval = '5';
  bool _showDatabaseSettings = false;
  bool _obscurePassword = true;
  
  @override
  void initState() {
    super.initState();
    _loadSettings();
  }
  
  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final provider = Provider.of<SolarDataProvider>(context, listen: false);
    
    setState(() {
      _serverUrlController.text = prefs.getString('server_url') ?? provider.serverUrl;
      _databaseUsernameController.text = prefs.getString('db_username') ?? 'root';
      _databasePasswordController.text = prefs.getString('db_password') ?? '';
      _databaseNameController.text = prefs.getString('db_name') ?? 'solar_panel_db';
      _isDarkMode = prefs.getBool('dark_mode') ?? false;
      _notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
      _updateInterval = prefs.getString('update_interval') ?? '5';
    });
  }
  
  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final provider = Provider.of<SolarDataProvider>(context, listen: false);
    
    // Save server URL and update provider
    final newServerUrl = _serverUrlController.text.trim();
    if (newServerUrl.isNotEmpty && newServerUrl != provider.serverUrl) {
      await prefs.setString('server_url', newServerUrl);
      provider.updateServerUrl(newServerUrl);
    }
    
    // Save database settings
    await prefs.setString('db_username', _databaseUsernameController.text.trim());
    await prefs.setString('db_password', _databasePasswordController.text);
    await prefs.setString('db_name', _databaseNameController.text.trim());
    
    // Save other settings
    await prefs.setBool('dark_mode', _isDarkMode);
    await prefs.setBool('notifications_enabled', _notificationsEnabled);
    await prefs.setString('update_interval', _updateInterval);
    
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Settings saved')),
    );
  }
  
  @override
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
              const SizedBox(height: 16),
              TextFormField(
                controller: _serverUrlController,
                decoration: const InputDecoration(
                  labelText: 'Server URL',
                  hintText: 'http://192.168.1.100:8080',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.link),
                ),
                keyboardType: TextInputType.url,
              ),
              const SizedBox(height: 16),
              
              // MySQL Database Settings
              ExpansionTile(
                title: const Text(
                  'MySQL Database Settings',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                leading: const Icon(Icons.storage),
                initiallyExpanded: _showDatabaseSettings,
                onExpansionChanged: (expanded) {
                  setState(() {
                    _showDatabaseSettings = expanded;
                  });
                },
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16.0),
                    child: Column(
                      children: [
                        TextFormField(
                          controller: _databaseUsernameController,
                          decoration: const InputDecoration(
                            labelText: 'Database Username',
                            hintText: 'root',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.person),
                          ),
                        ),
                        const SizedBox(height: 16),
                        TextFormField(
                          controller: _databasePasswordController,
                          obscureText: _obscurePassword,
                          decoration: InputDecoration(
                            labelText: 'Database Password',
                            border: const OutlineInputBorder(),
                            prefixIcon: const Icon(Icons.lock),
                            suffixIcon: IconButton(
                              icon: Icon(
                                _obscurePassword ? Icons.visibility : Icons.visibility_off,
                              ),
                              onPressed: () {
                                setState(() {
                                  _obscurePassword = !_obscurePassword;
                                });
                              },
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        TextFormField(
                          controller: _databaseNameController,
                          decoration: const InputDecoration(
                            labelText: 'Database Name',
                            hintText: 'solar_panel_db',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.database),
                          ),
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          onPressed: () async {
                            final provider = Provider.of<SolarDataProvider>(context, listen: false);
                            final isConnected = await provider.checkDatabaseConnection();
                            
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(
                                  isConnected
                                      ? 'Database connection successful!'
                                      : 'Failed to connect to database. Check settings.',
                                ),
                                backgroundColor: isConnected ? Colors.green : Colors.red,
                              ),
                            );
                          },
                          icon: const Icon(Icons.check_circle),
                          label: const Text('Test Connection'),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(double.infinity, 48),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 24),
              const Text(
                'App Settings',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              SwitchListTile(
                title: const Text('Dark Mode'),
                subtitle: const Text('Enable dark theme for the app'),
                value: _isDarkMode,
                onChanged: (value) {
                  setState(() {
                    _isDarkMode = value;
                  });
                },
              ),
              SwitchListTile(
                title: const Text('Notifications'),
                subtitle: const Text('Enable alerts and notifications'),
                value: _notificationsEnabled,
                onChanged: (value) {
                  setState(() {
                    _notificationsEnabled = value;
                  });
                },
              ),
              ListTile(
                title: const Text('Update Interval'),
                subtitle: const Text('How often to update data (seconds)'),
                trailing: DropdownButton<String>(
                  value: _updateInterval,
                  onChanged: (value) {
                    if (value != null) {
                      setState(() {
                        _updateInterval = value;
                      });
                    }
                  },
                  items: const [
                    DropdownMenuItem(value: '1', child: Text('1 second')),
                    DropdownMenuItem(value: '5', child: Text('5 seconds')),
                    DropdownMenuItem(value: '10', child: Text('10 seconds')),
                    DropdownMenuItem(value: '30', child: Text('30 seconds')),
                    DropdownMenuItem(value: '60', child: Text('1 minute')),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _saveSettings,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 48),
                ),
                child: const Text('Save Settings'),
              ),
              const SizedBox(height: 16),
              TextButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('About'),
                      content: const Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Solar Panel Monitor App'),
                          SizedBox(height: 8),
                          Text('Version: 1.1.0 (MySQL Edition)'),
                          SizedBox(height: 8),
                          Text('This app monitors solar panel performance and detects faults using machine learning.'),
                        ],
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Close'),
                        ),
                      ],
                    ),
                  );
                },
                child: const Text('About'),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _serverUrlController.dispose();
    _databaseUsernameController.dispose();
    _databasePasswordController.dispose();
    _databaseNameController.dispose();
    super.dispose();
  }
}
