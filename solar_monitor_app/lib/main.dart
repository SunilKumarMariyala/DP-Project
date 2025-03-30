import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:solar_monitor_app/providers/solar_data_provider.dart';
import 'package:solar_monitor_app/screens/home_screen.dart';
import 'package:solar_monitor_app/screens/login_screen.dart';
import 'package:solar_monitor_app/screens/settings_screen.dart';
import 'package:solar_monitor_app/utils/app_theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

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

class SolarMonitorApp extends StatelessWidget {
  const SolarMonitorApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Solar Panel Monitor',
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.system,
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/login': (context) => const LoginScreen(),
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}
