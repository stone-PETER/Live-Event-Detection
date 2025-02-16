import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'dart:async';
import 'package:http/http.dart' as http;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _requestPermissions();
  runApp(const MyApp());
}

Future<void> _requestPermissions() async {
  await [Permission.microphone, Permission.location].request();
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Background Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Audio Background App'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final _recorder = Record();
  Timer? _timer;

  Future<void> _startRecording() async {
    if (await _recorder.hasPermission()) {
      await _recorder.start();
      // Periodically send audio data
      _timer = Timer.periodic(const Duration(seconds: 5), (timer) async {
        final path = await _recorder.stop();
        if (path != null) {
          final bytes = await File(path).readAsBytes();
          // Send to server
          await http.post(
            Uri.parse('https://example.com/upload'),
            body: bytes,
          );
          // Restart
          await _recorder.start();
        }
      });
    }
  }

  Future<void> _stopRecording() async {
    _timer?.cancel();
    await _recorder.stop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          ElevatedButton(
            onPressed: _startRecording,
            child: const Text('Start Recording'),
          ),
          ElevatedButton(
            onPressed: _stopRecording,
            child: const Text('Stop Recording'),
          ),
        ]),
      ),
    );
  }
}
