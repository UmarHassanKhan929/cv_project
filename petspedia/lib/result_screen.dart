import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:csv/csv.dart';

class ResultScreen extends StatefulWidget {
  final String imagePath;

  ResultScreen({required this.imagePath});

  @override
  _ResultScreenState createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late Interpreter _interpreter;
  String _classificationResult = '';

  @override
  void initState() {
    super.initState();
    Future.delayed(Duration.zero, () async {
      await _loadModel();
      await _classifyImage(File(widget.imagePath));
    });
    // _loadModel();
    // _classifyImage(File(widget.imagePath));
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('dog_breeds_model.tflite');
  }

  Future<void> _classifyImage(File image) async {
    // Load the image and convert it to a suitable format for the model
    final loadedImage = img.decodeImage(await image.readAsBytes());
    final resizedImage = img.copyResize(loadedImage!,
        width: 299, height: 299); // Update the width and height
    final imageBytes = resizedImage.getBytes(order: img.ChannelOrder.rgb);

    // Prepare input tensor
    final inputShape = _interpreter.getInputTensor(0).shape;
    final inputType = _interpreter.getInputTensor(0).type;
    print("=====================================");
    print(inputShape);
    final inputSize = inputShape.reduce((value, element) => value * element);
    var input = Uint8List(inputSize).buffer.asFloat32List();
    print("=++++++++++++++++++++++++++++=");
    print(input);

    // Normalize the image and fill the input tensor
    int index = 0;
    for (int i = 0; i < resizedImage.height; i++) {
      for (int j = 0; j < resizedImage.width; j++) {
        for (int k = 0; k < 3; k++) {
          int pixelIndex = (i * resizedImage.width + j) * 3;
          input[index] = imageBytes[pixelIndex + k] / 255.0;
          index++;
        }
      }
    }

    // Prepare output tensor
    final outputShape = _interpreter.getOutputTensor(0).shape;
    final outputType = _interpreter.getOutputTensor(0).type;
    print("=====================================");
    print(outputShape);
    var output =
        Uint8List(1 * outputShape[0] * outputShape[1]).buffer.asFloat32List();

    // Run inference
    _interpreter.run(
        input.buffer.asFloat32List().reshape([1, 299, 299, 3]), output);

    // Process and display the classification result
    int maxIndex = output.indexOf(output.reduce(max));

    // Load the labels of the dog breeds
    List<String> labels = await _loadLabels(
        'assets/labels.csv'); // Replace with the path to your labels CSV file

    setState(() {
      _classificationResult = labels[maxIndex];
    });
  }

  Future<List<String>> _loadLabels(String assetPath) async {
    String labelsData = await rootBundle.loadString(assetPath);
    List<List<dynamic>> rowsAsListOfValues =
        const CsvToListConverter().convert(labelsData);

    // Assuming one label per line in the CSV file
    return rowsAsListOfValues.map((row) => row[0].toString()).toList();
  }

  Future<void> _search() async {
    // Send a request to your API with the classification result and the selected dropdown options.
    // Receive and display the response.
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Result'),
      ),
      body: ListView(
        children: [
          Image.file(File(widget.imagePath)),
          Text('Classification result: $_classificationResult'),
          // Add your dropdown options here.
          ElevatedButton(
            onPressed: _search,
            child: Text('Search'),
          ),
          // Display the API response here.
        ],
      ),
    );
  }
}
