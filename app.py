from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Load TFLite Model ──
interpreter = Interpreter(model_path='model/model_unquant.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Load Labels ──
with open('model/labels.txt', 'r') as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# ── Remedy Map ──
remedies = {
    "Healthy": {
        "severity": "None",
        "remedy": "No action needed. Continue regular monitoring."
    },
    "Fruit Rot": {
        "severity": "Early Stage",
        "remedy": "Remove affected fruits immediately. Spray Bordeaux mixture 1%. Improve drainage around the tree base."
    },
    "Yellow Leaf Disease": {
        "severity": "Moderate",
        "remedy": "Apply recommended fertilizers. Remove yellowing leaves. Consult your local agricultural officer."
    }
}

# ── Predict Function ──
def predict_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output)
    confidence = float(output[predicted_index]) * 100

    disease = labels[predicted_index]
    info = remedies.get(disease, remedies["Healthy"])

    return {
        "disease": disease,
        "confidence": f"{confidence:.1f}%",
        "severity": info["severity"],
        "remedy": info["remedy"]
    }

# ── Sensor Data Store ──
sensor_data = {
    "temperature": "--",
    "humidity": "--",
    "timestamp": "--"
}

# ── Routes ──

@app.route('/')
def index():
    images = os.listdir(UPLOAD_FOLDER)
    latest_image = None
    latest_result = None

    if images:
        latest_image = sorted(images)[-1]
        image_path = os.path.join(UPLOAD_FOLDER, latest_image)
        latest_result = predict_disease(image_path)

    return render_template('index.html',
                           image=latest_image,
                           result=latest_result,
                           sensor=sensor_data)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files['image']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)

    if request.headers.get('X-Requested-From') == 'esp32':
        return jsonify({"status": "success", "result": result}), 200

    return redirect(url_for('index'))


@app.route('/sensor', methods=['POST'])
def sensor():
    data = request.get_json()
    if data:
        sensor_data['temperature'] = data.get('temperature', '--')
        sensor_data['humidity'] = data.get('humidity', '--')
        sensor_data['timestamp'] = datetime.now().strftime('%H:%M:%S')
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)