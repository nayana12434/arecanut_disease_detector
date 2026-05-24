from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Placeholder prediction (replace with real model later) ──
def predict_disease(image_path):
    """
    Temporary rule-based prediction.
    Will be replaced with TFLite model.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    avg_color = img_array.mean(axis=(0, 1))
    r, g, b = avg_color

    if r > 0.5 and g < 0.4:
        return {
            "disease": "Fruit Rot",
            "confidence": "72%",
            "severity": "Early Stage",
            "remedy": "Remove affected fruits. Spray Bordeaux mixture 1%. Improve drainage around the tree."
        }
    elif g > 0.5 and r > 0.45:
        return {
            "disease": "Yellow Leaf Disease",
            "confidence": "68%",
            "severity": "Moderate",
            "remedy": "Apply recommended fertilizers. Remove yellowing leaves. Consult local agri officer."
        }
    else:
        return {
            "disease": "Healthy",
            "confidence": "85%",
            "severity": "None",
            "remedy": "No action needed. Continue regular monitoring."
        }

# ── Store latest sensor data ──
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
    """Receives image from ESP32-CAM or manual upload"""
    if 'image' not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files['image']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)

    # If request is from browser (form), redirect to dashboard
    # If request is from ESP32-CAM (API), return JSON
    if request.headers.get('X-Requested-From') == 'esp32':
        return jsonify({"status": "success", "result": result}), 200
    
    from flask import redirect, url_for
    return redirect(url_for('index'))


@app.route('/sensor', methods=['POST'])
def sensor():
    """Receives sensor data from ESP32"""
    data = request.get_json()
    if data:
        sensor_data['temperature'] = data.get('temperature', '--')
        sensor_data['humidity'] = data.get('humidity', '--')
        sensor_data['timestamp'] = datetime.now().strftime('%H:%M:%S')
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
