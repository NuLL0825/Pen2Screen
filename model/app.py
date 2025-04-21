from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image

app = Flask(__name__, static_folder='temp')

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Save uploaded file
        image = request.files['image']
        input_path = os.path.join(app.static_folder, "input.jpg")
        output_path = os.path.join(app.static_folder, "output.jpg")
        image.save(input_path)
        
        # Process image
        binarize_image(input_path, output_path)

        # Return path to processed image (accessible via a new endpoint)
        return jsonify({"message": "Processing complete", "processed_image_path": "http://192.168.129.178:5000/processed_image.jpg"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
  
@app.route('/processed_image.jpg')
def get_processed_image():
    print("Serving processed image from:", os.path.join(app.static_folder, 'output.jpg'))
    return send_from_directory(app.static_folder, 'output.jpg')

def binarize_image(input_image_path, output_image_path, threshold=128):
    with Image.open(input_image_path) as img:
        gray_img = img.convert("L")
        binarized_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
        binarized_img.save(output_image_path)

if __name__ == '__main__':
    if not os.path.exists(app.static_folder):
        os.makedirs(app.static_folder)
    app.run(host="0.0.0.0", port=5000)
