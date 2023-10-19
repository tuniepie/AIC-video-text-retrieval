from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Replace this with actual image paths
image_paths = [
    'image1.jpg',
    'image2.jpg',
    'image3.jpg',
    # Add more image paths here
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_images():
    data = request.get_json()
    search_term = data.get('input', '').lower()

    filtered_images = [image for image in image_paths if search_term in image.lower()]

    return jsonify({'imagePaths': filtered_images})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8282, debug=True)
