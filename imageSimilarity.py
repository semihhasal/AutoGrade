# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, current_app
import cv2
import numpy as np
import os
import sys
import io
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS

# Karakter kodlama hatalarını önlemek için stdout ve stderr'i yeniden yönlendirin
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

# VGG16 Modelini Yükle
base_model = VGG16(weights='imagenet')
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similar_regions = [i for i in matches if i.distance < 59]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

def structural_sim(img1, img2):
    sim, _ = ssim(img1, img2, full=True)
    return sim

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def vgg16_feature_similarity(model, image1, image2):
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    feat1 = model.predict(img1)[0]
    feat2 = model.predict(img2)[0]
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity

def resize_image_to_match(image, reference_image):
    dimensions = reference_image.shape[1], reference_image.shape[0]
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

@app.route('/api/compare', methods=['GET'])
def compare_images():
    try:
        image_path = request.args.get('image')
        full_image_path = os.path.join('images', image_path)
        base_image_path = 'c1.jpg'

        current_app.logger.info(f"Received request to compare {full_image_path} with {base_image_path}")
        current_app.logger.info(f"Checking if images exist")

        if not os.path.exists(full_image_path) or not os.path.exists(base_image_path):
            current_app.logger.error(f"Image not found: {full_image_path} or base image {base_image_path}")
            return jsonify({'error': 'Image not found'}), 404

        current_app.logger.info(f"Comparing base image {base_image_path} with {full_image_path}")

        comparison_image = cv2.imread(full_image_path, 0)
        base_image = cv2.imread(base_image_path, 0)
        comparison_image = resize_image_to_match(comparison_image, base_image)

        orb_sim_score = orb_similarity(base_image, comparison_image)
        ssim_score = structural_sim(base_image, comparison_image)
        vgg16_score = vgg16_feature_similarity(model, base_image_path, full_image_path)

        orb_sim_score = float(orb_sim_score)
        ssim_score = float(ssim_score)
        vgg16_score = float(vgg16_score)

        current_app.logger.info(f"ORB Score: {orb_sim_score}, SSIM Score: {ssim_score}, VGG16 Score: {vgg16_score}")

        return jsonify({
            'orb': orb_sim_score,
            'ssim': ssim_score,
            'vgg16': vgg16_score
        })

    except Exception as e:
        current_app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare/upload', methods=['POST'])
def compare_uploaded_image():
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        temp_image_path = 'temp_uploaded_image.png'
        file.save(temp_image_path)

        base_image_path = 'c1.jpg'
        current_app.logger.info(f"Received file upload request")

        if not os.path.exists(base_image_path):
            current_app.logger.error(f"Base image not found: {base_image_path}")
            return jsonify({'error': 'Base image not found'}), 404

        current_app.logger.info(f"Comparing base image {base_image_path} with uploaded image {temp_image_path}")

        comparison_image = cv2.imread(temp_image_path, 0)
        base_image = cv2.imread(base_image_path, 0)
        comparison_image = resize_image_to_match(comparison_image, base_image)

        orb_sim_score = orb_similarity(base_image, comparison_image)
        ssim_score = structural_sim(base_image, comparison_image)
        vgg16_score = vgg16_feature_similarity(model, base_image_path, temp_image_path)

        orb_sim_score = float(orb_sim_score)
        ssim_score = float(ssim_score)
        vgg16_score = float(vgg16_score)

        current_app.logger.info(f"ORB Score: {orb_sim_score}, SSIM Score: {ssim_score}, VGG16 Score: {vgg16_score}")

        os.remove(temp_image_path)

        return jsonify({
            'orb': orb_sim_score,
            'ssim': ssim_score,
            'vgg16': vgg16_score
        })

    except Exception as e:
        current_app.logger.error(f"Error processing uploaded image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
