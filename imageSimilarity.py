# -*- coding: utf-8 -*-
import math

from flask import Flask, request, jsonify, current_app
import cv2
import numpy as np
import os
import sys
import io
import csv
import random
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from keras.api.applications.vgg16 import VGG16, preprocess_input
from keras.api.preprocessing.image import img_to_array
from keras.api.models import Model

# Karakter kodlama hatalarını önlemek için stdout ve stderr'i yeniden yönlendirin
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

# VGG16 Modelini Yükle
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


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


def vgg16_feature_similarity(image1, image2):
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


# KNN için gerekli fonksiyonlar
def euclidean_distance(ORB_train, SSIM_train, VGG16_train, orb_test, ssim_test, vgg16_test, index):
    x_axis = (orb_test[index] - ORB_train) ** 2
    y_axis = (ssim_test[index] - SSIM_train) ** 2
    z_axis = (vgg16_test[index] - VGG16_train) ** 2
    distance = math.sqrt(x_axis + y_axis + z_axis)
    return distance


def rearranged_grade(sorted_list, distance_list, grade_list):
    reArranged = []
    for i in range(len(sorted_list)):
        first_index = 0
        for j in range(len(sorted_list)):
            if sorted_list[i] == distance_list[j]:
                first_index = j
                break
        reArranged.append(grade_list[first_index])
    return reArranged


def weighted_grade(grades, distance, real_grades, k):
    around_grade = 0
    total_diverse_distance = 0
    for i in range(k):
        around_grade += grades[i] * (1 / distance[i])
    for i in range(k):
        total_diverse_distance += (1 / distance[i])
    around_grade = around_grade / total_diverse_distance
    return around_grade


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
        vgg16_score = vgg16_feature_similarity(base_image_path, full_image_path)

        orb_sim_score = float(orb_sim_score)
        ssim_score = float(ssim_score)
        vgg16_score = float(vgg16_score)

        # ANN modelini çalıştır ve tahmini al
        ann_input = np.array([[orb_sim_score, ssim_score, vgg16_score]], dtype=np.float32)
        ann_input = scaler.transform(ann_input)
        predicted_grade = ann_model.predict(ann_input)[0][0]
        predicted_grade = float(predicted_grade)

        # Weighted Grade Hesapla
        k = 3  # k değerini belirleyin
        distances = []
        for i in range(len(orb_list)):
            distances.append(euclidean_distance(orb_list[i], ssim_list[i], vgg16_list[i], [orb_sim_score], [ssim_score],
                                                [vgg16_score], 0))
        sorted_distances = sorted(distances)
        sorted_grades = rearranged_grade(sorted_distances, distances, grade_list)
        second_predicted_grade = weighted_grade(sorted_grades, sorted_distances, grade_list, k)

        current_app.logger.info(
            f"ORB Score: {orb_sim_score}, SSIM Score: {ssim_score}, VGG16 Score: {vgg16_score}, Predicted Grade: {predicted_grade}, Second Predicted Grade: {second_predicted_grade}")

        return jsonify({
            'orb': orb_sim_score,
            'ssim': ssim_score,
            'vgg16': vgg16_score,
            'predicted_grade': predicted_grade,
            'second_predicted_grade': second_predicted_grade
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
        vgg16_score = vgg16_feature_similarity(base_image_path, temp_image_path)

        orb_sim_score = float(orb_sim_score)
        ssim_score = float(ssim_score)
        vgg16_score = float(vgg16_score)

        # ANN modelini çalıştır ve tahmini al
        ann_input = np.array([[orb_sim_score, ssim_score, vgg16_score]], dtype=np.float32)
        ann_input = scaler.transform(ann_input)
        predicted_grade = ann_model.predict(ann_input)[0][0]
        predicted_grade = float(predicted_grade)

        # Weighted Grade Hesapla
        k = 3  # k değerini belirleyin
        distances = []
        for i in range(len(orb_list)):
            distances.append(euclidean_distance(orb_list[i], ssim_list[i], vgg16_list[i], [orb_sim_score], [ssim_score],
                                                [vgg16_score], 0))
        sorted_distances = sorted(distances)
        sorted_grades = rearranged_grade(sorted_distances, distances, grade_list)
        second_predicted_grade = weighted_grade(sorted_grades, sorted_distances, grade_list, k)

        current_app.logger.info(
            f"ORB Score: {orb_sim_score}, SSIM Score: {ssim_score}, VGG16 Score: {vgg16_score}, Predicted Grade: {predicted_grade}, Second Predicted Grade: {second_predicted_grade}")

        os.remove(temp_image_path)

        return jsonify({
            'orb': orb_sim_score,
            'ssim': ssim_score,
            'vgg16': vgg16_score,
            'predicted_grade': predicted_grade,
            'second_predicted_grade': second_predicted_grade
        })

    except Exception as e:
        current_app.logger.error(f"Error processing uploaded image: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # CSV'den verileri oku
    orb_list = []
    ssim_list = []
    vgg16_list = []
    grade_list = []

    with open('train_data.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            orb_list.append(float(row['ORB'].strip()))
            ssim_list.append(float(row['SSIM'].strip()))
            vgg16_list.append(float(row['VGG16'].strip()))
            grade_list.append(int(row['GRADED'].strip()))

    # ANN Modelini yükle ve eğit
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # Veriyi oku ve ön işle
    train_data_path = 'train_data.csv'
    train_df = pd.read_csv(train_data_path, delimiter=';', header=None, skiprows=1,
                           names=['ORB', 'SSIM', 'VGG16', 'GRADED'])

    # Özellikleri normalize et
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[['ORB', 'SSIM', 'VGG16']])
    y_train = train_df['GRADED'].values

    # Modeli tanımla ve derle
    ann_model = Sequential([
        Dense(10, input_dim=3, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')
    ])
    ann_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Erken durdurma kullanarak modeli eğit
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ann_model.fit(X_train, y_train, epochs=200, validation_split=0.3, callbacks=[early_stopping], verbose=1)

    app.run(debug=True)
