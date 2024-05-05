import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def orb_similarity(img1, img2):
    """ ORB benzerliğini hesaplar """
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
    """ SSIM benzerliğini hesaplar """
    sim, _ = ssim(img1, img2, full=True)
    return sim


def preprocess_image(image_path):
    """ Görüntüyü VGG16 modeline uygun şekilde işler """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def vgg16_feature_similarity(model, image1, image2):
    """ İki görüntünün VGG16 özellik benzerliğini hesaplar """
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    feat1 = model.predict(img1)[0]
    feat2 = model.predict(img2)[0]
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity


# VGG16 Modelini Yükle
base_model = VGG16(weights='imagenet')
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


def compare_images(model, base_image, comparison_image, comparison_image_path):
    """ İki görüntü arasındaki ORB, SSIM ve VGG16 benzerliğini hesaplar """
    comparison_image_resized = cv2.resize(comparison_image, (base_image.shape[1], base_image.shape[0]))

    orb_sim_score = orb_similarity(base_image, comparison_image_resized)
    ssim_score = structural_sim(base_image, comparison_image_resized)
    vgg16_score = vgg16_feature_similarity(model, "c1.jpg", comparison_image_path)

    return orb_sim_score, ssim_score, vgg16_score


# Görüntüleri oku
image_c1 = cv2.imread("c1.jpg", 0)

# Sonuçları saklayacak DataFrame oluştur
results = pd.DataFrame(columns=['Image', 'ORB', 'SSIM', 'VGG16'])

# Karşılaştırma resimlerini işle
for i in range(1, 111):
    img_name = f"images/o{i}.jpg"
    if os.path.exists(img_name):
        comparison_image = cv2.imread(img_name, 0)
        orb_sim, ssim_sim, vgg_sim = compare_images(model, image_c1, comparison_image, img_name)
        results.loc[len(results)] = [img_name, orb_sim, ssim_sim, vgg_sim]
        print(f"Image: {img_name}, ORB: {orb_sim}, SSIM: {ssim_sim}, VGG16: {vgg_sim}")

# Sonuçları Excel dosyasına kaydet
results.to_excel('comparison_results.xlsx', index=False)
