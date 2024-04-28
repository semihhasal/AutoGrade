import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.vgg16 import VGG16, preprocpre_process_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

def orb_sim(img1, img2):
    """ ORB benzerliğini hesaplar """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
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

def vgg16_feature_similarity(image1, image2):
    """ İki görüntünün VGG16 özellik benzerliğini hesaplar """
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    feat1 = model.predict(img1)[0]
    feat2 = model.predict(img2)[0]
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity

# VGG16 Modelini Yükle
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


# Diğer kütüphane importları ve fonksiyon tanımları aynı kalacak

def compare_images(base_image, comparison_image, comparison_image_path):
    """ İki görüntü arasındaki ORB, SSIM ve VGG16 benzerliğini hesaplar ve yazdırır """
    # Görüntüleri aynı boyuta getir
    comparison_image_resized = cv2.resize(comparison_image, (base_image.shape[1], base_image.shape[0]))

    orb_similarity = orb_sim(base_image, comparison_image_resized)
    ssim_similarity = structural_sim(base_image, comparison_image_resized)
    vgg_similarity = vgg16_feature_similarity("c1.jpg", comparison_image_path)

    print(f"ORB Similarity: {orb_similarity}")
    print(f"SSIM Similarity: {ssim_similarity}")
    print(f"VGG16 Similarity: {vgg_similarity}")


# Görüntüleri okuma ve döngü aynı kalacak

# Görüntüleri oku
image_c1 = cv2.imread("c1.jpg", 0)  # Gri tonlama için 0

# Karşılaştırmak için resim isimlerini bir liste olarak sakla
comparison_images = ["o1.jpg", "o2.jpg", "o3.jpg"]

# Her bir karşılaştırma resmi için döngü
for img_name in comparison_images:
    if os.path.exists(img_name):
        comparison_image = cv2.imread(img_name, 0)
        print(f"\nComparing c1 and {img_name}:")
        compare_images(image_c1, comparison_image, img_name)
    else:
        print(f"Resim bulunamadı: {img_name}")
