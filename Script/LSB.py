from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd






image : Image.Image


def LSB(image_path,message):
    image = image_path
    if image.mode == "RGB":
        img = np.array(image)  # Image couleur
        color = True
    else:
        img = np.array(image.convert("L"))  # Conversion en niveaux de gris
        color = False
    W, H = img.shape[:2]
    message = message.encode("ascii")  # Conversion en ASCII
    message_length = len(message)
    message_bits = f"{message_length:016b}" + ''.join([format(byte, '08b') for byte in message])
    img_flat = img.flatten()
    if len(message_bits) > len(img_flat):
        raise ValueError("Le message est trop long pour être caché dans l'image !")
    modified_pixels = []

    for i, bit in enumerate(message_bits):
        original_pixel = img_flat[i]
        new_pixel = (img_flat[i] & 0b11111110) | int(bit)
        img_flat[i] = new_pixel
        modified_pixels.append([i, original_pixel, new_pixel, bit])


    # **3. Sauvegarde de l'image encodée**
    return img_flat
def reformat(img_flat,image):
    if image.mode == "RGB":
        img = np.array(image)  # Image couleur
        color = True
    else:
        img = np.array(image.convert("L"))  # Conversion en niveaux de gris
        color = False
    img_encoded = img_flat.reshape(img.shape)
    encoded_image = Image.fromarray(img_encoded)
    return encoded_image


def decode(img_flat):
    idx = 0
    message_length_bits = ''.join(str(img_flat[i] & 1) for i in range(16))
    message_length = int(message_length_bits, 2)
    msg = "".join(chr(int(''.join(str(img_flat[i] & 1) for i in range(idx, idx + 8)), 2)) for idx in range(16, 16 + message_length * 8, 8))
    return msg


def calculate_psnr(original, modified):
    original1 = np.array(original, dtype=np.float32)
    modified1 = np.array(modified, dtype=np.float32)
    mse = np.mean((original1 - modified1) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
