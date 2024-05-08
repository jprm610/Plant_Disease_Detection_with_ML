import os
from PIL import Image
import numpy as np

class Preproceso:
    def __init__(self, source_folder, target_folder, threshold=155):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.threshold = threshold

    def main(self):
        print(f"Procesando imágenes en {self.source_folder}...")
        for filename in os.listdir(self.source_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Encontrada imagen {filename}, procesando...")
                self.process_image(filename)

    def process_image(self, filename):
        try:
            path = os.path.join(self.source_folder, filename)
            image = Image.open(path)
            print(f"Imagen {filename} abierta correctamente.")

            yuv_image = image.convert('YCbCr')
            yuv_array = np.array(yuv_image)
            u_channel = yuv_array[:, :, 2]
            processed_channel = 255 * (u_channel > self.threshold).astype(np.uint8)
            #processed_channel=u_channel

            processed_image = Image.fromarray(processed_channel)
            save_path = os.path.join(self.target_folder, filename)
            processed_image.save(save_path)
            print(f"Imagen guardada en {save_path}")
        except Exception as e:
            print(f"Error al procesar la imagen {filename}: {e}")
