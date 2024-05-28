import os
from PIL import Image, ImageFilter
import numpy as np
from rembg import remove

class Preproceso:
    def __init__(self, source_folder, target_folder, threshold=155):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.threshold = threshold

    def main(self):
        if not os.path.exists(self.source_folder):
            print(f"El directorio de origen {self.source_folder} no existe.")
            return
        
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

            # Eliminar el fondo
            image_np = np.array(image)
            image_no_bg = remove(image_np)
            image_no_bg = Image.fromarray(image_no_bg)
            print(f"Fondo eliminado para la imagen {filename}.")

            # Aplicar un filtro gaussiano para suavizar la imagen
            blurred_image = image_no_bg.filter(ImageFilter.GaussianBlur(radius=5))  # Puedes ajustar el radio según sea necesario

            # Convertir a YUV y extraer el canal v (Cb)
            yuv_image = blurred_image.convert('YCbCr')
            yuv_array = np.array(yuv_image)
            y_channel = yuv_array[:, :, 0]
            cb_channel = yuv_array[:, :, 1]
            cr_channel = yuv_array[:, :, 2]

            # Aplicar umbralización en el canal U (Cb)
            processed_cb_channel = 255 * (cb_channel > self.threshold).astype(np.uint8)

            # Combinar los canales Y, Cb procesado, y Cr
            processed_yuv_array = np.stack((y_channel, processed_cb_channel, cr_channel), axis=-1)
            processed_yuv_image = Image.fromarray(processed_yuv_array, 'YCbCr')

            # Convertir de YUV de vuelta a RGB
            processed_rgb_image = processed_yuv_image.convert('RGB')

            # Guardar la imagen resultante
            save_path = os.path.join(self.target_folder, filename)
            processed_rgb_image.save(save_path)
            print(f"Imagen guardada en {save_path}")

        except Exception as e:
            print(f"Error al procesar la imagen {filename}: {e}")


