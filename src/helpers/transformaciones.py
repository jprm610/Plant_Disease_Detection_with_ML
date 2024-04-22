import cv2
import numpy as np

class Transformaciones :
    @classmethod
    def apply_f(cls, img, f, args):
        #Crear una matriz de ceros del tamaño de la imagen de entrada
        res = np.zeros(img.shape, np.uint8)
        #Aplicar la transformación f sobre cada canal del espacio de color RGB
        res[:,:,0] = f(img[:,:,0], *args)
        res[:,:,1] = f(img[:,:,1], *args)
        res[:,:,2] = f(img[:,:,2], *args)
        
        return res

    @classmethod
    def cuadraticTransform(cls, img, a, b ,c):
        img_copy = img.copy().astype(np.float32)/255.0
        res_a = cv2.pow(img_copy, 2)
        res_a = cv2.multiply(res_a, a)
        res_b = cv2.multiply(img_copy, b)
        res = cv2.add(res_a, res_b)
        res = cv2.add(res, c)
        
        res[res < 0] = 0
        res = res * 255
        res[res > 255] = 255
        res = res.astype(np.uint8)
        
        return res

    @classmethod
    def rootTransform(cls, img, a, b): 
        img_copy = img.astype(np.float32)/255.0
        res_a = cv2.pow(img_copy, 0.5)
        res_a = cv2.multiply(res_a, a)
        res = cv2.add(res_a, b)

        res[res < 0] = 0
        res = res * 255
        res[res > 255] = 255
        res = res.astype(np.uint8)
        
        return res

    @classmethod
    def gammaCorrection(cls, img, a, gamma):
        
        img_copy = img.copy().astype(np.float32)/255.0
        res_gamma = cv2.pow(img_copy,gamma)
        res = cv2.multiply(res_gamma, a)
        
        res[res<0] = 0
        res = res*255.0
        res[res>255] = 255
        res = res.astype(np.uint8)
        
        return res

    @classmethod
    def histogram_expansion(cls, img):
        
        #Crear matriz de ceros del tamaño de la imagen y tipo de dato flotante
        res = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
        
        #Extraer el mínimo y el máximo del conjunto de datos
        m = float(np.min(img))
        M = float(np.max(img))
        #Aplicar la función de expansión(normalización) y asegurar datos uint8
        res = (img-m)*255.0/(M-m)
        res = res.astype(np.uint8)
        
        return res

    @classmethod
    def clahe_f(cls, img, a=5, x=8, y=8) :
        clahe = cv2.createCLAHE(clipLimit=a, tileGridSize=(x,y))
        img = cls.apply_f(img, clahe.apply, [])
        
        return img