import cv2
import numpy as np

class MyImages :
    MODES = ['RGB', 'YUV', 'HSV', 'LAB', 'HLS', 'XYZ', 'YCRCB', 'CMY', 'YIQ']

    @classmethod
    def img_read(cls, filename, color_space='RGB') :
        """
        Leer una imagen con un espacio de color especificado.

        Input:
            - filename (str): Ruta de la imagen
            - color_space (str): Espacio de color de la imagen (RGB, GRAY, YUV, HSV, LAB, HLS, XYZ, YCrCb, YIQ, CMY) (default: 'RGB')

        Output
            - img (numpy.ndarray): Matriz de la imagen obtenida en el espacio de color deseado
        """
        color_space = color_space.upper()
        if color_space == 'RGB' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        elif color_space == 'GRAY' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        elif color_space == 'YUV' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2YUV)
        elif color_space == 'HSV' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2LAB)
        elif color_space == 'HLS' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HLS)
        elif color_space == 'XYZ' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2XYZ)
        elif color_space == 'YCRCB' :
            return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2YCrCb)
        elif color_space == 'CMY' :
            img = cls.img_read(filename)
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            new_img = np.zeros(img.shape, np.uint8)

            C = 255 - R
            M = 255 - G
            Y = 255 - B

            new_img[:, :, 0] = C
            new_img[:, :, 1] = M
            new_img[:, :, 2] = Y

            return new_img
        elif color_space == 'YIQ' :
            img = cls.img_read(filename)
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            new_img = np.zeros(img.shape, np.uint8)

            Y = 0.299*R + 0.587*G + 0.114*B
            I = 0.596*R - 0.274*G - 0.322*B
            Q = 0.211*R - 0.523*G + 0.312*B

            new_img[:, :, 0] = Y
            new_img[:, :, 1] = I
            new_img[:, :, 2] = Q

            return new_img
        else :
            raise Exception('INPUT ERROR: Espacio de color incorrecto')

    @classmethod
    def convert_image(cls, img, color_space='GRAY') :
        """
        Convertir una imagen en RGB al espacio de color especificado.

        Input:
            - img (numpy.ndarray): Matriz de la imagen en RGB
            - color_space (str): Espacio de color al que se quiera convertir la imagen (RGB, GRAY, YUV, HSV, LAB, HLS, XYZ, YCrCb, YIQ, CMY) (default: 'GRAY')

        Output
            - img (numpy.ndarray): Matriz de la imagen obtenida en el espacio de color deseado
        """
        color_space = color_space.upper()
        if color_space == 'RGB' :
            return img
        elif color_space == 'GRAY' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif color_space == 'YUV' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'HSV' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LAB' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif color_space == 'HLS' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'XYZ' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
        elif color_space == 'YCRCB' :
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'CMY' :
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            new_img = np.zeros(img.shape, np.uint8)

            C = 255 - R
            M = 255 - G
            Y = 255 - B

            new_img[:, :, 0] = C
            new_img[:, :, 1] = M
            new_img[:, :, 2] = Y

            return new_img
        elif color_space == 'YIQ' :
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            new_img = np.zeros(img.shape, np.uint8)

            Y = 0.299*R + 0.587*G + 0.114*B
            I = 0.596*R - 0.274*G - 0.322*B
            Q = 0.211*R - 0.523*G + 0.312*B

            new_img[:, :, 0] = Y
            new_img[:, :, 1] = I
            new_img[:, :, 2] = Q

            return new_img
        else :
            raise Exception('INPUT ERROR: Espacio de color incorrecto')
    
    @classmethod
    def threshold_v_channel_inverted(images, threshold=155):
        thresholded_images = []
        for img in images:
            if img is not None:
                # Aplicar umbralización binaria inversa
                _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
                thresholded_images.append(thresh_img)
        return thresholded_images
        