import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import classification_report

class AugmentedDatasetEvaluator:
    def __init__(self, model_path, original_dataset_path, csv_path, augmented_dataset_size):
        self.model_path = Path(model_path)
        self.original_dataset_path = Path(original_dataset_path)
        self.csv_path = Path(csv_path)
        self.augmented_dataset_size = augmented_dataset_size
        self.model = None
        self.datagen = None
        self.augmented_dataset = None
        
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"El modelo no se encuentra en {self.model_path}")
        self.model = load_model(self.model_path)
        
    def create_augmented_dataset(self):
        if not self.original_dataset_path.exists():
            raise FileNotFoundError(f"El directorio de datos originales no se encuentra en {self.original_dataset_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"El archivo CSV no se encuentra en {self.csv_path}")
        
        # Configurar el generador de datos aumentados
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            zoom_range=0.05
        )
        
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Contenido del CSV:\n{df.head()}")
            print(f"Columnas del CSV: {df.columns}")
            
            # Verificar que las columnas necesarias existen
            if 'image_id' not in df.columns or 'label' not in df.columns:
                raise ValueError("El CSV no contiene las columnas 'image_id' y 'label' necesarias")
            
            # Usar el DataFrame para crear el generador de datos
            self.augmented_dataset = self.datagen.flow_from_dataframe(
                dataframe=df,
                directory=self.original_dataset_path,
                x_col="image_id",
                y_col="label",
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical'
            )
        except Exception as e:
            print(f"Error al leer el archivo CSV o crear el generador de datos: {e}")
            raise
        
    def evaluate_model(self):
        if self.model is None:
            self.load_model()
        
        if self.augmented_dataset is None:
            self.create_augmented_dataset()
        
        if self.augmented_dataset.samples == 0:
            raise ValueError("No se encontraron imágenes en el conjunto de datos")
        
        # Evaluar el modelo en el conjunto de datos aumentado
        results = self.model.evaluate(self.augmented_dataset, steps=self.augmented_dataset_size // 32)
        
        # Generar reporte de clasificación
        y_true = []
        y_pred = []
        for _ in range(self.augmented_dataset_size // self.augmented_dataset.batch_size):
            x, y = next(self.augmented_dataset)
            predictions = self.model.predict(x)
            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))

        report = classification_report(y_true, y_pred)
        
        return {metric: value for metric, value in zip(self.model.metrics_names, results)}, report
    
    def run(self):
        try:
            print(f"Ruta absoluta del modelo: {self.model_path.absolute()}")
            print(f"Ruta absoluta del conjunto de datos: {self.original_dataset_path.absolute()}")
            print(f"Ruta absoluta del archivo CSV: {self.csv_path.absolute()}")
            
            self.load_model()
            self.create_augmented_dataset()
            evaluation_results, classification_report = self.evaluate_model()
            
            print("Resultados de la evaluación en el conjunto de datos aumentado:")
            for metric, value in evaluation_results.items():
                print(f"{metric}: {value}")
                
            print("Reporte de clasificación:")
            print(classification_report)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error de valor: {e}")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
            print(f"Tipo de error: {type(e)}")
            print(f"Detalles adicionales: {e.args}")

# Uso del módulo
evaluator = AugmentedDatasetEvaluator(
    model_path='artifacts\\modelcandi\\model.keras',
    original_dataset_path='artifacts\\data\\images',
    csv_path='artifacts\\data\\df.csv',
    augmented_dataset_size=10000  # Ajuste según el tamaño deseado del conjunto aumentado
)

evaluator.run()