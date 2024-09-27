# Detección Automática de Enfermedades en Hojas de Manzano Utilizando Redes Neuronales Convolucionales
Desarrollado por Andrés Felipe García y Juan Pablo Robledo

## Descripción
Este proyecto desarrolla y despliega un modelo de red neuronal convolucional (CNN) para la identificación automática de enfermedades en hojas de manzano. Utilizando TensorFlow y Keras, el modelo clasifica las imágenes de hojas en cuatro categorías distintas: hojas sanas, y hojas afectadas por Apple Scab (Venturia inaequalis), Apple Black Rot (Botryosphaeria obtusa), y Apple Cedar Rust (Gymnosporangium juniperi-virginianae). El objetivo es facilitar la detección temprana y precisa de enfermedades, reduciendo las pérdidas económicas y mejorando la gestión de cultivos a gran escala a través de tecnologías de visión por computadora.

## Instalación
- Clonar el repositorio:
    git clone https://github.com/jprm610/Plant_Disease_Detection_with_ML.git

## Descripción general del directorio
- research: Contiene los jupyter notebooks en los cuales se realizaron pruebas antes de escribir el código en forma modular.
- src: Este el modulo principal del proyecto en donde se encuntran las clases y métodos usados en todo el proceso.
- artifacts: Esta carpeta no existe aún en el repositorio ya que se creará dinámicamente. Aquí se descargarán las imágenes y se exportará el modelo y sus resultados.
- parameters.py: Archivo que contiene todos los hiperparámetros modificables en el modelo.
- main.py: El archivo principal desde el cual se ejecuta el proyecto de acuerdo a los parámetros definidos en parameters.py

## Configuración

El archivo `parameters.py` contiene las configuraciones necesarias para personalizar la ejecución del proyecto. Aquí se describen las principales secciones y cómo ajustarlas según tus necesidades:

### Descarga de Datos
- `FIRST_RUN`: Indica si es la primera vez que se ejecuta el proyecto. Útil para inicializaciones que solo necesitan ejecutarse una vez.
- `DOWNLOAD`: Configura los parámetros para la descarga de las imágenes desde el repositorio de GitHub. Cambia `github_user`, `github_repo`, y los demás parámetros según necesites.

### Aumento de Datos
- `DATA_AUGMENTATION`: Permite activar o desactivar el aumento de datos y ajustar parámetros como la ruta de origen y la cantidad deseada de imágenes por etiqueta.

### Entrenamiento del Modelo
- `DEEP_LEARNING`: Configura el entrenamiento del modelo con parámetros como la ruta de las imágenes, ruta de exportación del modelo, número de épocas y configuración de early stopping.

Solo se recomienda modificar los parámetros FIRST_RUN y EXECUTE en cada fase, los cuales funcionan simplemente como switches que permiten ejecutar una fase del proyecto.

## Uso

Para iniciar el proceso de entrenamiento y evaluación del modelo, asegúrate de que el archivo `parameters.py` esté configurado correctamente y ejecuta:

python main.py

## Tecnologías Utilizadas
- Python
- TensorFlow
- Keras
  
