from pathlib import Path
from src.download import Download
from src.dataAugmentation import DataAugmentation
import os
from src.preproceso import Preproceso

# PARAMETERS
#ROOT_PATH = Path("C:\\Users\\agarc\\OneDrive\\Documentos\\GitHub\\Plant_Disease_Detection_for_ML")
ROOT_PATH = os.path.abspath("")

DOWNLOAD_DATA = False
DOWNLOAD_PATH = Path("artifacts/sourceData")

AUGMENT_DATA = False
SOURCE_PATH = Path("artifacts\\sourceData")
FINAL_PATH = Path("artifacts\\workData")

PREPROCESO = True
NEDD_PATH = Path("artifacts\\workData\\images")
NEW_PATH = Path("artifacts\\preprocesamiento")

# DOWNLOAD DATA
if DOWNLOAD_DATA:
    download = Download(DOWNLOAD_PATH)
    download.main()

os.chdir(ROOT_PATH)  # Cambia el directorio de trabajo al directorio raíz del proyecto

# DATA AUGMENTATION
if AUGMENT_DATA:
    data_augmentation = DataAugmentation(SOURCE_PATH, FINAL_PATH)
    data_augmentation.main()

# PREPROCESO
if PREPROCESO:
    preproceso = Preproceso(NEDD_PATH, NEW_PATH, 150)
    preproceso.main()

print("Proceso completado.")


    

