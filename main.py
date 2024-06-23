import os
from pathlib import Path
from setup import Setup
from src.download import Download
from src.dataAugmentation import DataAugmentation
from src.preproceso import Preproceso

# PARAMETERS
#ROOT_PATH = Path("C:\\Users\\agarc\\OneDrive\\Documentos\\GitHub\\Plant_Disease_Detection_for_ML")
ROOT_PATH = os.path.abspath("")

FIRST_RUN = False

DOWNLOAD = {
    "DOWNLOAD_DATA": True,
    "PARAMETERS": {
        "github_user": "spMohanty",
        "github_repo": "PlantVillage-Dataset",
        "target_repo_folder": Path("raw/color"),
        "destination": Path("artifacts/data/images"),
        "labels": ["Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"]
    }
}

AUGMENT_DATA = False
SOURCE_PATH = Path("artifacts\\sourceData")
FINAL_PATH = Path("artifacts\\workData")

PREPROCESO = False
NEDD_PATH=Path("artifacts\\workData\\images")
NEW_PATH = Path("artifacts\\preprocesamiento")

if __name__ == "__main__" :
    if FIRST_RUN :
        Setup().create_directory_structure()

    # DOWNLOAD DATA
    if DOWNLOAD["DOWNLOAD_DATA"] :
        download = Download(**DOWNLOAD["PARAMETERS"])
        download.main()

    os.chdir(ROOT_PATH)  # Cambia el directorio de trabajo al directorio raíz del proyecto

    # DATA AUGMENTATION
    if AUGMENT_DATA:
        data_augmentation = DataAugmentation(SOURCE_PATH, FINAL_PATH)
        data_augmentation.main()

    # PREPROCESO
    if PREPROCESO:
        preproceso_n = Preproceso(NEDD_PATH, NEW_PATH, 150)
        preproceso_n.main()

    print("Proceso completado.")


    

