from pathlib import Path
from src.download import Download
from src.dataAugmentation import DataAugmentation

# PARAMETERS
DOWNLOAD_DATA = True
DOWNLOAD_PATH = Path("artifacts/sourceData")

AUGMENT_DATA = True
SOURCE_PATH = Path("artifacts/sourceData")
FINAL_PATH = Path("artifacts/workData")

# DOWNLOAD DATA
if DOWNLOAD_DATA :
    download = Download(DOWNLOAD_PATH)
    download.main()

# DATA AUGMENTATION
if AUGMENT_DATA :
    data_augmentation = DataAugmentation(SOURCE_PATH, FINAL_PATH)
    data_augmentation.main()
