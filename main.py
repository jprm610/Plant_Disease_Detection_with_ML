from pathlib import Path
from src.download import Download
from src.dataAugmentation import DataAugmentation

# PARAMETERS
DOWNLOAD_DATA = False
DF_PATH = Path("artifacts/train.csv")
IMGS_PATH = Path("artifacts/images")

# DOWNLOAD DATA
download = Download(DOWNLOAD_DATA)
download.main()

# DATA AUGMENTATION
data_augmentation = DataAugmentation(DF_PATH, IMGS_PATH)
data_augmentation.main()
