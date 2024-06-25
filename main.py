import os
from pathlib import Path
from parameters import Parameters
from src.setup import Setup
from src.download import Download
from src.dataAugmentation import DataAugmentation

if __name__ == "__main__" :

    # SETUP PROJECT
    if Parameters.FIRST_RUN :
        Setup().create_directory_structure()

    # DOWNLOAD DATA
    if Parameters.DOWNLOAD["EXECUTE"] :
        download = Download(**Parameters.DOWNLOAD["PARAMETERS"])
        download.main()

    if Parameters.DATA_AUGMENTATION["EXECUTE"] :
        dataAugmentation = DataAugmentation(**Parameters.DATA_AUGMENTATION["PARAMETERS"])
        dataAugmentation.main()
