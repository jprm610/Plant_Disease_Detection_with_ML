from parameters import Parameters
from src.setup import Setup
from src.download import Download
from src.dataAugmentation import DataAugmentation
from src.deep_learning import DeepLearning

if __name__ == "__main__" :

    # SETUP PROJECT
    if Parameters.FIRST_RUN :
        Setup().create_directory_structure()

    # DOWNLOAD DATA
    if Parameters.DOWNLOAD["EXECUTE"] :
        download = Download(**Parameters.DOWNLOAD["PARAMETERS"])
        download.main()

    # DATA AUGMENTATION
    if Parameters.DATA_AUGMENTATION["EXECUTE"] :
        dataAugmentation = DataAugmentation(**Parameters.DATA_AUGMENTATION["PARAMETERS"])
        dataAugmentation.main()

    # DEEP_LEARNING
    if Parameters.DEEP_LEARNING["EXECUTE"] :
        deepLearning = DeepLearning(**Parameters.DEEP_LEARNING["PARAMETERS"])
        deepLearning.main()











