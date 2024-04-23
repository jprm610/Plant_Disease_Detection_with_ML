from pathlib import Path

class Download :
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def main(self) :
        self.download(self.source_path)
        print('Data downloaded!')

    def download(self, download_path: Path) :
        import os
        import zipfile

        import os

        # Relative path to a file or directory
        relative_path = ".kaggle"
        # Get the absolute path
        kaggle_path = os.path.abspath(relative_path)
        # Set the environment variable
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_path

        os.chdir(download_path)
        os.system("kaggle competitions download -c plant-pathology-2020-fgvc7")
        with zipfile.ZipFile("plant-pathology-2020-fgvc7.zip") as zip :
            zip.extractall()
        os.chdir('../')
