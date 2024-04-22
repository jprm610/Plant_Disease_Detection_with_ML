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

        os.chdir(download_path)
        os.system("kaggle competitions download -c plant-pathology-2020-fgvc7")
        with zipfile.ZipFile("plant-pathology-2020-fgvc7.zip") as zip :
            zip.extractall()
        os.chdir('../')
