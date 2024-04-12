class Download :
    def __init__(self, download_data: bool) -> None:
        self.download_data = download_data

    def main(self) :
        if self.download_data :
            self.download()
        print('Data downloaded!')

    def download(self) :
        import os
        import zipfile

        os.chdir('artifacts')
        os.system("kaggle competitions download -c plant-pathology-2020-fgvc7")
        with zipfile.ZipFile("plant-pathology-2020-fgvc7.zip") as zip :
            zip.extractall()
        os.chdir('../')
