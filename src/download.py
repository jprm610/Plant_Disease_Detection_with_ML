import os
import fsspec
from pathlib import Path
import pandas as pd

class Download :
    def __init__(self, github_user: str, github_repo: str, target_repo_folder: Path, destination: Path,
                 labels: list[str]) -> None:
        self.github_user = github_user
        self.github_repo = github_repo
        self.target_folder = target_repo_folder
        self.destination = destination
        self.labels = labels

    def main(self) -> None :
        print("Downloading images...")
        # Crate an empty dataframe in order to save each image and it's label.
        df = pd.DataFrame(columns=['image_id', 'label'])

        for label in self.labels :
            print(f"\tDownloading images for Apple___{label}")
            target_folder = self.target_folder / f"Apple___{label}"
            destination = self.destination / label
            self.download(target_folder, destination)
            df = self.buildDataframe(df=df, destination=destination, label=label)
            print("\tComplete!")
        
        df = df.reset_index()
        df.to_csv(f"{self.destination.parent}/df.csv")
        print("Download complete!")

    def download(self, target_folder: Path, destination: Path) -> None :
        """
        Download a folder from a github repository.
        """
        fs = fsspec.filesystem("github", org=self.github_user, repo=self.github_repo)
        fs.get(fs.ls(target_folder.as_posix()), destination.as_posix())

    def buildDataframe(self, df: pd.DataFrame, destination: Path, label: Path) -> pd.DataFrame :
        """
        Build the dataframe from the downloaded images.
        """

        from shutil import rmtree

        rows = []
        for image_id in os.listdir(destination) :
            row = {
                'image_id' : image_id,
                'label' : label
            }
            rows.append(row)
            image_route = destination / Path(image_id)
            image_route.rename(self.destination / image_id)
        df_to_append = pd.DataFrame(rows)
        df = pd.concat([df, df_to_append], ignore_index=True)
        rmtree(destination)

        return df
