import numpy as np
import pandas as pd
from uuid import uuid4
from parameters import Parameters
from pathlib import Path
from random import seed, choice
from PIL import Image, ImageEnhance, ImageOps

class DataAugmentation :
    def __init__(self, source_path: Path, goal_amount_per_label=0) -> None:
        self.source_path = source_path
        self.goal_amount_per_label = goal_amount_per_label
        self.df = pd.read_csv(f"{self.source_path.parent}/df.csv")

    def main(self) :
        print("Augmenting images...")
        amount_per_label = {
            label: len(self.df[self.df['label'] == label]) for label in Parameters.DOWNLOAD["PARAMETERS"]['labels']
        }

        # In case that the user doesn't specify a goal amount or it is less than the maximum, 
        # all labels will be balanced to the most populated one.
        if self.goal_amount_per_label < max(amount_per_label.values()) :
            self.goal_amount_per_label = max(amount_per_label.values())

        rows = []
        for label in Parameters.DOWNLOAD["PARAMETERS"]['labels'] :
            print(f"\tAugmenting for {label}...")
            while amount_per_label[label] < self.goal_amount_per_label :
                seed(self.goal_amount_per_label)
                source_img_id = choice(list(self.df['image_id'][self.df['label'] == label]))
                with Image.open(f"{self.source_path}/{source_img_id}") as img :
                    new_img = self.random_transformation(img)
                    new_img_id = f"{uuid4()}.JPG"
                    new_img_path = f"{self.source_path}/{new_img_id}"
                    new_img.save(new_img_path)

                row = {'image_id' : new_img_id, 'label': label}
                rows.append(row)

                amount_per_label[label] += 1
            print("\tComplete!")
        
        df_to_append = pd.DataFrame(rows)
        self.df = pd.concat([self.df, df_to_append], ignore_index=True)
        self.df = self.df.reset_index()
        self.df.to_csv(f"{self.source_path.parent}/df.csv")
        print("Data augmentation complete!")
        return
    
    def random_transformation(self, image: Image) -> Image :
        import random
        """Apply a single random transformation to an image."""
        transformations = [
            ImageOps.mirror,  # Horizontal Flip
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.5, 1.5)),  # Contrast Adjustment
            lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.5, 1.5)),  # Brightness Adjustment
            lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.5, 1.5)),  # Color Jitter
            lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.5, 2.0)),  # Sharpness Enhancement
            self.add_gaussian_noise  # Gaussian Noise
        ]

        # Apply a single random transformation
        transformation = random.choice(transformations)
        return transformation(image)

    def add_gaussian_noise(self, image):
        import random
        """Add Gaussian noise to an image."""
        np_image = np.array(image)
        row, col, ch = np_image.shape
        mean = 0
        sigma = random.uniform(1, 25)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch).astype('uint8')
        noisy = np_image + gauss
        return Image.fromarray(np.clip(noisy, 0, 255).astype('uint8'))
    