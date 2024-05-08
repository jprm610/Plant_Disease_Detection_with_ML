import numpy as np
import pandas as pd
from pathlib import Path
from random import seed, choice
from PIL import Image, ImageEnhance, ImageOps

class DataAugmentation :
    def __init__(self, source_path: Path, final_path: Path) -> None:
        self.df = pd.read_csv(source_path / Path("train.csv"))
        self.source_path = source_path
        self.final_path = final_path
        self.target_size = (2048, 1365)

    def main(self) :
        self.df = self.Copy_Images(self.source_path, self.final_path)

        separated_dfs = {
            "healthy" : self.df[self.df.healthy == 1],
            "multiple_diseases" : self.df[self.df.multiple_diseases == 1],
            "rust" : self.df[self.df.rust == 1],
            "scab" : self.df[self.df.scab == 1]
        }

        goal_amount = max([len(df) for df in separated_dfs.values()])

        last_index = len(self.df)
        for col in self.df.columns[1:] :
            iterations = goal_amount - len(separated_dfs[col])
            while iterations > 0 :
                seed(iterations)
                img_id = choice(list(separated_dfs[col]['image_id']))
                with Image.open(f"{self.final_path}/images/{img_id}.jpg") as img :
                    new_img = self.random_transformation(img)
                    new_img = new_img.resize(self.target_size)
                    new_img_id = f"img_{last_index}"
                    new_img_path = f"{self.final_path}/images/{new_img_id}.jpg"
                    new_img.save(new_img_path)

                row = {'image_id': new_img_id, 'healthy': 0, 'multiple_diseases': 0, 'rust': 0, 'scab': 0}
                row[col] = 1
                
                new_row_df = pd.DataFrame([row])

                # Use concat to add the new row to the DataFrame
                self.df = pd.concat([self.df, new_row_df], ignore_index=True)

                last_index += 1
                iterations -= 1

        self.df.to_csv(f"{self.final_path}/df.csv")
        print("Data augmentation done!")
        return
    
    def Copy_Images(self, sourcePath: Path, finalPath: Path) :
        """
        Only copy train images, which are already labeled, in the new folder,
        and change base dataframe.
        """

        new_df = self.df.copy()
        for i in range(len(self.df)) :
            new_df.loc[i, 'image_id'] = f"img_{i}"
            with Image.open(f"{sourcePath}/images/Train_{i}.jpg") as img :
                if img.size != self.target_size :
                    img = img.rotate(90)
                img.save(f"{finalPath}/images/img_{i}.jpg")

        new_df.to_csv(f"{finalPath}/df.csv")
        print("\tCopied base images!")
        return new_df
    
    def random_transformation(self, image: Image) -> Image :
        import random
        """Apply a single random transformation to an image."""
        transformations = [
            ImageOps.mirror,                                          # Horizontal Flip
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.5, 1.5)),  # Contrast Adjustment
            lambda x: x.resize((int(x.width * random.uniform(0.7, 1.3)), int(x.height * random.uniform(0.7, 1.3)))),  # Scaling
            ImageOps.flip,                                            # Vertical Flip
            lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.5, 1.5)),  # Brightness Adjustment
            lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.5, 1.5)),  # Color Jitter
            lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.5, 2.0)),  # Sharpness Enhancement
            lambda x: self.add_gaussian_noise(x),                          # Gaussian Noise
            lambda x: self.crop_and_resize(x)                              # Crop and Resize
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

    def crop_and_resize(self, image):
        import random
        """Crop the image randomly and resize it back to original dimensions."""
        original_size = image.size
        left = random.randint(0, original_size[0] // 4)
        top = random.randint(0, original_size[1] // 4)
        right = random.randint(3 * original_size[0] // 4, original_size[0])
        bottom = random.randint(3 * original_size[1] // 4, original_size[1])
        image = image.crop((left, top, right, bottom))
        return image.resize(original_size)
    