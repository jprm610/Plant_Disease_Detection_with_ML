import os

class Setup :
    @classmethod
    def create_directory_structure(cls):
        print("Setting up project...")

        # Create the main directory
        main_dir = "artifacts"
        os.makedirs(main_dir, exist_ok=True)
        
        # Create subdirectories
        subdirectories = ["data/images", "preprocesamiento"]
        for subdir in subdirectories:
            os.makedirs(os.path.join(main_dir, subdir), exist_ok=True)

        print("Setup complete!")
