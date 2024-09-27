import os

class Setup :
    @classmethod
    def create_directory_structure(cls):
        print("Setting up project...")

        os.system("pip install -r requirements.txt")

        # Create the main directory
        try :
            main_dir = "artifacts"
            os.makedirs(main_dir, exist_ok=True)
            
            # Create subdirectories
            subdirectories = ["data/images", "model"]
            for subdir in subdirectories:
                os.makedirs(os.path.join(main_dir, subdir), exist_ok=True)
        except :
            pass

        print("Setup complete!")
