import os

def create_directory_structure():
    # Create the main directory
    main_dir = "artifacts"
    os.makedirs(main_dir, exist_ok=True)
    
    # Create subdirectories
    subdirectories = ["sourceData", "workData/images", "preprocesamiento"]
    for subdir in subdirectories:
        os.makedirs(os.path.join(main_dir, subdir), exist_ok=True)

if __name__ == "__main__":
    os.system("pip install -r requirements.txt")

    create_directory_structure()
