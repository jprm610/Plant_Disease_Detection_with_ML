from pathlib import Path

class Parameters :
    ROOT_PATH = Path().absolute()

    FIRST_RUN = True

    DOWNLOAD = {
        "EXECUTE": True,
        "PARAMETERS": {
            "github_user": "spMohanty",
            "github_repo": "PlantVillage-Dataset",
            "target_repo_folder": Path("raw/color"),
            "destination": Path("artifacts/data/images"),
            "labels": ["Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"]
        }
    }

    DATA_AUGMENTATION = {
        "EXECUTE": True,
        "PARAMETERS": {
            "source_path": DOWNLOAD["PARAMETERS"]["destination"],
            "goal_amount_per_label": 0
        }
    }

    DEEP_LEARNING = {
        "EXECUTE": True,
        "PARAMETERS": {
            "source_path": DOWNLOAD["PARAMETERS"]["destination"],
            "export_path": Path("artifacts/model"),
            "epochs": 100,
            "early_stopping": 5
        }
    }
