from pathlib import Path

DATA_FOLDER = Path(__name__).parent / "data"
OUTPUT_FOLDER = Path(__name__).parent / "a7" / "output"
OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

IMAGES_FOLDER = DATA_FOLDER / "images"
SAMPLES_FOLDER = DATA_FOLDER / "sound-samples" 
IMPULSES_FOLDER = DATA_FOLDER / "sound-impulses"
