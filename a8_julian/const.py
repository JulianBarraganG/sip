from pathlib import Path

DATA_FOLDER = Path(__name__).parent / "data"
OUTPUT_FOLDER = Path(__name__).parent / "a8" / "output"
OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

TEST_IMAGES_FOLDER = DATA_FOLDER / "test_images" / "images"
TEST_SEGMENTATIONS_FOLDER = DATA_FOLDER / "test_images" / "seg"
IMAGES_FOLDER = DATA_FOLDER / "images"
SAMPLES_FOLDER = DATA_FOLDER / "sound-samples" 
IMPULSES_FOLDER = DATA_FOLDER / "sound-impulses"
