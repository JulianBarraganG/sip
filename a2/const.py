from pathlib import Path

DATA_FOLDER = Path(__name__).parent / "data"
OUTPUT_FOLDER = Path(__name__).parent / "a1" / "output"

IMAGES_FOLDER = DATA_FOLDER / "images"
SAMPLES_FOLDER = DATA_FOLDER / "sound-samples"
IMPULSES_FOLDER = DATA_FOLDER / "sound-impulses"
CLAPS_FOLDER = IMPULSES_FOLDER / "claps"
SPLASHES_FOLDER = IMPULSES_FOLDER / "splashes"
