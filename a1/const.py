from pathlib import Path
from os.path import join

DATA_FOLDER = Path(__name__).parent / "data"
SAMPLES_FOLDER = join(DATA_FOLDER, "sound-samples")
IMPULSES_FOLDER = join(DATA_FOLDER, "sound-impulses")
CLAPS_FOLDER = join(IMPULSES_FOLDER, "claps")
SPLASHES_FOLDER = join(IMPULSES_FOLDER, "splashes")

OUTPUT_FOLDER = Path(__name__) / "output"
