from pathlib import Path
import numpy as np
from utils import geofiles, experiment_manager
from utils.experiment_manager import CfgNode


def get_aoi_ids(cfg: CfgNode, run_type: str) -> list:
    if run_type == 'train':
        aoi_ids = list(cfg.DATASET.TRAIN_IDS)
    elif run_type == 'val':
        aoi_ids = list(cfg.DATASET.VAL_IDS)
    elif run_type == 'test':
        aoi_ids = list(cfg.DATASET.TEST_IDS)
    else:
        raise Exception('unkown run type!')
    return aoi_ids


def get_geo(cfg: CfgNode, aoi_id: str) -> tuple:
    data_path = Path(cfg.PATHS.DATASET)
    images_path = data_path / 'train' / aoi_id / 'images'
    file = [f for f in images_path.glob('**/*') if f.is_file()][0]
    _, transform, crs = geofiles.read_tif(file)
    return transform, crs
