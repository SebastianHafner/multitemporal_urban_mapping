import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles
import cv2


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        self.pad = cfg.DATALOADER.PAD_BORDERS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_planetscope_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        m, n, _ = img.shape
        if self.pad and (m != 1024 or n != 1024):
            # https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/
            img = cv2.copyMakeBorder(img, 0, 1024-m, 0, 1024-n, borderType=cv2.BORDER_REPLICATE)
        return img.astype(np.float32)

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = geofiles.read_tif(file)
        m, n, _ = label.shape
        if self.pad and (m != 1024 or n != 1024):
            label = cv2.copyMakeBorder(label, 0, 1024-m, 0, 1024-n, borderType=cv2.BORDER_REPLICATE)
            label = label[:, :, None]  # cv2 squeezes array
        label = label > 0
        return label.astype(np.float32)

    def _load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def _load_mask(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_mask.tif'
        mask, _, _ = geofiles.read_tif(file)
        return mask.astype(np.int8)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class TrainSingleDateDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 disable_multiplier: bool = False):
        super().__init__(cfg)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        self.metadata = geofiles.load_json(self.root_path / f'metadata_siamesessl.json')

        if run_type == 'train':
            self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
        elif run_type == 'val':
            self.aoi_ids = list(cfg.DATASET.VAL_IDS)
        elif run_type == 'test':
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            raise Exception('unkown run type!')

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]

        timestamps = [ts for ts in self.metadata[aoi_id] if not ts['mask']]

        t = sorted(np.random.randint(0, len(timestamps), size=1))[0]
        timestamp = timestamps[t]
        dataset, year, month = timestamp['dataset'], timestamp['year'], timestamp['month']

        img = self._load_planetscope_mosaic(aoi_id, dataset, year, month)
        buildings = self._load_building_label(aoi_id, year, month)

        img, buildings = self.transform((img, buildings))

        item = {
            'x': img,
            'y': buildings,
            'aoi_id': aoi_id,
            'year': year,
        }

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class EvalSingleDateDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__(cfg)

        # handling transformations of data
        self.transform = augmentations.compose_transformations(cfg, no_augmentations=True)

        self.metadata = geofiles.load_json(self.root_path / f'metadata_siamesessl.json')

        if run_type == 'train':
            self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
        elif run_type == 'val':
            self.aoi_ids = list(cfg.DATASET.VAL_IDS)
        elif run_type == 'test':
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            raise Exception('unkown run type!')

        self.samples = []
        for aoi_id in self.aoi_ids:
            for timestamp in self.metadata[aoi_id]:
                if not timestamp['mask']:
                    self.samples.append(timestamp)

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.samples = manager.list(self.samples)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        aoi_id, dataset, year, month = sample['aoi_id'], sample['dataset'], sample['year'], sample['month']

        img = self._load_planetscope_mosaic(aoi_id, dataset, year, month)
        buildings = self._load_building_label(aoi_id, year, month)

        img, buildings = self.transform((img, buildings))

        item = {
            'x': img,
            'y': buildings,
            'aoi_id': aoi_id,
            'year': year,
        }

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class EvalSingleAOIDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, aoi_id: str):
        super().__init__(cfg)

        self.aoi_id = aoi_id

        # handling transformations of data
        self.transform = augmentations.compose_transformations(cfg, no_augmentations=True)

        self.metadata = geofiles.load_json(self.root_path / f'metadata_siamesessl.json')

        self.samples = []
        for timestamp in self.metadata[self.aoi_id]:
            if not timestamp['mask']:
                self.samples.append(timestamp)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        dataset, year, month = sample['dataset'], sample['year'], sample['month']

        img = self._load_planetscope_mosaic(self.aoi_id, dataset, year, month)
        buildings = self._load_building_label(self.aoi_id, year, month)

        img, buildings = self.transform((img, buildings))

        item = {
            'x': img,
            'y': buildings,
            'aoi_id': self.aoi_id,
            'year': year,
        }

        return item

    def get_dims(self) -> tuple:
        sample = self.samples[0]
        dataset, year, month = sample['dataset'], sample['year'], sample['month']
        buildings = self._load_building_label(self.aoi_id, year, month)
        m, n, _ = buildings.shape
        return (m, n)

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'