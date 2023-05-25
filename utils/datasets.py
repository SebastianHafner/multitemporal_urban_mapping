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

    def load_planet_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
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

    def load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = geofiles.read_tif(file)
        m, n, _ = label.shape
        if self.pad and (m != 1024 or n != 1024):
            label = cv2.copyMakeBorder(label, 0, 1024-m, 0, 1024-n, borderType=cv2.BORDER_REPLICATE)
            label = label[:, :, None]  # cv2 squeezes array
        label = label > 0
        return label.astype(np.float32)

    def load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self.load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self.load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def load_mask(self, aoi_id: str, year: int, month: int) -> np.ndarray:
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


class TrainDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 disable_multiplier: bool = False):
        super().__init__(cfg)

        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        self.include_change_label = cfg.DATALOADER.INCLUDE_CHANGE_LABEL

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

        t_values = sorted(np.random.randint(0, len(timestamps), size=self.T))
        timestamps = sorted([timestamps[t] for t in t_values], key=lambda ts: int(ts['year']) * 12 + int(ts['month']))

        images = [self.load_planet_mosaic(aoi_id, ts['dataset'], ts['year'], ts['month']) for ts in timestamps]
        labels = [self.load_building_label(aoi_id, ts['year'], ts['month']) for ts in timestamps]

        images, labels = self.transform((np.stack(images), np.stack(labels)))

        item = {
            'x': images,
            'y': labels,
            'aoi_id': aoi_id,
            'dates': [(int(ts['year']), int(ts['month'])) for ts in timestamps],
        }

        if self.include_change_label:
            labels_ch = []
            for t in range(len(timestamps) - 1):
                labels_ch.append(torch.ne(labels[t + 1], labels[t]))
            labels_ch.append(torch.ne(labels[-1], labels[0]))
            item['y_ch'] = torch.stack(labels_ch)

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class EvalDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, tiling: int = None, aoi_id: str = None):
        super().__init__(cfg)

        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        self.include_change_label = cfg.DATALOADER.INCLUDE_CHANGE_LABEL
        self.tiling = tiling
        self.eval_train_threshold = cfg.DATALOADER.EVAL_TRAIN_THRESHOLD

        # handling transformations of data
        self.transform = augmentations.compose_transformations(cfg, no_augmentations=True)

        self.metadata = geofiles.load_json(self.root_path / f'metadata_siamesessl.json')

        if aoi_id is None:
            if run_type == 'train':
                self.aoi_ids = np.array(list(cfg.DATASET.TRAIN_IDS))
                self.aoi_ids = list(self.aoi_ids[np.random.rand(len(self.aoi_ids)) > self.eval_train_threshold])
            elif run_type == 'val':
                self.aoi_ids = list(cfg.DATASET.VAL_IDS)
            elif run_type == 'test':
                self.aoi_ids = list(cfg.DATASET.TEST_IDS)
            else:
                raise Exception('unkown run type!')
        else:
            self.aoi_ids = [aoi_id]

        if tiling is None:
            self.tiling = 1024

        self.samples = []
        for aoi_id in self.aoi_ids:
            for i in range(0, 1024, self.tiling):
                for j in range(0, 1024, self.tiling):
                    self.samples.append((aoi_id, (i, j)))

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.samples)

    def __getitem__(self, index):

        aoi_id, (i, j) = self.samples[index]

        timestamps = [ts for ts in self.metadata[aoi_id] if not ts['mask']]
        t_values = list(np.linspace(0, len(timestamps), self.T, endpoint=False, dtype=int))
        timestamps = sorted([timestamps[t] for t in t_values], key=lambda ts: int(ts['year']) * 12 + int(ts['month']))

        images = [self.load_planet_mosaic(ts['aoi_id'], ts['dataset'], ts['year'], ts['month']) for ts in timestamps]
        images = np.stack(images)[:, i:i + self.tiling, j:j + self.tiling]
        labels = [self.load_building_label(aoi_id, ts['year'], ts['month']) for ts in timestamps]
        labels = np.stack(labels)[:, i:i + self.tiling, j:j + self.tiling]

        images, labels = self.transform((images, labels))

        item = {
            'x': images,
            'y': labels,
            'aoi_id': aoi_id,
            'i': i,
            'j': j,
            'dates': [(int(ts['year']), int(ts['month'])) for ts in timestamps],
        }

        if self.include_change_label:
            labels_ch = []
            for t in range(len(timestamps) - 1):
                labels_ch.append(torch.ne(labels[t + 1], labels[t]))
            labels_ch.append(torch.ne(labels[-1], labels[0]))
            item['y_ch'] = torch.stack(labels_ch)

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
