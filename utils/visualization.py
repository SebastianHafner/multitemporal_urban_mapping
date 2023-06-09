import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class DateColorMap(object):

    def __init__(self, ts_length: int, color_map: str = 'jet'):
        self.ts_length = ts_length
        default_cmap = cm.get_cmap(color_map, ts_length - 1)
        dates_colors = default_cmap(np.linspace(0, 1, ts_length - 1))
        no_change_color = np.array([0, 0, 0, 1])
        cmap_colors = np.zeros((ts_length, 4))
        cmap_colors[0, :] = no_change_color
        cmap_colors[1:, :] = dates_colors
        self.cmap = mpl.colors.ListedColormap(cmap_colors)

    def get_cmap(self):
        return self.cmap

    def get_vmin(self):
        return 0

    def get_vmax(self):
        return self.ts_length


def plot_change_date_label(ax, aoi_id: str):
    ts = dataset_helpers.get_timeseries(aoi_id)
    change_date_label = label_helpers.generate_change_date_label(aoi_id)
    cmap = DateColorMap(len(ts))
    ax.imshow(change_date_label, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_classification(ax, pred: np.ndarray, dataset: str, aoi_id: str):
    label = label_helpers.generate_change_label(dataset, aoi_id, config.include_masked()).astype(np.bool)
    pred = pred.squeeze().astype(np.bool)
    tp = np.logical_and(pred, label)
    fp = np.logical_and(pred, ~label)
    fn = np.logical_and(~pred, label)

    img = np.zeros(pred.shape, dtype=np.uint8)

    img[tp] = 1
    img[fp] = 2
    img[fn] = 3

    colors = [(0, 0, 0), (1, 1, 1), (142 / 255, 1, 0), (140 / 255, 25 / 255, 140 / 255)]
    cmap = mpl.colors.ListedColormap(colors)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])


