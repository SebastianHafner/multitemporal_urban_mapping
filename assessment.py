import torch
import matplotlib.pyplot as plt
import numpy as np
from array2gif import write_gif
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
from pathlib import Path
import cv2
from utils import experiment_manager, networks, datasets, parsers, evaluation
FONTSIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_aoi_ids(run_type: str):
    if run_type == 'train':
        aoi_ids = list(cfg.DATASET.TRAIN_IDS)
    elif run_type == 'val':
        aoi_ids = list(cfg.DATASET.VAL_IDS)
    elif run_type == 'test':
        aoi_ids = list(cfg.DATASET.TEST_IDS)
    else:
        raise Exception('unkown run type!')
    return aoi_ids


def pred2frames(y_hat: np.ndarray, y: np.ndarray = None, color_type: str = None) -> np.ndarray:
    if color_type is None:
        y_hat = np.clip(y_hat * 255, 0, 255)
        frames = np.repeat(y_hat[:, :, :, None], 3, axis=-1)
    elif color_type == 'seg':
        assert(y is not None)
        frames = np.zeros((*y_hat.shape, 3), dtype=np.uint8)
        tp = np.logical_and(y_hat, y)
        fp = np.logical_and(y_hat, np.logical_not(y))
        fn = np.logical_and(np.logical_not(y_hat), y)

        frames[np.repeat(tp[:, :, :, None], 3, axis=-1)] = 255
        frames[:, :, :, 1][fp] = 255
        frames[:, :, :, 0][fn] = 255
        frames[:, :, :, 2][fn] = 255
    elif color_type == 'tc':
        assert(y is not None)
        frames = np.zeros((*y_hat.shape, 3), dtype=np.uint8)
        T = y_hat.shape[0]
        for t in range(1, T):
            gt_cons = y[t] == y[t-1]
            pred_incons = y_hat[t] != y_hat[t-1]
            incons = np.logical_and(pred_incons, gt_cons)
            frames[t, :, :, 0][incons] = 255
    else:
        raise Exception('unkown color type')

    return frames.astype(np.uint8)

def qualitative_temporal_consistency_assessment_video(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    aoi_ids = get_aoi_ids(run_type)
    for aoi_id in aoi_ids:
        print(aoi_id)
        ds = datasets.EvalSingleAOIDataset(cfg, aoi_id)

        # Set up the frames per second and frame size for the video
        fps = 1
        frame_size = (1024, 1024)

        video_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / 'gifs' / f'gif_{aoi_id}_{cfg.NAME}.avi'

        # Create a VideoWriter object
        # https://rsdharra.com/blog/lesson/5.html
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        video_writer = cv2.VideoWriter(str(video_file), fourcc, fps, frame_size, True)

        for i, item in enumerate(ds):
            with torch.no_grad():
                x = item['x'].to(device).unsqueeze(0)
                logits = net(x)
            y_hat = torch.sigmoid(logits)

            x = x.detach().cpu().squeeze().numpy()
            x = np.clip(x * 255, 0, 255)
            x = x.transpose(1, 2, 0).astype('uint8')
            m, n, _ = x.shape

            img = np.zeros((*frame_size, 3), dtype=np.uint8)
            img[:m, :n] = x

            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if i == 5:
                break


        # Release the VideoWriter object and close the video file
        video_writer.release()
        cv2.destroyAllWindows()
        # https://pypi.org/project/array2gif/
        # out_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / 'gifs' / f'gif_{aoi_id}_{cfg.NAME}.gif'
        # write_gif(images[0], out_file)
        break


def qualitative_tc_gif(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    aoi_ids = get_aoi_ids(run_type)
    for aoi_id in aoi_ids:
        print(aoi_id)
        ds = datasets.EvalDataset(cfg, run_type, aoi_id=aoi_id)

        # Set up the frames per second and frame size for the video
        img_unit = 1024
        fps = 1
        frame_size = (img_unit, 3 * img_unit)
        n_frames = len(ds)

        gif_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / 'gifs' / f'gif_{aoi_id}_{cfg.NAME}.gif'
        frames = np.random.randint(256, size=[20, 64, 64, 3], dtype=np.uint8)  # YOUR DATA HERE
        frames = np.zeros((n_frames, *frame_size, 3), dtype=np.uint8)

        for t, item in enumerate(ds):
            with torch.no_grad():
                x = item['x'].to(device).unsqueeze(0)
                logits = net(x)
            y_hat = torch.sigmoid(logits) > 0.5
            y_hat = y_hat.detach().cpu().squeeze().numpy()
            y_hat = np.clip(y_hat * 255, 0, 255)

            x = x.detach().cpu().squeeze().numpy()
            x = np.clip(x * 255, 0, 255)
            x = x.transpose(1, 2, 0).astype('uint8')
            m, n, _ = x.shape
            frames[t, :m, :n] = x

            m, n = y_hat.shape
            y_hat_rgb = np.repeat(y_hat[:, :, np.newaxis], 3, axis=2)
            frames[t, :m, 2 * img_unit:2 * img_unit + n, :] = y_hat_rgb

            y = item['y'].cpu().squeeze().numpy()
            y = np.clip(y * 255, 0, 255)
            m, n = y.shape
            y_rgb = np.repeat(y[:, :, np.newaxis], 3, axis=2)
            frames[t, :m, img_unit:img_unit + n, :] = y_rgb

        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(str(gif_file), fps=fps)


def qualitative_tc_gif_colored(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    aoi_ids = get_aoi_ids(run_type)
    for aoi_id in aoi_ids:
        print(aoi_id)
        tiling = cfg.AUGMENTATION.CROP_SIZE
        ds = datasets.EvalDataset(cfg, run_type, aoi_id=aoi_id, tiling=tiling)

        # Set up the frames per second and frame size for the video
        img_unit = 1024
        fps = 1
        frame_size = (img_unit, 3 * img_unit)
        n_frames = cfg.DATALOADER.EVAL_TIMESERIES_LENGTH

        gif_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / 'gifs' / f'gif_{aoi_id}_{cfg.NAME}.gif'
        frames = (np.random.rand(n_frames, *frame_size, 3) * 255).astype(np.uint8)

        for item in ds:
            i, j = item['i'], item['j']
            y = item['y'].cpu().squeeze().numpy()

            x = item['x'].to(device)
            with torch.no_grad():
                logits = net(x.unsqueeze(0))

            y_hat = torch.sigmoid(logits) > 0.5
            y_hat = y_hat.detach().cpu().squeeze().numpy()

            pred_frames = pred2frames(y_hat, y, color_type='seg')
            frames[:, i:i + tiling, img_unit + j:img_unit + j + tiling, :] = pred_frames

            tc_frames = pred2frames(y_hat, y, color_type='tc')
            frames[:, i:i + tiling, 2 * img_unit + j: 2 * img_unit + j + tiling, :] = tc_frames

            img = x.detach().cpu().squeeze().numpy()
            img = np.clip(img * 255, 0, 255)
            img = img.transpose(0, 2, 3, 1).astype('uint8')
            frames[:, i:i + tiling, j:j + tiling, :] = img

        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(str(gif_file), fps=fps)
        break


def quantitative_temporal_consistency_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    values = []

    aoi_ids = get_aoi_ids(run_type)
    for aoi_id in aoi_ids:
        print(aoi_id)
        ds = datasets.EvalSingleAOIDataset(cfg, aoi_id)
        measurer = evaluation.TCMeasurer(aoi_id)

        for t, item in enumerate(ds):
            with torch.no_grad():
                x = item['x'].to(device).unsqueeze(0)
                logits = net(x)
            y_hat = torch.sigmoid(logits).detach().squeeze()
            y = item['y'].to(device).squeeze()
            measurer.add_sample(y, y_hat)

        values.append(measurer.tc().item())

    fig, ax = plt.subplots(1, 1)
    ax.boxplot(values)

    print(values)

if __name__ == '__main__':
    args = parsers.deployement_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # quantitative_temporal_consistency_assessment(cfg)
    qualitative_tc_gif_colored(cfg)
    # qualitative_assessment_change(cfg, run_type=args.run_type)
    # qualitative_assessment_sem(cfg, run_type=args.run_type)
