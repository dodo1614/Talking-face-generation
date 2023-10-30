import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
import cv2
import collections
from model import Audio2Face384
# Our libs
from dataset import ValDataset
from tqdm import tqdm


def user_scattered_collate(batch):
    return batch


def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    return img


def evaluate(model, loader, gpu_id):
    model.eval()
    results = []
    frames = []
    i = 0
    os.makedirs('results', exist_ok=True)
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        img_resized_list = batch_data['recon']

        with torch.no_grad():
            background = batch_data['frame']
            up, down, left, right = batch_data['bound']
            feed_dict = batch_data.copy()
            if results:
                first = results[-1]
                patch = cv2.resize(first[up:down, left:right], (384, 384))
                img = np.float32(np.array(patch)) / 255.
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img.copy())
                feed_dict['recon'][:, :3, :, :] = img
            feed_dict = async_copy_to(feed_dict, gpu_id)
            scores_tmp = model(feed_dict['mel'], feed_dict['recon'])
            tmp = scores_tmp[:, 3:6, :, :]
            pred = (tmp.squeeze(0).cpu().numpy() * 255).astype('uint8').transpose((1, 2, 0))
            pred = pred.astype('uint8')
            pred = np.where(batch_data['label'][1] < 1, pred, batch_data['label'][1])
            pred = cv2.resize(pred, (down - up, right - left))
            background[1][up:down, left:right] = pred
            results.append(background[1].copy())
            cv2.imwrite(os.path.join('results', f'{i:05d}.jpg'), background[1])
            i += 1


def main(args):
    gpu_id = int(args.gpu)
    torch.cuda.set_device(gpu_id)
    dataset_val = ValDataset(args.video, args.audio)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    weights = "checkpoints.pth"
    model = Audio2Face384(input_dim=12)
    model.load_state_dict(
        torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    model.cuda()

    # Main loop
    evaluate(model, loader_val, gpu_id)
    os.system("ffmpeg -r 25 -f image2 -i results/%05d.jpg -y results/mute.mp4")
    os.system(f"ffmpeg -i results/mute.mp4 -i ./data/audio/{args.audio} -y result.mp4")
    os.system("rm -rf ./results")
    print('Evaluation Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch Audio2Face Validation"
    )
    parser.add_argument(
        "--gpu",
        default=0,
    )
    parser.add_argument(
        "--video",
        default='obama.mp4',
    )
    parser.add_argument(
        "--audio",
        default='obama1.wav',
    )
    args = parser.parse_args()
    main(args)
