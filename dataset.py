import os
import torch
import numpy as np
import audio
import cv2


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, video, speech):
        super(ValDataset, self).__init__()
        self.root_dataset = './data'
        self.list_sample = []
        dir_name = video.split('.')[0]
        self.video_name = dir_name
        test_wav_path = os.path.join('./data/audio/', speech)
        wav = audio.load_wav(test_wav_path, 16000)
        rect_path = os.path.join(self.root_dataset, 'rect', f'{dir_name}.npy')
        self.rect = np.load(rect_path)
        self.frame_num = self.rect.shape[0]
        landmark_path = rect_path.replace('rect', 'landmark')
        self.landmark = np.load(landmark_path)
        self.mel = audio.melspectrogram(wav).T
        self.output_length = 384
        self.max_frame = 5
        self.mel_length = int(3.2 * self.max_frame)
        self.num_sample = int(self.mel.shape[0] / 3.2)
        print(self.num_sample)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img

    def compute_new_ratio(self, this_landmark, cx, cy, half_length, width, height):
        x, y = this_landmark
        new_x = round(((x - cx) * width / half_length / 2 + 0.5) * self.output_length)
        new_y = round(((y - cy) * height / half_length / 2 + 0.5) * self.output_length)
        return new_x, new_y

    def get_frame_window(self, num):
        video = self.video_name
        num = num % self.frame_num
        background_path = os.path.join(self.root_dataset, 'frame', video, f'{num:05d}.jpg')
        background = cv2.imread(background_path)
        backgrounds = []
        recons = []
        labels = []
        success, xmin, ymin, rect_w, rect_h = self.rect[num]
        print(num, success)
        height, width, _ = background.shape
        half_length = round(max(rect_w * width, rect_h * height) * 0.9)
        cxr, cyr = xmin + rect_w / 2, ymin + rect_h / 2
        cx, cy = round((xmin + rect_w / 2) * width), round((ymin + rect_h / 2) * height)
        half_length = min(half_length, cy, height - cy, cx, width - cx)
        for i in range(num - self.max_frame // 2 + 1, num + self.max_frame - self.max_frame // 2 - 1):
            this_num = min(max(i, 0), self.frame_num - 1)
            label_path = os.path.join(self.root_dataset, 'frame', video, f'{this_num:05d}.jpg')
            label = cv2.imread(label_path)
            backgrounds.append(label.copy())
            segm_path = os.path.join('./data/parsing', video, f'{this_num:05d}.png')
            segm = cv2.imread(segm_path)
            label = label[cy - half_length:cy + half_length, cx - half_length:cx + half_length]
            label = cv2.resize(label, (self.output_length, self.output_length))
            recon = label.copy()
            segm = segm[cy - half_length:cy + half_length, cx - half_length:cx + half_length]
            segm = cv2.resize(segm, (self.output_length, self.output_length), interpolation=cv2.INTER_NEAREST)
            mask = np.zeros(segm.shape)
            mask[segm == 0] = 255
            mask[segm == 16] = 255
            mask = segm.copy()
            mask = np.where(mask < 16, mask, 0)
            mask = np.where(mask < 1, mask, 1)
            r = 10
            k = np.ones((r, r), np.uint8)
            mask = cv2.dilate(mask, k)
            mask[segm > 17] = 0
            landmark = self.landmark[this_num]
            points = [[0, self.output_length // 2]]
            line = [127, 34, 143, 35, 226, 130, 25, 110, 24, 23, 22, 26, 112, 243, 244, 245, 122, 6, 351, 465, 464, 463,
                    341, 256, 252, 253, 254, 339, 255, 359, 446, 265, 372, 264, 356]
            for p in line:
                x, y = self.compute_new_ratio(landmark[p], cxr, cyr, half_length, width, height)
                points.append([x, y])
            x1, y1 = points[1]
            x2, y2 = points[-1]
            xl, yl = 0, round((x1 * y2 - x2 * y1) / (x1 - x2))
            xr, yr = self.output_length, round(
                (y1 * self.output_length - y2 * self.output_length + x1 * y2 - x2 * y1) / (x1 - x2))
            points[0] = [xl, yl]
            points.extend([[self.output_length, yr], [self.output_length, 0], [0, 0], [0, yl]])
            mask = cv2.fillPoly(mask, [np.array(points)], color=[0, 0, 0])
            recon[mask > 0] = 0
            recons.append(recon)
        labels = recons.copy()
        sample = labels[1]
        recons.append(sample)
        recon = self.img_transform(np.concatenate(recons, axis=2))
        bound = [cy - half_length, cy + half_length, cx - half_length, cx + half_length]
        return labels, recon, backgrounds, bound

    def __getitem__(self, index):
        batch_recons = torch.zeros(1, 3 * (self.max_frame - 1), self.output_length, self.output_length)
        batch_mels = torch.zeros(1, 1, 80, self.mel_length)
        base = int((index - self.max_frame // 2) * 3.2)
        start = int(max(index - self.max_frame // 2, 0) * 3.2)
        end = int(min((index + self.max_frame - self.max_frame // 2) * 3.2, self.mel.shape[0]))
        mel = np.zeros((self.mel_length, 80))
        mel[start - base:end - base, :] = self.mel[start:end]
        batch_mels[0, 0] = torch.from_numpy(mel.copy()).T
        label, recon, background, bound = self.get_frame_window(index)
        batch_recons[0] = recon
        output = dict()
        output['label'] = label
        output['recon'] = batch_recons
        output['mel'] = batch_mels
        output['frame'] = background
        output['bound'] = bound
        return output

    def __len__(self):
        return self.num_sample
