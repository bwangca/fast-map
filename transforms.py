import cv2
import numpy as np

class LetterBox(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image_height = float(sample['height'])
        image_width = float(sample['width'])
        long_side = max(image_height, image_width)
        s = self.size / long_side
        tx = self.size / 2 - image_width / 2 * s
        ty = self.size / 2 - image_height / 2 * s
        m_image = np.array([
            [s, 0, tx],
            [0, s, ty]
        ])
        sample['image'] = cv2.warpAffine(sample['image'], m_image, (self.size,self.size))
        m_box = m_image
        xmins = m_box[0][0] * sample['xmins'] + m_box[0][1] * sample['ymins'] + m_box[0][2]
        ymins = m_box[1][0] * sample['xmins'] + m_box[1][1] * sample['ymins'] + m_box[1][2]
        xmaxs = m_box[0][0] * sample['xmaxs'] + m_box[0][1] * sample['ymaxs'] + m_box[0][2]
        ymaxs = m_box[1][0] * sample['xmaxs'] + m_box[1][1] * sample['ymaxs'] + m_box[1][2]
        sample['ymins'] = np.clip(ymins, 0, self.size - 1)
        sample['xmins'] = np.clip(xmins, 0, self.size - 1)
        sample['ymaxs'] = np.clip(ymaxs, 0, self.size - 1)
        sample['xmaxs'] = np.clip(xmaxs, 0, self.size - 1)
        return sample

class ToUnit(object):

    def __call__(self, sample):
        sample['image'] = np.float32(sample['image']) / 255
        return sample

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.mean) / self.std
        return sample
