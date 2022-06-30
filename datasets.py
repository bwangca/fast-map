import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

from pathlib import Path
from torch.utils.data import *


class PascalVOC(Dataset):

    CLASSES = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    
    def __init__(self, dataset_path, split, transforms=[], encode=None):
        
        assert os.path.isdir(dataset_path)
        dataset_path = Path(dataset_path)
        
        if split == 'train':
            sample_lists = {'2007':'trainval.txt','2012':'trainval.txt'}
        elif split == 'test':
            sample_lists = {'2007':'test.txt'}
        else:
            raise ValueEror
        
        self.transforms = transforms
        self.encode = encode
        
        self.max_objects = 0
        
        self.easy_objects = {i : 0 for i in range(len(self.CLASSES))}
        
        self.image_paths = []
        self.ymins = []
        self.xmins = []
        self.ymaxs = []
        self.xmaxs = []
        self.labels = []
        self.easy_flags = []
        self.num_objects = []
        
        annotation_paths = []
        
        for key,value in sample_lists.items():
            year_dir = 'VOC' + key
            sample_list_path = dataset_path/year_dir/'ImageSets'/'Main'/value
            
            with open(sample_list_path, 'r') as sample_list_file:
                lines = sample_list_file.readlines()
                
                for line in lines:
                    sample_id = line.split(' ')[0].strip('\n')
                    self.image_paths.append(str(dataset_path/year_dir/'JPEGImages'/(sample_id+'.jpg')))
                    annotation_paths.append(str(dataset_path/year_dir/'Annotations'/(sample_id+'.xml')))
        
        for annotation_path in annotation_paths:
            annotation_node = ET.parse(annotation_path).getroot()
            
            ymins = []
            xmins = []
            ymaxs = []
            xmaxs = []
            labels = []
            easy_flags = []
            
            object_nodes = annotation_node.findall('object')
            
            num_objects = len(object_nodes)
            self.num_objects.append(num_objects)
            
            if num_objects > self.max_objects:
                self.max_objects = num_objects
            
            for object_node in object_nodes:
                bndbox_node = object_node.find('bndbox')
                
                ymins.append(float(bndbox_node.find('ymin').text) - 1)
                xmins.append(float(bndbox_node.find('xmin').text) - 1)
                ymaxs.append(float(bndbox_node.find('ymax').text) - 1)
                xmaxs.append(float(bndbox_node.find('xmax').text) - 1)
                
                label = self.CLASSES.index(object_node.find('name').text)
                labels.append(label)
                
                easy_flag = 1 - int(object_node.find('difficult').text)
                easy_flags.append(easy_flag)
                if easy_flag:
                    self.easy_objects[label] += 1
            
            self.ymins.append(np.array(ymins, dtype=np.float32))
            self.xmins.append(np.array(xmins, dtype=np.float32))
            self.ymaxs.append(np.array(ymaxs, dtype=np.float32))
            self.xmaxs.append(np.array(xmaxs, dtype=np.float32))
            self.labels.append(np.array(labels, dtype=np.uint8))
            self.easy_flags.append(np.array(easy_flags, dtype=np.bool))

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        
        # image is in BGR format
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        num_objects = self.num_objects[idx]
        
        ymins = np.zeros((self.max_objects), dtype=np.float32)
        ymins[:num_objects] = self.ymins[idx].copy()
        
        xmins = np.zeros((self.max_objects), dtype=np.float32)
        xmins[:num_objects] = self.xmins[idx].copy()
        
        ymaxs = np.zeros((self.max_objects), dtype=np.float32)
        ymaxs[:num_objects] = self.ymaxs[idx].copy()
        
        xmaxs = np.zeros((self.max_objects), dtype=np.float32)
        xmaxs[:num_objects] = self.xmaxs[idx].copy()
        
        labels = np.zeros((self.max_objects), dtype=np.uint8)
        labels[:num_objects] = self.labels[idx].copy()
        
        easy_flags = np.zeros((self.max_objects), dtype=np.bool)
        easy_flags[:num_objects] = self.easy_flags[idx].copy()
        
        sample = {
            'image':image,
            'height':np.array([height], dtype=np.float32),
            'width':np.array([width], dtype=np.float32),
            'ymins':ymins,
            'xmins':xmins,
            'ymaxs':ymaxs,
            'xmaxs':xmaxs,
            'labels':labels,
            'easy_flags':easy_flags
        }
        
        for transform in self.transforms:
            sample = transform(sample)
        
        if self.encode:
            sample = self.encode(sample)
        
        return sample
