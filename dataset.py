
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import os

class UAVDataset(Dataset):
    def __init__(self, data_folder, label_list, transform):

      self.data_folder = data_folder
      self.transform = transform
      self.label_list = label_list
      self.data = []
      self.load_data()

    def load_data(self):
      for label_dir in tqdm(os.listdir(self.data_folder)):

          image_folder = os.path.join(self.data_folder, label_dir)

          image_paths = os.listdir(image_folder)

          for image_path in image_paths:
            id = int(label_dir)

            image = str(os.path.join(image_folder, image_path))

            sample = {
                'label' : id,
                'image' : image,
                'character' : self.label_list[id]['character'],
                'latin' : self.label_list[id]['latin'],
                'length' : self.length(self.label_list[id]['latin'])
            }

            self.data.append(sample)

    def get_latin(self, index):
        return [self.label_list[each]['latin'] for each in index]
        
    def length(self, latin):
        if latin != 'zc': return len(latin)
        return 1

    def __len__(self):
      return len(self.data)

    def num_labels(self):
      return len(self.label_list)

    def __getitem__(self, index):
      sample = self.data[index]

      image = Image.open(sample['image'])
      if self.transform is not None:
        transformed = self.transform(image)

      return {
          'label': sample['label'],
          'image': transformed.repeat(3, 1, 1),
          'character': sample['character'],
          'latin': sample['latin'],
          'length': sample['length']
      }
