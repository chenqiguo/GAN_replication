# version 1: my 1st try

from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyCustomDataset(Dataset):
    """Custom Dataset.
    """
    
    def __init__(self, root, label_file, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            label_file (string): Path to the txt file with class labels.
            transform (callable): Transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.data = datasets.ImageFolder(root, transform)
        
        #self.classes = np.unique(self.data.targets)
        #self.targets = self.data.targets
        # read class labels from txt file:
        self.targets = []
        with open(label_file) as file_in:
            for line in file_in:
                line_ = line.split(' ')[-1]
                line_ = line_.strip()
                this_target = int(line_) - 1 # labels should start from 0!!!
                self.targets.append(this_target)
        self.classes = np.unique(self.targets)
    
    def __len__(self):
        return len(self.data.imgs)
    
    def __getitem__(self, index):
        #target = self.data.targets[index]
        target = self.targets[index]
        
        image_name = self.data.imgs[index][0]
        image = Image.open(image_name)

        if self.transform is not None:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        
        return pos_1, pos_2, target





