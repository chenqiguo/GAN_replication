# version 2: based on v1, do the following updates:
# each image represents a class (instead of using the original class labels) !


from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyCustomDataset_v2(Dataset):
    """Custom Dataset.
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable): Transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.data = datasets.ImageFolder(root, transform)
        
        # each image represents a class:
        total_num_img = len(self.data.imgs)
        self.targets = [*range(total_num_img)] # labels should start from 0!!!
        self.classes = np.unique(self.targets) # should be the same as self.targets
    
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





