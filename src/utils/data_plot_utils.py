import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import os

load_dotenv()
IMG_DIM = int(os.getenv('IMG_DIM'))

class KannadaDataClass(Dataset):
    def __init__(self, images, labels, transform, classes):
        """
        instantiate KannadaData class with class artifacts
        """
        self.X = images
        self.y = labels
        self.tranform = transform
        self.classes = classes
        
    def __len__(self):
        """
        mandatory function
        """
        return len(self.X)
    
    def __getitem__(self, idx=None):
        """
        will return label in softmax format [0,0,0,1,0,...]
        reshape image data to (28,28,1)
        return both image and label.
        """
        img = np.array(self.X.iloc[idx,:], dtype='uint8').reshape((IMG_DIM,IMG_DIM,1))
        if self.y is not None:
            y = np.zeros(self.classes, dtype='float32')
            y[self.y.iloc[idx]] = 1
            return img, y
        else:
            return img

def plot_digits(X_train, y_train):
    """
    get all images for each digit and print first 10 for each digit
    saved under full_figure.png in the plots folder at the root level
    """
    # saves image to plots folder
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))
    for i in range(10): # Column by column
        num_i = X_train[y_train == i]
        ax[0][i].set_title(i)
        for j in range(10): # Row by row
            ax[j][i].axis('off')
            plot= ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
            fig.savefig('full_figure.png')


# common technique in image recognition to add rotations etc to the images
# you train on to try improve regularization

train_transform = transforms.Compose([
    transforms.Resize([IMG_DIM,IMG_DIM]),
    #transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

# test data is not augmented, kept as it is.
test_transform = transforms.Compose([
    transforms.Resize([IMG_DIM,IMG_DIM]),
    #transforms.ToPILImage(),
    transforms.ToTensor()
])

