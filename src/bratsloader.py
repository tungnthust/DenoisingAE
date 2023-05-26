import torch 
import numpy as np
import torch.nn.functional as F
import pickle

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    if non_zeros.sum() > 0:
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)
        image = normalize(image)
    return image

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        
        super().__init__()
        self.datapaths = []
        if mode != 'test':
            with open(f'/kaggle/working/DenoisingAE/data/brats20/{mode}_healthy_brats20_datapaths.pickle', 'rb') as fp:
                self.datapaths = pickle.load(fp)
        else:
            with open(f'/kaggle/working/DenoisingAE/data/brats20/val_unhealthy_brats20_datapaths.pickle', 'rb') as fp:
                self.datapaths = pickle.load(fp)

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = image[[1, 2, 3, 0], :, :]
        image = F.interpolate(torch.Tensor(np.expand_dims(image, axis=0)), mode="bilinear", size=(128, 128))[0]
        for i in range(image.shape[0]):
            image[i] = irm_min_max_preprocess(image[i])
        mask = data['mask']
        mask = F.interpolate(torch.Tensor(np.expand_dims(mask, axis=(0, 1))), mode="bilinear", size=(128, 128))[0][0]
        mask = np.where(mask > 0, 1, 0)
        # padding_image = np.zeros((4, 256, 256))
        # padding_image[:, 8:-8, 8:-8] = image
        # padding_mask = np.zeros((256, 256))
        # padding_mask[8:-8, 8:-8] = mask
        label = 1 if np.sum(mask) > 0 else 0
        cond = {}
        cond['y'] = label
        return np.float32(image), np.float32(mask)

    def __len__(self):
        return len(self.datapaths)