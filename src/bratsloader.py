import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import pandas as pd

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
    def __init__(self, mode="train", fold=1, test_flag=False, transforms=None):
        
        super().__init__()
        self.datapaths = []
        self.transforms = transforms
        if self.transforms:
            print("Transform for data augmentation.")
        else:
            print("No data augmentation")
            
        data_split = np.load('/kaggle/working/DenoisingAE/data/brats20/data_split.npz', allow_pickle=True)
        meta_data_df = pd.read_csv('/kaggle/working/DenoisingAE/data/brats20/meta_data.csv')
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        if not test_flag:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids) & (meta_data_df['label'] == 0)]['path'].values
        else:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids) & (meta_data_df['label'] == 1)]['path'].values
        print(f'Loaded data fold {fold}, test_flag = {test_flag}. Number of {mode} data: {len(self.datapaths)}')

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = image[[1, 2, 3, 0], :, :]
        for i in range(image.shape[0]):
            image[i] = irm_min_max_preprocess(image[i])
        mask = data['mask']
        padding_image = np.zeros((4, 256, 256))
        padding_image[:, 8:-8, 8:-8] = image
        padding_mask = np.zeros((256, 256))
        padding_mask[8:-8, 8:-8] = mask
        # image_resized = F.interpolate(torch.Tensor(np.expand_dims(padding_image, axis=0)), mode="bilinear", size=(128, 128))[0]
        # mask_resized = F.interpolate(torch.Tensor(np.expand_dims(padding_mask, axis=(0, 1))), mode="bilinear", size=(128, 128))[0][0]
        # padding_image = np.array(image_resized)
        # padding_mask = np.array(mask_resized)
        # padding_mask = padding_mask > 0
        label = 1 if np.sum(mask) > 0 else 0
        cond = {}
        cond['y'] = label
        if self.transforms:
            padding_image = self.transforms(torch.Tensor(padding_image))
        return np.float32(padding_image), cond, label, np.float32(padding_mask)

    def __len__(self):
        return len(self.datapaths)