from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import ToTensor
import utils.general as general
from tqdm import tqdm

class Satelite_images(Dataset):
    def __init__(self, path_to_patches, endpoint, transformer = ToTensor()) -> None:
        opt_img = np.load(os.path.join(general.PATH, f'OPT_img.npy'))
        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))

        #self.labels = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).reshape((-1,1)).astype(np.int64)
        self.labels = np.load(os.path.join(general.PATH, f'LABEL' + endpoint)).flatten().astype(np.int64)
        self.n_classes = np.unique(self.labels).shape[0]
        self.patches = np.load(path_to_patches)
        self.transformer = transformer
        
        
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch_idx = self.patches[index]
        opt_tensor = self.transformer(self.opt_img[patch_idx])
        #label_tensor = self.transformer(self.labels[patch_idx].astype(np.int64)).squeeze(0)
        label_tensor = torch.tensor(self.labels[patch_idx])
        return (
            opt_tensor,
            label_tensor
        )

    def getindex(self):
        n = self.__len__()
        index = []
        for i in range(n):
            index.append(i)
        return np.array(index)
    
    def getweight(self):
        n = self.__len__()
        weights = torch.zeros(7)
        with tqdm(total=n) as pbar:
            for index in range(n):
                patch_idx = self.patches[index]
                label_tensor = torch.tensor(self.labels[patch_idx])
                for i in range(64):
                    for j in range(64):
                        if label_tensor[i][j] < 7:
                            weights[label_tensor[i][j]] += 1
                pbar.update(1)

        weights = 1 - weights/weights.sum()
        weights = weights/weights.sum()
        return weights
    

        
