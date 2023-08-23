import torch
from utils.general import(
    DEVICE,
    PATCHES_VAL_PATH
)
from utils.dataloader import Satelite_images
from utils.utils import build_image, get_pcamodel
from utils.pred_step import openPCS


if __name__ == "__main__":
    model = torch.load("model/open_set_model_UNET.pth")
    model = model.to(DEVICE)
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    model_full = get_pcamodel(model, dl)
    build_image(model, openPCS, model_full, 0, 'openPCS')