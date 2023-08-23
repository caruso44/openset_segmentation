import torch
from utils.general import(
    DEVICE,
    PATCHES_VAL_PATH,
)
from utils.dataloader import Satelite_images
from utils.utils import get_distances, get_mean
import torch.nn.functional as F
from utils.utils import build_image, get_weibull_model
from utils.pred_step import openFCN


def get_lists(net):
    val_loader = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    print("Calculando as medias")
    mean =  get_mean(val_loader, net)
    #mean = np.load("mean/mean_eucos.npy")
    print("Calculando as distâncias")
    dist = get_distances(val_loader, mean, net)
    

if __name__ == "__main__":
    model = torch.load("model/open_set_model_UNET.pth")
    model = model.to(DEVICE)
    get_lists(model)
    weibull_model = get_weibull_model(10000)
    build_image(model, openFCN, weibull_model,0, 'openFCN')