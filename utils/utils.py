from utils.dataloader import Satelite_images
from utils.general import(
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    MAP_COLOR,
    COLOR_TO_RGB,
    NUM_KNOWN_CLASSES,
    IMAGE_PATH
)
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.openmax import compute_distance       
from utils.evt import weibull_tailfitting
from osgeo import gdal, osr
from torchvision import transforms
from utils.openpca import fit_pca_model, fit_quantiles, pred_pixelwise
import gc



def create_dataloader(path_to_patches, endpoint):
    dl = Satelite_images(path_to_patches, endpoint)
    index = list(range(len(dl)))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= index)
    return train_loader, dl.getweight()


def get_distances(dl, mean, model):
    model.eval()
    collection = [[] for _ in range(7)]
    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                output = output[0]
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu").numpy()
                predictions = predictions.squeeze(0).to("cpu").numpy()
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        if np.argmax(predictions[:,i,j]) == mask[i][j] and mask[i,j] < 7:
                            centroid = mean[mask[i,j]]
                            distance = compute_distance(output[:,i,j], centroid, 'eucos')
                            collection[mask[i,j]].append(distance)
            pbar.update(1)
    np.save("distance/distances_eucos.npy", np.array(collection, dtype=object), allow_pickle=True)
    

def read_image_gdal(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image = dataset.ReadAsArray()
    return image


def image_to_tensor(image):
    image = np.transpose(image, (1, 2, 0))  
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    tensor = transform(image)
    return tensor

def get_mean(dl, model):
    amount = np.zeros(7)
    mean = np.zeros((7,7))
    model.eval()
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                output = output[0]
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                predictions = predictions.numpy()
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        if label[i][j] < 7 and np.argmax(predictions[:,i,j]) == label[i][j]:
                            mean[label[i][j]] += output[:,i,j].numpy()
                            amount[label[i][j]] += 1
            pbar.update(1)
    for i in range(7):
        mean[i] = mean[i]/amount[i]
    np.save("mean/mean_eucos.npy", mean)
    return mean


def get_weibull_model(tail):
    print("Iniciando a determinação do modelo weibull")
    dist_list = np.load("distance/distances_eucos.npy", allow_pickle= True)
    mean_list = np.load("mean/mean_eucos.npy")
    weibull_model = weibull_tailfitting(mean_list, dist_list, NUM_KNOWN_CLASSES, tailsize=tail)
    return weibull_model


def save_image(image, original_image):
    colored_image = np.array([[MAP_COLOR[label] for label in row] for row in image])   
    rgb_image = np.array([[COLOR_TO_RGB[color] for color in row] for row in colored_image])
    original_image = original_image/256
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title('Mascara')

    axes[1].imshow(original_image)
    axes[1].set_title('Imagem original')

    plt.tight_layout()
    plt.savefig("output/image_mask.jpg", dpi = 300, bbox_inches = 'tight')

def save_np_image(image_np):
    colored_image = np.array([[MAP_COLOR[label] for label in row] for row in image_np])   
    rgb_image = np.array([[COLOR_TO_RGB[color] for color in row] for row in colored_image])
    np.save('output/np_mask_image.npy', rgb_image)

def save_image_tif(arr):
    height, width = arr.shape
    output_path = 'output/tif_mask.tif'
    driver = gdal.GetDriverByName('GTiff')
    bands = 3
    out_dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)
    out_dataset.GetRasterBand(1).WriteArray(arr)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_dataset.SetProjection(srs.ExportToWkt())
    out_dataset = None

def build_image(model, pred_function, open_model, t, name):
    image = read_image_gdal(IMAGE_PATH)
    image_tensor = image_to_tensor(image)
    image_numpy = image_tensor.numpy().astype(np.int64)
    model = model.to(DEVICE)
    model.eval()
    _,lenght,width = image_tensor.size()
    output = torch.zeros((lenght,width))
    with torch.no_grad():
        i = 0
        with tqdm(total = int(lenght/64)) as pbar:
            while(i + 64 <= lenght):
                tensor =  image_tensor[:,i:i+64,width -64: width]
                tensor = tensor.float().unsqueeze(0).to(DEVICE)
                output = pred_function(tensor, model, open_model, output, i, width - 64, t)
                i += 64
                pbar.update(1)
        i = 0
        with tqdm(total = int(width/64)) as pbar:
            while(i + 64 <= width):
                tensor =  image_tensor[:,lenght -64: lenght,i:i+64]
                tensor = tensor.float().unsqueeze(0).to(DEVICE)
                output = pred_function(tensor, model, open_model, output, lenght - 64, i, t)
                i += 64
                pbar.update(1)
        with tqdm(total= int(lenght/64) * int(width/64)) as pbar:
            i = 0
            j = 0
            while(i + 64 <= lenght):
                j = 0
                while(j + 64 <= width):
                    tensor =  image_tensor[:,i:i+64,j:j+64]
                    tensor = tensor.float().unsqueeze(0).to(DEVICE)
                    output = pred_function(tensor, model, open_model, output, i, j, t)
                    j += 64
                    pbar.update(1)
                i+= 64
    output = output.numpy()
    output = output.astype(np.int64)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    save_image(output, image_numpy[:,:,0:3], name)
    save_np_image(image_numpy[:,:,0:3], name)


def get_pcamodel(model, dl):
    model.eval()
    pred_list = []
    mask_list = []
    feat_list = []
    model_list = []
    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                out, lay1, lay2 = model(image)
                soft_out = F.softmax(out, dim = 1)
                pred = torch.argmax(soft_out, dim = 1)
                pred = pred.squeeze(0).to("cpu").numpy().ravel()
                mask = mask.numpy().ravel()
                out = torch.cat([out.squeeze(0), lay1.squeeze(0), lay2.squeeze(0)], 0).to("cpu")
                out = out.permute(1, 2, 0).contiguous().view(out.size(1) * out.size(2), out.size(0)).numpy()
                pred_list.append(pred)
                mask_list.append(mask)
                feat_list.append(out)
                pbar.update(1)
    gc.collect()
    feat_list = np.asarray(feat_list)
    feat_list = feat_list.reshape(feat_list.shape[0] * feat_list.shape[1], feat_list.shape[2])
    mask_list = np.asarray(mask_list)
    mask_list = mask_list.reshape(mask_list.shape[0] * mask_list.shape[1])
    pred_list = np.asarray(pred_list)
    pred_list = pred_list.reshape(pred_list.shape[0] * pred_list.shape[1])
    gc.collect()
    for c in range(NUM_KNOWN_CLASSES):
        print(f"Fiting model for class {c}")
        model = fit_pca_model(feat_list, mask_list, pred_list, c, 7)
        model_list.append(model)
    
    scr_threshold = fit_quantiles(model_list, feat_list, pred_list, NUM_KNOWN_CLASSES)
    model_full = {'generative': model_list,
                  'thresholds': scr_threshold}
    return model_full