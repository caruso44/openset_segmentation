import numpy as np
import torch
from tqdm import tqdm
from utils.general import(
    DEVICE,
    PATCHES_VAL_PATH,
    NUM_KNOWN_CLASSES,
    PATCHES_TEST_PATH
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataloader import Satelite_images
from utils.openpca import fit_pca_model, fit_quantiles, pred_pixelwise
import gc

def validate_openpca(model, dl):
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


def test_openpca(model_full, net, dl, t, n):
    net.eval()
    confusion_matrix = np.zeros((8,8))
    th = model_full['thresholds'][t]
    with torch.no_grad():
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                with torch.no_grad():
                    image = image.float().unsqueeze(0).to(DEVICE)
                    out, lay1, lay2 = net(image)
                    soft_out = F.softmax(out, dim = 1)
                    pred = torch.argmax(soft_out, dim = 1)
                    pred = pred.squeeze(0).to("cpu").numpy().ravel()
                    mask = mask.numpy().ravel()
                    out = torch.cat([out.squeeze(0), lay1.squeeze(0), lay2.squeeze(0)], 0).to("cpu")
                    out = out.permute(1, 2, 0).contiguous().view(out.size(1) * out.size(2), out.size(0)).numpy()
                    preds_post, scores = pred_pixelwise(model_full, out, pred, NUM_KNOWN_CLASSES, model_full['thresholds'][t])
                    for i in range(len(mask)):
                        if mask[i] == 8:
                            continue
                        confusion_matrix[mask[i]][preds_post[i]] += 1
                    pbar.update(1)
    precision = np.zeros(8)
    recall = np.zeros(8)
    f1_score = np.zeros(8)
    for i in range(8):
        precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    with open(f'results/result_2_{t}_openpca_{n}.txt', 'w') as file:
        file.write(f"A acuracia é {accuracy}\n")
        for i in range(8):
            file.write(f"A precisão para a classe {i} é {precision[i]}\n")  
            file.write(f"A recall para a classe {i} é {recall[i]}\n")  
            file.write(f"O F1 Score para a classe {i} é {f1_score[i]}\n")
    gc.collect()