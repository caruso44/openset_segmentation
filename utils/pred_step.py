import numpy as np
from utils.general import NUM_KNOWN_CLASSES
import torch.nn.functional as F
from utils.openmax import recalibrate_scores
import torch
from utils.openpca import pred_pixelwise


def openFCN(tensor, model, weibull_model, output, i, j, t):
    out = model(tensor)
    out = out[0]
    out_soft = F.softmax(out, dim = 1)
    out_soft = out_soft.squeeze(0).to("cpu")
    out = out.squeeze(0).to("cpu")
    probs = recalibrate_scores(
    weibull_model, out, out_soft, NUM_KNOWN_CLASSES, NUM_KNOWN_CLASSES, 'eucos'
)
    for k in range(len(probs)):
        row = int(k/64)
        col = int(k % 64)
        pred = np.argmax(probs[k])
        if probs[k, pred] < 0.9999999:
            pred = 7
        output[i+row, j+col] = pred
    
    return output


def openPCS(image, model, model_full, output, i, j, t):
    out, lay1, lay2 = model(image)
    soft_out = F.softmax(out, dim = 1)
    pred = torch.argmax(soft_out, dim = 1)
    pred = pred.squeeze(0).to("cpu").numpy().ravel()
    out = torch.cat([out.squeeze(0), lay1.squeeze(0), lay2.squeeze(0)], 0).to("cpu")
    out = out.permute(1, 2, 0).contiguous().view(out.size(1) * out.size(2), out.size(0)).numpy()
    preds_post, scores = pred_pixelwise(model_full, out, pred, NUM_KNOWN_CLASSES, model_full['thresholds'][t])
    for k in range(len(preds_post)):
        row = int(k/64)
        col = int(k % 64)
        output[i+row, j+col] = preds_post[k]
    return output


def openPixel(tensor, model, model2, output, i, j, t):
    out = model(tensor)
    out = out[0]
    out_soft = F.softmax(out, dim = 1)
    predictions = torch.argmax(out_soft, dim = 1)
    out_soft = out_soft.squeeze().to("cpu")
    predictions = predictions.squeeze(0).to("cpu")
    row = predictions.size()[0]
    col = predictions.size()[1]
    for u in range(row):
        for v in range(col):
            if out_soft[predictions[u][v], u, v] > 0.9999999: 
                output[i+u, j+v] = predictions[u][v]
            else:
                output[i+u, j+v] = 7
            
    return output