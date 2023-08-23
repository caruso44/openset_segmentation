import numpy as np
import torch
from tqdm import tqdm
from utils.general import DEVICE, NUM_KNOWN_CLASSES, PATCHES_TEST_PATH, PATCHES_VAL_PATH
import torch.nn.functional as F
from utils.dataloader import Satelite_images
from utils.openmax import recalibrate_scores
from utils.utils import get_distances, get_mean
from utils.evt import weibull_tailfitting

def print_confusion_matrix(confusion_matrix, size, th, tail):
    precision = np.zeros(size)
    recall = np.zeros(size)
    f1_score = np.zeros(size)
    for i in range(size):
        precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    with open(f'results/result_{th}_tail_{tail}.txt', 'w') as file:
        file.write(f"A acuracia é {accuracy}\n")
        for i in range(size):
            file.write(f"A precisão para a classe {i} é {precision[i]}\n")  
            file.write(f"A recall para a classe {i} é {recall[i]}\n")  
            file.write(f"O F1 Score para a classe {i} é {f1_score[i]}\n")  


def get_lists(net):
    val_loader = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    print("Calculando as medias")
    mean =  get_mean(val_loader, net)
    #mean = np.load("mean.npy")
    print("Calculando as distâncias")
    dist = get_distances(val_loader, mean, net)
    

def get_weibull_model(tail):
    print("Iniciando a determinação do modelo weibull")
    dist_list = np.load("distances_eucos.npy", allow_pickle= True)
    mean_list = np.load("mean_eucos.npy")
    weibull_model = weibull_tailfitting(mean_list, dist_list, NUM_KNOWN_CLASSES, tailsize=tail)
    return weibull_model

def get_confusion_matrix(model, dl):
    model.eval()
    confusion_matrix = np.zeros((7,7))

    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)
                distribution = model(image)
                predictions = F.softmax(distribution, dim = 1)
                pbar.update(1)
                predictions = torch.argmax(predictions, dim = 1)
                mask = mask.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                for i in range(64):
                    for j in range(64):
                        if mask[i][j] != 7:
                            confusion_matrix[mask[i][j]][predictions[i][j]] += 1
    print_confusion_matrix(confusion_matrix, 7)

def test_openfcn(net, weibull_model, th, tail):
    dl = Satelite_images(PATCHES_TEST_PATH, "_test.npy")
    net.eval() 
    confusion_matrix = np.zeros((8, 8))
    with tqdm(total=len(dl)) as pbar:
        with torch.no_grad():
            for image, label in dl:
                label = label.numpy()
                label = label.reshape(label.shape[0] * label.shape[1])
                image = image.unsqueeze(0).to(DEVICE)
                output = net(image)
                output = output[0]
                output_soft = F.softmax(output, dim = 1)
                output_soft = output_soft.squeeze(0).to("cpu")
                output = output.squeeze(0).to("cpu")
                probs = recalibrate_scores(
                    weibull_model, output, output_soft, NUM_KNOWN_CLASSES, NUM_KNOWN_CLASSES, 'eucos'
                )
                for i in range(len(label)):
                    pred = np.argmax(probs[i])
                    if label[i] == 8:
                        continue
                    if pred != 7 and probs[i, pred] < th:
                        confusion_matrix[label[i], 7] += 1
                    else:
                        confusion_matrix[label[i], pred] += 1
                pbar.update(1)
        print_confusion_matrix(confusion_matrix, 8, th, tail)

