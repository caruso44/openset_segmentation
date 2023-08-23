from test_open.test_openfcn import test_openfcn, get_weibull_model
from test_open.test_openpca import test_openpca, validate_openpca
import torch
from utils.general import DEVICE

if __name__ == "__main__":
    model = torch.load("open_set_model_UNET.pth")
    model = model.to(DEVICE)
    #check_model_closed() # verificar o modelo atraves de uma abordagem em conjunto fechado
    #get_lists(model) # determinar e salvar a lista de distâncias e médias
    print("iniciando o teste")
    ths = [0.99999, 0.9999999, 0.99999999, 0.999999999]
    tails = [1000]
    for tail in tails:
        weibull_model = get_weibull_model(tail) # determinar e salvar o modelo de weibull
        for th in ths:
            test_openfcn(model, weibull_model, th, tail) # testar o modelo em conjunto aberto