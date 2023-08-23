import torch

#alterar os diretórios
IMAGE_PATH = ""
TRAIN_PATH = ""
TEST_PATH = ""
PATH = ''
PATCHES_PATH = ""
PATCHES_VAL_PATH = ""
PATCHES_TEST_PATH = ""
VALIDATION_SPLIT = 0.3
PATCH_OVERLAP = 0.8
PATCH_SIZE = 64
REMOVED_CLASSES = [10, 11]
DISCARDED_CLASS = 7
DISCARDED_CLASS_2 = 8
LEARNING_RATE = 1e-5
EPOCHS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEN_VECTOR = 200000

IMAGE_SIZE = 64

NUM_KNOWN_CLASSES = 7
BATCH_SIZE = 64
MAP_COLOR = ['white', 'green', 'blue', 'yellow', 'purple', 'black', 'grey', 'red']

COLOR_TO_RGB = {
    'white' : [1.0, 1.0, 1.0],
    'green' : [0.0, 1.0, 0.0],
    'blue' : [0.0, 0.0, 1.0],
    'yellow' : [1.0, 1.0, 0.0],
    'purple' : [1.0, 0.0, 1.0],
    'black' : [0.0, 0.0, 0.0],
    'grey' : [0.5, 0.5, 0.5],
    'red' : [1.0, 0.0, 0.0]
}