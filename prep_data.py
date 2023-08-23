import argparse
import pathlib
import numpy as np
from osgeo import gdal, gdalconst
import os
from utils.utils_prep import load_opt_image, load_label_image, save_dict
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
import sys
from utils.general import IMAGE_PATH, TRAIN_PATH, TEST_PATH, VALIDATION_SPLIT, PATCH_SIZE, PATCH_OVERLAP, REMOVED_CLASSES, DISCARDED_CLASS, PATH, DISCARDED_CLASS_2



outfile = f'data2/preparation.txt'
with open(outfile, 'w') as sys.stdout:
    img = load_opt_image(str(IMAGE_PATH))
    train_label = load_label_image(str(TRAIN_PATH))
    test_label = load_label_image(str(TEST_PATH))
    shape = train_label.shape
    for val in np.unique(train_label):
        print(f'Class {val}: N Train samples = {(train_label==val).sum():,}, N Test samples = {(test_label==val).sum():,}')

    #remove classes
    for r_class_id in REMOVED_CLASSES:
        train_label[train_label==r_class_id] = 98
        test_label[test_label==r_class_id] = 98
    
    uni = np.unique(train_label)
    new_labels = np.arange(uni.shape[0])
    remap_dict = dict(zip(new_labels, uni))

    #remap labels
    train_label_remap = np.empty_like(train_label, dtype=np.uint8)
    test_label_remap = np.empty_like(test_label, dtype=np.uint8)
    
    for dest_key, source_key in remap_dict.items():
        train_idx = train_label == source_key
        train_label_remap[train_idx] = dest_key

        test_idx = test_label == source_key
        test_label_remap[test_idx] = dest_key
    save_dict(remap_dict, f'{PATH}/map.data')
    print(remap_dict)

    del train_label, test_label

    patch_size = PATCH_SIZE
    train_step = int((1-PATCH_OVERLAP)*patch_size)


    idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

    label_patches = view_as_windows(train_label_remap, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
    idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
    print(idx_patches)
    keep_patches = np.mean(np.logical_and((label_patches != 0), np.logical_and((label_patches != DISCARDED_CLASS), (label_patches != DISCARDED_CLASS_2))), axis=(1,2)) >= 0.1
    idx_patches = idx_patches[keep_patches]
    np.random.seed(123)
    np.random.shuffle(idx_patches)
    n_patches = idx_patches.shape[0]
    n_val = int(VALIDATION_SPLIT * n_patches)
    train_idx_patches = idx_patches[n_val:]
    val_idx_patches = idx_patches[:n_val]
    np.save(f'{PATH}/OPT_img.npy', img[:,:,:4])
    np.save(f'{PATH}/LIDAR_img.npy', img[:,:,4:])
    np.save(f'{PATH}/LABEL_train.npy', train_label_remap)
    np.save(f'{PATH}/LABEL_test.npy', test_label_remap)

    np.save(f'{PATH}/train_patches.npy', train_idx_patches)
    np.save(f'{PATH}/val_patches.npy', val_idx_patches)

    idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)
    label_patches = view_as_windows(test_label_remap, (PATCH_SIZE, PATCH_SIZE), train_step).reshape((-1, PATCH_SIZE, PATCH_SIZE))
    idx_patches = view_as_windows(idx_matrix, (PATCH_SIZE, PATCH_SIZE), train_step).reshape((-1, PATCH_SIZE, PATCH_SIZE))
    keep_patches = np.mean(np.logical_and((label_patches != 0), (label_patches != DISCARDED_CLASS_2)), axis=(1,2)) >= 0.1
    idx_patches = idx_patches[keep_patches]
    np.save(f'{PATH}/test_patches.npy', idx_patches)