import numpy as np
from sklearn import decomposition


def fit_quantiles(model_list, feat_np, prds_np, num_known_classes):

    # Acquiring scores for training set sample.
    scores = np.zeros_like(prds_np, dtype=np.cfloat)
    for c in range(num_known_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            scores[feat_msk] = model_list[c].score_samples(feat_np[feat_msk, :])

    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scr_thresholds = np.quantile(scores, thresholds).tolist()
    return scr_thresholds
    
def fit_pca_model(feat_np, true_np, prds_np, cl, n_components):
    
    model = decomposition.PCA(n_components=n_components, random_state=12345)
    cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]
    perm = np.random.permutation(cl_feat_flat.shape[0])
    
    if perm.shape[0] > 32768:
        cl_feat_flat = cl_feat_flat[perm[:32768], :]
    model.fit(cl_feat_flat)
    return model

def pred_pixelwise(model_full, feat_np, prds_np, num_known_classes, threshold):
    
    scores = np.zeros_like(prds_np, dtype=np.cfloat)
    for c in range(num_known_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            scores[feat_msk] = model_full['generative'][c].score_samples(feat_np[feat_msk, :])
        
    prds_np[scores < threshold] = num_known_classes
    return prds_np, scores