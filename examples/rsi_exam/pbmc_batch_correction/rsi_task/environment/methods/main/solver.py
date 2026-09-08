"""Visible PCA baseline using a TF-IDF-normalized binary peak matrix."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

N_COMPONENTS = 50
SEED = 2002


def _tfidf(X):
    X = X.tocsr().astype(np.float64)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(1.0 + X.shape[0] / (1.0 + df))
    Xt = X.multiply(idf)
    return normalize(Xt.tocsr(), norm="l2", axis=1)


def embed(atac, batch_labels, peak_names):
    Xt = _tfidf(atac)
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
    return svd.fit_transform(Xt).astype(np.float64)
