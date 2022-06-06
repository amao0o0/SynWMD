import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
import pickle

def get_svd(words, model, n):
    vectors = [model[w] for w in words if w in model]
    svd = TruncatedSVD(n_components=n, random_state=0).fit(vectors)	
    return svd

def rm_pc(svd, vectors):
    proj = lambda a, b: a.dot(b.transpose()) * b
    # remove the weighted projections on the common discourse vectors
    for i in range(svd.n_components):
        lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
        pc = svd.components_[i]
        vectors = [ v_s - lambda_i * proj(v_s, pc) for v_s in vectors]
    return vectors

def compute_kernel_bias(vecs):
    """
    y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    vecs = np.array(vecs)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    return W, -mu

def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)

def transform_and_normalize_torch(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = torch.matmul((vecs + bias), kernel)
    return normalize(vecs)

def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def load_dataset(path):
    """
    loading AllNLI dataset.
    """
    senta_batch, sentb_batch = [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            items = line.strip().split('\t')
            senta, sentb = items[-3], items[-2]
            senta_batch.append(senta)
            sentb_batch.append(sentb)
    return senta_batch, sentb_batch

def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(path, 'wb') as f:
        pickle.dump(whiten, f)
    return

def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias