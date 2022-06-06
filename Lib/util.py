# importing sys
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '/home/chengwei/USC/MCL/NLP/Code/Sentence_similarity/WMD/Lib')
import lib_dependency as lib_dep
import networkx as nx
import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix, load_npz, save_npz
from collections import defaultdict, Counter
from math import log
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import normalize

from scipy.spatial.distance import cdist
from pyemd import emd, emd_with_flow
import logging
from gensim.corpora.dictionary import Dictionary
from numpy import (
    dot, float32 as REAL, double, array, zeros, vstack,
    ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

logger = logging.getLogger(__name__)

def safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def data2textandlabel(data):
    texts = []
    labels = []
    for label, sample in data:
        text = []
        for sent in sample:
            for word in sent:
                text.append(word['text'].lower())
        texts.append(text)
        labels.append(label)
    return texts, labels

def pmi(mat, positive=True):
    col_totals = np.sum(mat, axis=0)
    total = np.sum(mat)
    row_totals = np.sum(mat, axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = mat / expected
    df[np.isnan(df)] = 0.0
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

def pmi_sparse(mat, positive=True):
    col_totals = np.sum(mat, axis=0)
    total = np.sum(mat)
    row_totals = np.sum(mat, axis=1)
    pmimat = []
    for i,j,v in tqdm(zip(mat.row, mat.col, mat.data), total=len(mat.row)):
        pmi_value = col_totals[0,j]*row_totals[i,0]/total
        pmi_value = v / pmi_value
        pmimat.append(pmi_value)
    del col_totals, row_totals, total
    pmimat = np.array(pmimat)
    pmimat[np.isnan(pmimat)] = 0.0
    with np.errstate(divide='ignore'):
        pmimat = np.log(pmimat)
    pmimat[np.isinf(pmimat)] = 0.0  # log(0) = 0
    if positive:
        pmimat[pmimat < 0] = 0.0
    we_pmi = coo_matrix((pmimat, (mat.row, mat.col)), shape = mat.shape)
    return we_pmi

def text2rep(texts, wordembedding, dim=50):
    reps = []
    for text in tqdm(texts):
        text_rep = []
        for word in text:
            if word in wordembedding:
                text_rep.append(wordembedding[word])
        if text_rep:
            text_rep = np.average(text_rep, 0)
        else:
            text_rep = np.zeros(dim)
        reps.append(text_rep)
    reps = np.vstack(reps)
    return reps

def pmi_sparse_cds(mat, cds=0.75, positive=True):
    mat = mat.power(cds)
    col_totals = np.sum(mat, axis=0)
    total = np.sum(mat)
    row_totals = np.sum(mat, axis=1)
    pmimat = []
    for i,j,v in tqdm(zip(mat.row, mat.col, mat.data), total=len(mat.row)):
        pmi_value = col_totals[0,i]*row_totals[j,0]/total
        pmi_value = v / pmi_value
        pmimat.append(pmi_value)
    del col_totals, row_totals, total
    pmimat = np.array(pmimat)
    pmimat[np.isnan(pmimat)] = 0.0
    with np.errstate(divide='ignore'):
        pmimat = np.log(pmimat)
    pmimat[np.isinf(pmimat)] = 0.0  # log(0) = 0
    if positive:
        pmimat[pmimat < 0] = 0.0
    we_pmi = coo_matrix((pmimat, (mat.row, mat.col)), shape = mat.shape)
    return we_pmi

def kendell_score(score_better, score_worse):
    total = len(score_better)
    correct = sum(np.array(score_better) > np.array(score_worse))
    incorrect = total - correct
    return (correct - incorrect)/total

def kendell_dist(dist_better, dist_worse):
    total = len(dist_better)
    correct = sum(np.array(dist_better) < np.array(dist_worse))
    incorrect = total - correct
    return (correct - incorrect)/total

def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun

def wmd(x, y):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x = x.astype('float64')
    y = y.astype('float64')

    total_len = len(x) + len(y)
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    z = np.concatenate((x,y), axis=0)
    D = np.sqrt(np.square(z[:, None] - z[None, :]).sum(axis=2))
    if np.all(D==0):
        return 1
    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    return emd(p, q, D)
    # return wasserstein_distance(p, q, D)

def wms(x, y):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x = x.astype('float64')
    y = y.astype('float64')

    total_len = len(x) + len(y)
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    z = np.concatenate((x,y), axis=0)
    D = np.sqrt(np.square(z[:, None] - z[None, :]).sum(axis=2))

    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    _, flow = emd_with_flow(p, q, D)
    flow = np.array(flow, dtype=np.float32)
    score = 1./(1. + np.sum(flow * D))#1 - np.sum(flow * dst)
    return score

def wmd_distM(x_sm, y_sm, x_st, y_st, a=0.5):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x_sm, y_sm, x_st, y_st = x_sm.astype('float64'), y_sm.astype('float64'), x_st.astype('float64'), y_st.astype('float64')
    
    total_len = len(x_sm) + len(y_sm)
    p = np.ones(x_sm.shape[0]) / x_sm.shape[0]
    q = np.ones(y_sm.shape[0]) / y_sm.shape[0]
    z_sm = np.concatenate((x_sm, y_sm), axis=0)
    z_st = np.concatenate((x_st, y_st), axis=0)
    D_sm = np.sqrt(np.square(z_sm[:, None] - z_sm[None, :]).sum(axis=2))
    D_st = np.sqrt(np.square(z_st[:, None] - z_st[None, :]).sum(axis=2))

    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    return emd(p, q, (1-a)*D_sm+a*D_st)

def wrd(x, y):
    """WRD（Word Rotator's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x = x.astype('float64')
    y = y.astype('float64')


    total_len = len(x) + len(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    z = np.concatenate((x,y), axis=0)
    # z_norm = np.concatenate((x_norm, y_norm), axis=0)
    z_norm = (z**2).sum(axis=1, keepdims=True)**0.5
    # p = x_norm[:, 0] / x_norm.sum()
    # q = y_norm[:, 0] / y_norm.sum()
    # D = 1 - np.dot(z / z_norm, (z / z_norm).T)
    p = np.divide(x_norm[:, 0], x_norm.sum(), out=np.zeros_like(x_norm[:, 0]), where=x_norm.sum()!=0)
    q = np.divide(y_norm[:, 0], y_norm.sum(), out=np.zeros_like(y_norm[:, 0]), where=y_norm.sum()!=0)
    z_tmp = np.divide(z, z_norm, out=np.zeros_like(z), where=z_norm!=0)
    D = 1 - np.dot(z_tmp, z_tmp.T)

    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    return emd(p, q, D)

def wrs(x, y):
    """1 - WRD
    x.shape=[m,d], y.shape=[n,d]
    """
    return 1 - wrd(x, y)

def wmdistance(embedding, document1, document2, norm=True):
        """Compute the Word Mover's Distance between two documents.

        When using this code, please consider citing the following papers:

        * `Ofir Pele and Michael Werman "A linear time histogram metric for improved SIFT matching"
          <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_
        * `Ofir Pele and Michael Werman "Fast and robust earth mover's distances"
          <https://ieeexplore.ieee.org/document/5459199/>`_
        * `Matt Kusner et al. "From Word Embeddings To Document Distances"
          <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

        Parameters
        ----------
        document1 : list of str
            Input document.
        document2 : list of str
            Input document.
        norm : boolean
            Normalize all word vectors to unit length before computing the distance?
            Defaults to True.

        Returns
        -------
        float
            Word Mover's distance between `document1` and `document2`.

        Warnings
        --------
        This method only works if `pyemd <https://pypi.org/project/pyemd/>`_ is installed.

        If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
        will be returned.

        Raises
        ------
        ImportError
            If `pyemd <https://pypi.org/project/pyemd/>`_  isn't installed.

        """
        # If pyemd C extension is available, import it.
        # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in embedding]
        document2 = [token for token in document2 if token in embedding]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if not document1 or not document2:
            logger.warning("At least one of the documents had no words that were in the vocabulary.")
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed of a single unique token => zero distance.
            return 0.0

        doclist1 = list(set(document1))
        doclist2 = list(set(document2))
        v1 = np.array([embedding.get_vector(token, norm=norm) for token in doclist1])
        v2 = np.array([embedding.get_vector(token, norm=norm) for token in doclist2])
        doc1_indices = dictionary.doc2idx(doclist1)
        doc2_indices = dictionary.doc2idx(doclist2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)

        if abs(np_sum(distance_matrix)) < 1e-8:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents. This is what pyemd expects on input.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

##################### tree structure

# weight
def distance2subsapce(u, v):
    # u: subspace orthonormal matrix
    proj_matrix = np.dot(u,u.T)
    proj = np.dot(proj_matrix, v)
    dif = v - proj
    return np.linalg.norm(dif, 2)/np.linalg.norm(v, 2)

def form_subspace(parsed_samples, model_word, n_comp, thre=300, verbose=True):
    # collect pairwise data for each dependency relation [dep, head]
    data = {}
    for sent in tqdm(parsed_samples):
        for word in sent:
            if not word['deprel'] in data:
                data[word['deprel']] = []
            data[word['deprel']].append(word['text'].lower())
    # compute the subspace
    dep_subspace = {}
    for dep_relation, dep_data in data.items():
        if len(dep_data) >= thre:
            if verbose:
                print('--- processing keeped rel', dep_relation)
                print('len: ', len(dep_data))
            dep_vec = []
            for word in dep_data:
                # make sure words in vocabulary
                if word in model_word:
                    dep_vec.append(model_word[word])
            dep_vec = np.array(dep_vec)
            #print(dep_vec.shape)
            U, s, Vh = np.linalg.svd(np.transpose(dep_vec), full_matrices = False)
            dep_subspace[dep_relation] = U[:,:n_comp]
    return dep_subspace

def vec_word(sent, model_word, dep_subspace, tf, a, b, stop_words=[], rels=[]):
    # G = lib_dep.sentdic2dicgraph(sent)
    # source = [id for id, element in enumerate(sent) if element['deprel']=='root'][0]
    # neighb_dict = nx.single_source_shortest_path_length(G, source) # depth = neighb_dict[id]

    sent_word_vec = []
    weight = []
    for id, w in enumerate(sent):
        if w['text'].lower() in model_word and w['text'].lower() not in stop_words:
            sent_word_vec.append(model_word[w['text'].lower()])
            idf = 1
            if tf:
                if w['text'].lower() in tf.vocabulary_:
                    idf = tf.idf_[tf.vocabulary_[w['text'].lower()]]

            if w['deprel'] in rels:
                wei = a
            else: wei = 1

            if w['deprel'] in dep_subspace:
            # if w['deprel'] in dep_subspace:
                u = dep_subspace[w['deprel']]
                v = model_word[w['text'].lower()]
                distance = distance2subsapce(u, v)
                weight.append(wei*np.exp(b*distance)*idf)
            else: weight.append(wei*idf)
    if not sent_word_vec: 
        sent_word_vec = [np.ones(300)]
        weight.append(1)
    return np.array(sent_word_vec), weight

def weighted_distance_matrix(x, y, w1, w2, l2_dist=True, norm_flag=False):
    x = x.astype('float64')
    y = y.astype('float64')

    total_len = len(x) + len(y)
    p = (np.ones(x.shape[0]) / x.shape[0])*w1
    q = (np.ones(y.shape[0]) / y.shape[0])*w2
    p = p/np.sum(p)
    q = q/np.sum(q)
    z = np.concatenate((x,y), axis=0)
    # D = np.sqrt(np.square(z[:, None] - z[None, :]).sum(axis=2))
    if norm_flag:
        z = normalize(z)
    if l2_dist:
        D = euclidean_distances(z,z)
    else:
        D = cosine_distances(z,z)

    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    return p,q,D

def wmd_treeimp(sent1, sent2, model_word, dep_subspace, a=1.5, b=1, tf=[], stop_words=[], rels=[]):
    s_vec1, w1 = vec_word(sent1, model_word, dep_subspace, tf, a, b, stop_words, rels)
    s_vec2, w2 = vec_word(sent2, model_word, dep_subspace, tf, a, b, stop_words, rels)
    p,q,dist = weighted_distance_matrix(s_vec1, s_vec2, w1, w2)
    if np.all(dist==0):
        return 1
    return emd(p,q,dist)


# idf
def get_idf(parsed_samples, a):
    # collect pairwise data for each dependency relation [dep, head]
    data = {}
    for sent in tqdm(parsed_samples):
        for word in sent:
            if not word['deprel'] in data:
                data[word['deprel']] = []
            data[word['deprel']].append(word['text'].lower())
    # compute idf
    dep_idf = {}
    for dep_relation, dep_data in data.items():
        idf_count = Counter()
        num_docs = len(dep_data)

        idf_count.update(dep_data)

        # idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        # idf_dict.update({w:log((num_docs+1)/(c+1)) for (w, c) in idf_count.items()})
        idf_dict = defaultdict(lambda : a/(a+(1/num_docs)))
        idf_dict.update({w:a/(a+(c/num_docs)) for (w, c) in idf_count.items()})   
        dep_idf[dep_relation] = idf_dict

    # data = []
    # for sent in tqdm(parsed_samples):
    #     for word in sent:
    #         data.append(word['text'].lower())
    # # compute idf
    # dep_idf = {}
    # idf_count = Counter()
    # num_docs = len(data)
    # idf_count.update(data)
    # # idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    # # idf_dict.update({w:log((num_docs+1)/(c+1)) for (w, c) in idf_count.items()})
    # idf_dict = defaultdict(lambda : a/(a+(1/num_docs)))
    # idf_dict.update({w:a/(a+(c/num_docs)) for (w, c) in idf_count.items()})   
    # dep_idf = idf_dict
    return dep_idf

def vec_word_widf(sent, model_word, tf, a, stop_words=[], rels=[]):
    sent_word_vec = []
    weight = []
    for id, w in enumerate(sent):
        if w['text'].lower() in model_word and w['text'].lower() not in stop_words:
            sent_word_vec.append(model_word[w['text'].lower()])
            if w['deprel'] in rels:
                wei = a
            else: wei = 1
            if w['deprel'] in tf:
                idf_dict = tf[w['deprel']]
                idf = idf_dict[w['text'].lower()]
                weight.append(wei*idf)
            else: weight.append(0)
            # idf = tf[w['text'].lower()]
            # weight.append(wei*idf)
    if not sent_word_vec: 
        sent_word_vec = [np.ones(300)]
        weight.append(1)
    return np.array(sent_word_vec), weight

def wmd_widf(sent1, sent2, model_word, tf, a=1, stop_words=[], rels=[]):
    s_vec1, w1 = vec_word_widf(sent1, model_word, tf, a, stop_words, rels)
    s_vec2, w2 = vec_word_widf(sent2, model_word, tf, a, stop_words, rels)
    p,q,dist = weighted_distance_matrix(s_vec1, s_vec2, w1, w2)
    if np.all(dist==0):
        return 1
    return emd(p,q,dist)

def vec_word_idf(sent, model_word, dep_idf, a, b, stop_words=[], rels=[]):
    G = lib_dep.sentdic2dicgraph(sent)
    source = [id for id, element in enumerate(sent) if element['deprel']=='root'][0]
    neighb_dict = nx.single_source_shortest_path_length(G, source) # depth = neighb_dict[id]

    sent_word_vec = []
    weight = []
    for id, w in enumerate(sent):
        if w['text'].lower() in model_word and w['text'].lower() not in stop_words:
            sent_word_vec.append(model_word[w['text'].lower()])
            if w['deprel'] in rels:
                wei = a
            else: wei = 1
            if w['deprel'] in dep_idf:
                idf_dict = dep_idf[w['deprel']]
                idf = idf_dict[w['text'].lower()]
                weight.append(wei*b*idf)
            else: weight.append(wei*idf)
    if not sent_word_vec: 
        sent_word_vec = [np.ones(300)]
        weight.append(1)
    return np.array(sent_word_vec), weight

def wmd_treeiidf(sent1, sent2, dep_idf, a=1.5, b=1, stop_words=[], rels=[]):
    s_vec1, w1 = vec_word_idf(sent1, dep_idf, a, b, stop_words, rels)
    s_vec2, w2 = vec_word_idf(sent2, dep_idf, a, b, stop_words, rels)
    p,q,dist = weighted_distance_matrix(s_vec1, s_vec2, w1, w2)
    if np.all(dist==0):
        return 1
    return emd(p,q,dist)

# distance matrix
def distance_matrix(x, y, l2_dist=True, norm_flag=False):
    x = x.astype('float64')
    y = y.astype('float64')

    total_len = len(x) + len(y)
    p = (np.ones(x.shape[0]) / x.shape[0])
    q = (np.ones(y.shape[0]) / y.shape[0])
    z = np.concatenate((x,y), axis=0)
    # D = np.sqrt(np.square(z[:, None] - z[None, :]).sum(axis=2))
    if norm_flag:
        z = normalize(z)
    if l2_dist:
        D = euclidean_distances(z,z)
    else:
        D = cosine_distances(z,z)
    p = np.pad(p, (0, total_len-len(p)), 'constant')
    q = np.pad(q, (total_len-len(q), 0), 'constant')
    return p,q,D

def we_tree(sent, model_word, stop_words=[], rel_kept=[], tree= 'b', tf={}, hop_num=1):
    sent_word_all = [w['text'].lower() for w in sent]

    te_total = []
    id_total = []
    w_weight = []

    sent_word  = []
    id_sent = []
    for id, word in enumerate(sent):
        if word['text'].lower() not in stop_words:
            if word['text'].lower() in model_word:
                sent_word.append(word['text'].lower()) 
                id_sent.append(id)
                if word['text'].lower() in tf: w_weight.append(tf[word['text'].lower()])
                else: w_weight.append(1)
    sent_vec = np.array([model_word[w] for w in sent_word])
    if len(sent_vec)==0: sent_vec = np.zeros((1, 300))

    G = lib_dep.sentdic2dicgraph(sent)
    if tree == 'b':
        subtree = lib_dep.node_bottomsubtree(G)
    elif tree == 't':
        source = [id for id, element in enumerate(sent) if element['deprel']=='root'][0]
        subtree = lib_dep.node_topsubtree(G, source)
    elif tree == 's':
        subtree = lib_dep.node_smallsubtree(G, hop_num)
    elif tree == 'c':
        G = lib_dep.sentdic2undicgraph(sent)
        subtree = lib_dep.node_smallsubtree(G, hop_num)
    else:
        subtree = []

    for subt in subtree:
        if rel_kept:
            if not sent[subt[-1]]['deprel'] in rel_kept: continue
        subt_id = [idx for idx in subt if (sent_word_all[idx] in model_word) and (sent_word_all[idx] not in stop_words)]
        subt_word = [model_word[sent_word_all[idx]] for idx in subt if (sent_word_all[idx] in model_word) and (sent_word_all[idx] not in stop_words)]
        subt_weight = [w_weight[id_sent.index(idx)] for idx in subt if (sent_word_all[idx] in model_word) and (sent_word_all[idx] not in stop_words)]
        # subt_word = [model[sent_word[idx]] for idx in subt if sent_word[idx] in model]
        if len(subt_word) >= 2: # avoid out of vacabulary
            te_total.append(np.average(subt_word, axis=0, weights=subt_weight))
            id_total.append(subt_id)

    return sent_vec, te_total, id_total, id_sent, w_weight

from sklearn.metrics.pairwise import euclidean_distances
def tree_wmd(sent1, sent2, model_word, a = 1, stop_words=[], rel_kept=[], tree= 'b', tf={}, hop_num=1, l2_dist=True, norm_flag=False):
    # print("===================")
    sent1_vec, te1_total, id1_total, id1_sent, _ = we_tree(sent1, model_word, stop_words, rel_kept, tree, tf, hop_num)
    sent2_vec, te2_total, id2_total, id2_sent, _ = we_tree(sent2, model_word, stop_words, rel_kept, tree, tf, hop_num)

    # if len(te1_total)>=1 and len(te2_total)>=1:
    #     sent1_vec = np.concatenate((sent1_vec, np.array(te1_total)), axis=0)
    #     sent2_vec = np.concatenate((sent2_vec, np.array(te2_total)), axis=0)

    p,q,dist = distance_matrix(sent1_vec, sent2_vec, l2_dist, norm_flag)
    if len(te1_total)>=1 and len(te2_total)>=1:
        if norm_flag:
            te1_total = normalize(te1_total)
            te2_total = normalize(te2_total)
        if l2_dist:
            dist_tree = euclidean_distances(te1_total, te2_total)
        else:
            dist_tree = cosine_distances(te1_total, te2_total)
        dist_w = np.zeros((len(sent1_vec), len(sent2_vec)))
        dist_count = np.zeros((len(sent1_vec), len(sent2_vec)))
        for tree_id1, id1 in enumerate(id1_total):
            for tree_id2, id2 in enumerate(id2_total):
                tree_distance = dist_tree[tree_id1, tree_id2]
                if tree == 'c':
                    dist_w[id1_sent.index(id1[-1]),id2_sent.index(id2[-2])] += tree_distance
                    dist_count[id1_sent.index(id1[-1]),id2_sent.index(id2[-2])] += 1
                else:
                    for i1 in id1:
                        for i2 in id2:
                            dist_w[id1_sent.index(i1),id2_sent.index(i2)] += tree_distance
                            dist_count[id1_sent.index(i1),id2_sent.index(i2)] += 1
        dist_w = safe_divide(dist_w, dist_count) # average

        dist_w[dist_w==0] = np.max(dist_w)
        dist[len(sent1_vec):,:len(sent1_vec)] += a*dist_w.T
        dist[:len(sent1_vec), len(sent1_vec):] +=  a*dist_w

    if np.all(dist==0):
        return 1
    return emd(p, q, dist)

# combine
def combine(sent1, sent2, model_word, dep_subspace, a = 1, b=1, c=0, stop_words=[], rel_kept=[], tree= 'b', tf={}, hop_num=1, l2_dist=True, 
            norm_flag=False):
    # print("===================")
    sent1_vec, te1_total, id1_total, id1_sent, sent1_weight = we_tree(sent1, model_word, stop_words, rel_kept, tree, tf, hop_num)
    sent2_vec, te2_total, id2_total, id2_sent, sent2_weight = we_tree(sent2, model_word, stop_words, rel_kept, tree, tf, hop_num)

    # s_vec1, w1 = vec_word(sent1, model_word, dep_subspace, tf, b, c, stop_words, rel_kept)
    # s_vec2, w2 = vec_word(sent2, model_word, dep_subspace, tf, b, c, stop_words, rel_kept)
    p,q,dist = weighted_distance_matrix(sent1_vec, sent2_vec, sent1_weight, sent2_weight, l2_dist, norm_flag)

    if len(te1_total)>=1 and len(te2_total)>=1:
        if norm_flag:
            te1_total = normalize(te1_total)
            te2_total = normalize(te2_total)
        if l2_dist:
            dist_tree = euclidean_distances(te1_total, te2_total)
        else:
            dist_tree = cosine_distances(te1_total, te2_total)
        dist_w = np.zeros((len(sent1_vec), len(sent2_vec)))
        dist_count = np.zeros((len(sent1_vec), len(sent2_vec)))
        for tree_id1, id1 in enumerate(id1_total):
            for tree_id2, id2 in enumerate(id2_total):
                tree_distance = dist_tree[tree_id1, tree_id2]
                if tree == 'c':
                    dist_w[id1_sent.index(id1[-1]),id2_sent.index(id2[-2])] += tree_distance
                    dist_count[id1_sent.index(id1[-1]),id2_sent.index(id2[-2])] += 1
                else:
                    for i1 in id1:
                        for i2 in id2:
                            dist_w[id1_sent.index(i1),id2_sent.index(i2)] += tree_distance
                            dist_count[id1_sent.index(i1),id2_sent.index(i2)] += 1
        dist_w = safe_divide(dist_w, dist_count) # average

        dist[len(sent1_vec):,:len(sent1_vec)] += a*dist_w.T
        dist[:len(sent1_vec), len(sent1_vec):] +=  a*dist_w
        dist_w[dist_w==0] = np.max(dist_w)

    if np.all(dist==0):
        return 1
    return emd(p, q, dist)  