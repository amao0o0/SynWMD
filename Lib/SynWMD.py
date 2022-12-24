#%%
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import linalg as LA
import torch
import string
import os
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain
from tqdm.auto import tqdm, trange
import networkx as nx

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

import stanza
from Lib import lib_dependency as lib_dep
from Lib import whiten as pcr

from transformers import WEIGHTS_NAME, AutoTokenizer, AutoModel
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if os.environ.get('SynWMD_MODEL'):
    model_name = os.environ.get('SynWMD_MODEL')
else:
    model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
model.to(device)
model.eval()

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',
                    tokenize_pretokenized=True,
                    tokenize_no_ssplit=True,
                    verbose = True)
#%%
def truncate(tokens):
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0:(tokenizer.model_max_length - 2)]
    return tokens

def process(a):
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a))+["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

def get_weight_dict(arr, a, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : a/(a+(1/num_docs)))
    idf_dict.update({w:log(a/(a+(c/num_docs))+1) for (w, c) in idf_count.items()})   
    return idf_dict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        result = model(x.to(device), attention_mask = attention_mask.to(device))
    if model_name == 'distilbert-base-uncased':
        return result[1] 
    else:
        return result[2] 


def collate_idf(arr, tokenize, numericalize,
                pad="[PAD]"):
    
    tokens = [["[CLS]"]+truncate(tokenize(a))+["[SEP]"] for a in arr]  
    arr = [numericalize(a) for a in tokens]
    
    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)

    return padded, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer,
                       batch_size=-1):

    padded_sens, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, tokens

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def batched_cdist_l2(x1, x2):
    '''
    l2 distance
    '''
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def batched_cdist_cos(x1, x2):
    '''
    cos distance
    '''
    x1_norm = _safe_divide(x1, x1.norm(dim=-1, keepdim=True))
    x2_norm = _safe_divide(x2, x2.norm(dim=-1, keepdim=True))
    res = 1-torch.bmm(x1_norm, x2_norm.transpose(-2,-1)).clamp_min_(1e-30)
    return res

def embedding_layer(embedding_input, layer):
    if layer == 'first_last':
        embedding_output = (embedding_input[-1] + embedding_input[1])/2       
    elif layer == 'last': 
        embedding_output = embedding_input[-1]
    elif layer == 'last2': 
        embedding_output = (embedding_input[-1] + embedding_input[-2])/2
    elif isinstance(layer, int):
        embedding_output = embedding_input[layer]
    return embedding_output

def whitening_prepare(refs, hyps, layer, batch_size=64, stop_words=[]):
    embeddings = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        
        ref_embedding, ref_lens, ref_masks, ref_tokens = get_bert_embedding(batch_refs, model, tokenizer)
        hyp_embedding, hyp_lens, hyp_masks, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer)

        ref_embedding = embedding_layer(ref_embedding, layer)
        hyp_embedding = embedding_layer(hyp_embedding, layer)

        ref_embedding = ref_embedding.double().cpu().numpy()
        hyp_embedding = hyp_embedding.double().cpu().numpy()

        bz = len(ref_tokens)
        for i in range(bz):  
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                if w in stop_words or '##' in w 
                                or w in set(string.punctuation)]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                if w in stop_words or '##' in w
                                or w in set(string.punctuation)]
          
            ref_embedding[i, ref_ids,:] = 0                        
            hyp_embedding[i, hyp_ids,:] = 0

        ref_embedding = ref_embedding.reshape(-1,ref_embedding.shape[-1])
        hyp_embedding = hyp_embedding.reshape(-1,hyp_embedding.shape[-1])

        idx = np.argwhere(np.all(ref_embedding[..., :] == 0, axis=1))
        ref_embedding = np.delete(ref_embedding, idx, axis=0)
        idx = np.argwhere(np.all(hyp_embedding[..., :] == 0, axis=1))
        hyp_embedding = np.delete(hyp_embedding, idx, axis=0)

        embeddings.append(ref_embedding)
        embeddings.append(hyp_embedding)
        
    embeddings = np.vstack(embeddings)
    print("#whiten info:", embeddings.shape)
    # idx = np.argwhere(np.all(embeddings[..., :] == 0, axis=1))
    # embeddings = np.delete(embeddings, idx, axis=0)
    kernel, bias = pcr.compute_kernel_bias(embeddings)

    return kernel, bias

def SynWMD(refs, hyps, word2weight, stop_words=[], batch_size=256, whiten_flag = True, 
            l2_dist = True, tree = 't', a=1, rel_kept=[], hop_num=1, layer='last',
            pre_whiten=False, pre_kernel=[], pre_bias=[], dist_flag = False):
    # num_ngram = 0
    preds = []
    # white
    if whiten_flag and not pre_whiten:
        kernel, bias = whitening_prepare(refs, hyps, layer,
                        batch_size=64, stop_words=stop_words)
    if pre_whiten:
        kernel, bias = pre_kernel, pre_bias

    for batch_start in tqdm(range(0, len(refs), batch_size)):
    # for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        
        ref_embedding, ref_lens, ref_masks, ref_tokens = get_bert_embedding(batch_refs, model, tokenizer)
        hyp_embedding, hyp_lens, hyp_masks, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer)

        ref_w, hyp_w = torch.zeros(ref_masks.size()), torch.zeros(hyp_masks.size())

        ref_embedding = embedding_layer(ref_embedding, layer)
        hyp_embedding = embedding_layer(hyp_embedding, layer)

        if whiten_flag:
            ref_embedding = pcr.transform_and_normalize_torch(ref_embedding, torch.from_numpy(kernel).to(device), torch.from_numpy(bias).to(device))
            hyp_embedding = pcr.transform_and_normalize_torch(hyp_embedding, torch.from_numpy(kernel).to(device), torch.from_numpy(bias).to(device))

        batch_size_tmp = len(ref_tokens)

        ref_parsing_batch= []
        hyp_parsing_batch= []
        ref_t2w_batch = []
        hyp_t2w_batch = []

        for i in range(batch_size_tmp):         
            # matching diff tokenization
            ref_t2w, ref_word_sqe = berttk2wordtk(ref_tokens[i])
            hyp_t2w, hyp_word_sqe = berttk2wordtk(hyp_tokens[i])
            ref_t2w_batch.append(ref_t2w)
            hyp_t2w_batch.append(hyp_t2w)
            ref_parsing_batch.append(' '.join(ref_word_sqe))
            hyp_parsing_batch.append(' '.join(hyp_word_sqe))

        # parsing
        ref_parsing_batch= [s if not (s.isspace() or len(s)==0) else 'good' for s in ref_parsing_batch]
        ref_parsing_data= nlp('\n\n'.join(ref_parsing_batch))
        ref_parsing_data = lib_dep.stanza2dic(ref_parsing_data)
        hyp_parsing_batch= [s if not (s.isspace() or len(s)==0) else 'good' for s in hyp_parsing_batch]
        hyp_parsing_data= nlp('\n\n'.join(hyp_parsing_batch))
        hyp_parsing_data = lib_dep.stanza2dic(hyp_parsing_data)

        ### form subtree
        ref_tree_batch = []
        hyp_tree_batch = []
        for i in range(batch_size_tmp):
            assert len(ref_parsing_data[i]) == len(ref_t2w_batch[i])
            assert len(hyp_parsing_data[i]) == len(hyp_t2w_batch[i])
            ref_subtree = form_subtree(ref_parsing_data[i], ref_t2w_batch[i], tree, rel_kept, hop_num)
            ref_tree_batch.append(ref_subtree)
            hyp_subtree = form_subtree(hyp_parsing_data[i], hyp_t2w_batch[i], tree, rel_kept, hop_num)
            hyp_tree_batch.append(hyp_subtree)
            # num_ngram += len(ref_subtree)
            # num_ngram += len(hyp_subtree)

        # assign weight
        for i in range(batch_size_tmp):
            for w_id, w in enumerate(ref_parsing_data[i]):
                if w['text'] in word2weight:
                    ref_w[i,ref_t2w_batch[i][w_id]] = word2weight[w['text']] #np.power(word2idf[w['text']],power)
                else:
                    ref_w[i,ref_t2w_batch[i][w_id]] = min(word2weight.values()) # np.power(min(word2idf.values()),power)
            for w_id, w in enumerate(hyp_parsing_data[i]):
                if w['text'] in word2weight: 
                    hyp_w[i,hyp_t2w_batch[i][w_id]] = word2weight[w['text']] # np.power(word2idf[w['text']],power)
                else:
                    hyp_w[i,hyp_t2w_batch[i][w_id]] = min(word2weight.values()) # np.power(min(word2idf.values()),power)
        for i in range(batch_size_tmp):  
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                if w in stop_words or '##' in w 
                                or w in set(string.punctuation)]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                if w in stop_words or '##' in w
                                or w in set(string.punctuation)]
          
            ref_embedding[i, ref_ids,:] = 0                        
            hyp_embedding[i, hyp_ids,:] = 0
            
            ref_w[i, ref_ids] = 0
            hyp_w[i, hyp_ids] = 0

        raw = torch.cat([ref_embedding, hyp_embedding], 1)                             
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
        
        if l2_dist:
            distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()
        else:
            distance_matrix = batched_cdist_cos(raw, raw).double().cpu().numpy()

        # DWD
        if not tree == 'n': 
            # distance matrix modification
            from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
            for i in range(batch_size_tmp):
                if len(ref_tree_batch[i])>=1 and len(hyp_tree_batch[i])>=1:
                    # form embedding for tree
                    ref_te = []
                    for tree_idx in ref_tree_batch[i]:
                        embed = ref_embedding[i,tree_idx,:]
                        weight = torch.as_tensor(ref_w[i, tree_idx])[:, None].to(device)
                        # weight = torch.pow(weight, alpha)
                        weight =  _safe_divide(weight, torch.sum(weight))
                        embed = embed*weight

                        embed = torch.sum(embed, dim=0)
                        embed.div_(torch.norm(embed, dim=-1).unsqueeze(-1) + 1e-30)
                        ref_te.append(embed.double().cpu().numpy())
                        # ref_te.append(torch.mean(ref_embedding[i,tree_idx,:], dim=0).double().cpu().numpy())
                    hyp_te = []
                    for tree_idx in hyp_tree_batch[i]:
                        embed = hyp_embedding[i,tree_idx,:]
                        weight = torch.as_tensor(hyp_w[i, tree_idx])[:, None].to(device)
                        # weight = torch.pow(weight, alpha)
                        weight =  _safe_divide(weight, torch.sum(weight))
                        embed = embed*weight

                        embed = torch.sum(embed, dim=0)
                        embed.div_(torch.norm(embed, dim=-1).unsqueeze(-1) + 1e-30)
                        hyp_te.append(embed.double().cpu().numpy())
                        # hyp_te.append(torch.mean(hyp_embedding[i,tree_idx,:], dim=0).double().cpu().numpy())
                    # distance
                    if l2_dist:
                        dist_tree = euclidean_distances(ref_te, hyp_te)
                    else:
                        dist_tree = cosine_distances(ref_te, hyp_te)

                    dist_w = np.zeros((len(ref_embedding[i]), len(hyp_embedding[i])))
                    dist_count = {tuple([id1, id2]):[] for id1 in range(len(ref_embedding[i])) for id2 in range(len(hyp_embedding[i]))}
                    for tree_id1, id1 in enumerate(ref_tree_batch[i]):
                        for tree_id2, id2 in enumerate(hyp_tree_batch[i]):
                            tree_distance = dist_tree[tree_id1, tree_id2]
                            if tree == 'c':
                                dist_count[tuple([id1[-1],id2[-1]])].append(tree_distance)
                            else:
                                for i1 in id1:
                                    for i2 in id2:
                                        dist_count[tuple([i1,i2])].append(tree_distance)
                    for key, value in dist_count.items():
                        idx = list(key)
                        if value:
                            dist_w[idx[0], idx[1]] = np.average(value)
                        else:
                            dist_w[idx[0], idx[1]] = 0
                    # dist_w[dist_w==0] = np.max(dist_w)
                    distance_matrix[i,len(ref_embedding[i]):,:len(ref_embedding[i])] \
                        += a*dist_w.T
                    distance_matrix[i,:len(ref_embedding[i]), len(ref_embedding[i]):] \
                        +=  a*dist_w


        for i in range(batch_size_tmp):  
            c1 = np.zeros(raw.shape[1], dtype=np.float64)
            c2 = np.zeros(raw.shape[1], dtype=np.float64)
            c1[:len(ref_w[i])] = ref_w[i]
            c2[len(ref_w[i]):] = hyp_w[i]
            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            
            dst = distance_matrix[i]
            if dist_flag:
                score, _ = emd_with_flow(c1, c2, dst)
            else:
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1./(1. + np.sum(flow * dst))#1 - np.sum(flow * dst)
            preds.append(score)
    # print('num of ngram:', num_ngram)
    return preds

def subword2word(seq_word):
    seq = []
    for w in seq_word:
        if len(w) == 1: seq.append(w[0])
        else:
            w = [w[0]] + [x[2:] for x in w[1:]]
            seq.append(''.join(w))
    return seq

def berttk2wordtk(berttk):
    seq_id = []
    seq_word = []
    for i, tk in enumerate(berttk):
        if tk in ['[CLS]', '[SEP]']: continue
        if '##' in tk:
            seq_id[-1].append(i)
            seq_word[-1].append(tk)
        else:
            seq_id.append([i])
            seq_word.append([tk])
    seq_word = subword2word(seq_word)
    return seq_id, seq_word

def form_subtree(parsed_sent, t2w, tree='t', rel_kept=[], hop_num=1):
    G = lib_dep.sentdic2dicgraph(parsed_sent)
    if tree == 's':
        subtree_w = lib_dep.node_subtree(G, hop_num)
    else:
        subtree_w = []
    
    # convert to token id
    subtree_t = []
    for sbt in subtree_w:
        if len(rel_kept)>0:
            if not parsed_sent[sbt[-1]]['deprel'] in rel_kept: continue
        subtree_t.append([idy for idx in sbt for idy in t2w[idx]])
        # subtree_t.append([t2w[idx][0] for idx in sbt])
    return subtree_t