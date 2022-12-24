#%%
import numpy as np
from numpy import linalg as LA
from scipy.stats import pearsonr
import nltk
import io
import random
import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
#%% Parsing
def stanza2dic(stanza_result):
    dic_doc = []
    for sent in stanza_result.sentences:
        dic_sent = []
        for word in sent.words:       
            # tmp = {'id': (word.id), 'text': word.text, 'upos': word.upos, 'xpos': word.xpos, 'feats': word.feats, 'misc': word.misc,\
            #     'lemma': word.lemma, 'head': word.head, 'deps': word.deps, 'deprel': word.deprel}
            tmp = {'id': (word.id), 'text': word.text, 'upos': word.upos, 'xpos': word.xpos,\
                  'head': word.head, 'deps': word.deps, 'deprel': word.deprel}           
            dic_sent.append(tmp)
        dic_doc.append(dic_sent)
    return dic_doc

def dic2undic(input_Dic, delete_list=[]):
    # convert to undirectional
    undic = []
    for sentDic in input_Dic:
        sentUndirDic = []
        for wordDic in sentDic:       
            wordTmp = {'id': wordDic['id'], 'text': wordDic['text'], 'lemma': wordDic['lemma'], \
                'h_id': [], 'h_word': [], 'h_deprel': [], \
                'd_id': [], 'd_word': [], 'd_deprel': []}
            sentUndirDic.append(wordTmp)
        for idx, word in enumerate(sentDic):
            if word['head'] == None:
                continue
            # deprel by head
            if not word['deprel'] in delete_list:
                sentUndirDic[idx]['h_id'].append(word['head'])
                sentUndirDic[idx]['h_word'].append(sentUndirDic[word['head']-1]['text'])
                sentUndirDic[idx]['h_deprel'].append(word['deprel'])
            # deprel by dependent
                if not word['head'] == 0:
                    sentUndirDic[word['head']-1]['d_id'].append(idx+1)
                    sentUndirDic[word['head']-1]['d_word'].append(word['text'])
                    sentUndirDic[word['head']-1]['d_deprel'].append(word['deprel'])
        undic.append(sentUndirDic)     
    return undic

def sentdic2undicgraph(sentDic, delete_list=[]):
    # convert to undirectional graph
    G = nx.Graph()
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((d_word_idx, h_word_idx))
    G.add_edges_from(edge_list)
    G.remove_node(-1) # remove root
    return G

def sentdic2undicgraph_wAttri(sentDic, delete_list=[]):
    # convert to undirectional graph
    G = nx.Graph()
    for d_word_idx, word in enumerate(sentDic):
        G.add_node(d_word_idx, text=word['text'].lower())
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((d_word_idx, h_word_idx, {'deprel': word['deprel']}))
    G.add_edges_from(edge_list)
    G.remove_node(-1) # remove root
    return G

def draw_graph(G):
    pos = nx.spring_layout(G)
    # fig = plt.figure(1, figsize=(8, 8))
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G,'text')
    nx.drawing.nx_pylab.draw_networkx_labels(G, pos, labels = node_labels)
    edge_labels = nx.get_edge_attributes(G,'deprel')
    nx.drawing.nx_pylab.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

def sentdic2dicgraph(sentDic, delete_list=[]):
    # convert to undirectional graph
    G = nx.DiGraph()
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((h_word_idx, d_word_idx))
    G.add_edges_from(edge_list)
    G.remove_node(-1) # remove root
    return G

def sentdic2dicgraph_wAttri(sentDic, delete_list=[]):
    # convert to undirectional graph
    G = nx.DiGraph()
    for d_word_idx, word in enumerate(sentDic):
        G.add_node(d_word_idx, text=word['text'].lower())
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((h_word_idx, d_word_idx, {'deprel': word['deprel']}))
    G.add_edges_from(edge_list)
    G.remove_node(-1) # remove root
    return G

def node_subtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        for hop in range(1, hop_num+1):
            subt = neighborhood_v2(G, node, hop)
            if subt: # avoid only one node
                subt.append(node) # append parent itself
                if subtree:
                    if subtree[-1] == subt: break # stop when no growing
                subtree.append(subt)
    return subtree

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
                    if length == n]

def neighborhood_v2(G, node, n):
    neighb_dict = nx.single_source_shortest_path_length(G, node, cutoff=n)

    neighbors = []
    for neighb_idx, hop in neighb_dict.items():
        if hop == 0: continue # avoid the word itself
        neighbors.append(neighb_idx)
    return neighbors