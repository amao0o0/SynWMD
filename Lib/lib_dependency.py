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

def node_bottomsubtree(G):
    descendants = []
    for node in list(G.nodes()):
        dcdt = list(nx.descendants(G, node))
        if dcdt:
            dcdt.append(node) #append the parent itself
            descendants.append(dcdt)
    return descendants

def node_smallsubtree(G, hop_num=1):
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

def node_exactsubtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        subt = neighborhood_v2(G, node, hop_num)
        if subt: # avoid only one node
            subt.append(node) # append parent itself
            if subtree:
                if subtree[-1] == subt: break # stop when no growing
            subtree.append(subt)
    return subtree

def node_centersubtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        subt = neighborhood_v2(G, node, hop_num)
        if subt: # avoid only one node
            subt.append(node) # append parent itself
            subtree.append(subt)
    return subtree

def node_topsubtree(G, source):
    subtree = []
    for hop in range(1, len(list(G.nodes()))):
        subt = neighborhood_v2(G, source, hop)
        subt.append(source) # append parent itself
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

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel
# %%
def generate_graph(parsed_sent, word2idf, edge_penalty):
    G = nx.DiGraph()
    edge_list = []
    for d_word_idx, word in enumerate(parsed_sent):
        h_word_idx = word["head"]-1
        count = 1
        if word['text'] not in word2idf:
            d_idf = min(word2idf.values())
        else:
            d_idf = word2idf[word['text']]
        if parsed_sent[h_word_idx]['text'] not in word2idf:
            h_idf = min(word2idf.values())
        else:
            h_idf = word2idf[parsed_sent[h_word_idx]['text']]
        edge_list.append((d_word_idx, h_word_idx, h_idf*count))
        edge_list.append((h_word_idx, d_word_idx, d_idf*count))
        edge_list.append((d_word_idx, d_word_idx, d_idf*count*2))
    G.add_weighted_edges_from(edge_list)
    G.remove_node(-1) # remove root
    # connect higher-hops
    edge_new_list = []
    for s in G.nodes():
        for t in G.nodes():
            if not nx.has_path(G,source=s, target=t): continue
            path = nx.shortest_path(G, source=s, target=t)
            if len(path) <= 2: continue
            count = []
            for cur in range(len(path)-1):
                count.append(G.get_edge_data(path[cur], path[cur+1])['weight'])
            count = sum(count)/((len(path)-1)**edge_penalty)
            edge_new_list.append((s, t, count))
    G.add_weighted_edges_from(edge_new_list)
    return G

def generate_graph1(parsed_sent, word2idf, edge_penalty):
    G = sentdic2undicgraph(parsed_sent)
    edge_list = []
    for d_word_idx, word in enumerate(parsed_sent):
        for h_word_idx, head in enumerate(parsed_sent):
            if word['text'] not in word2idf:
                d_idf = min(word2idf.values())
            else:
                d_idf = word2idf[word['text']]
            if head['text'] not in word2idf:
                h_idf = min(word2idf.values())
            else:
                h_idf = word2idf[head['text']]
            path = nx.shortest_path(G, source=d_word_idx, target=h_word_idx)
            edge_list.append((d_word_idx, h_word_idx, h_idf/(len(path)**edge_penalty)))
            edge_list.append((h_word_idx, d_word_idx, d_idf/(len(path)**edge_penalty)))
        G.add_weighted_edges_from(edge_list)
        # G.remove_node(-1) # remove root
    return G

def generate_tree(parsed_sent, rel_weight):
    G = nx.Graph()
    edge_list = []
    for d_word_idx, word in enumerate(parsed_sent):
        h_word_idx = word["head"]-1
        # if word['deprel'] in rel_weight:
        #     count = rel_weight[word['deprel']]
        # else:
        #     count = min(rel_weight.values())
        count = 1
        edge_list.append((d_word_idx, h_word_idx, count))
    G.add_weighted_edges_from(edge_list)
    G.remove_node(-1) # remove root
    return G

def tree2graph(parsed_sent, rel_weight, word2idf, edge_penalty):
    tree = generate_tree(parsed_sent, rel_weight)
    G = nx.DiGraph()
    edge_list = []
    for s in range(len(parsed_sent)):
        for t in range(s+1, len(parsed_sent)):
            path = nx.shortest_path(tree, source=s, target=t)
            count = []
            for cur in range(len(path)-1):
                count.append(tree.get_edge_data(path[cur], path[cur+1])['weight'])
            count = sum(count)/((len(path)-1)**edge_penalty)
            # count = 1
            if parsed_sent[s]['text']not in word2idf:
                s_idf = min(word2idf.values())
            else:
                s_idf = word2idf[parsed_sent[s]['text']]
            if parsed_sent[t]['text'] not in word2idf:
                t_idf = min(word2idf.values())
            else:
                t_idf = word2idf[parsed_sent[t]['text']]
            edge_list.append((s, t, t_idf*count))
            edge_list.append((t, s, s_idf*count))
            edge_list.append((s, s, s_idf*max(rel_weight.values())))
    if len(edge_list) == 0:
        G.add_weighted_edges_from([(0,0,1)])
    else:
        G.add_weighted_edges_from(edge_list)
    return G

def steady_state_prop(p):
    p = np.array(p)
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)

def mk_fw(parsed_sent, word2idf, edge_penalty=2, base=10, power=1):
    tree = generate_graph(parsed_sent, word2idf, edge_penalty)
    trM = np.zeros((len(parsed_sent), len(parsed_sent)))

    for d_word_idx, word in enumerate(parsed_sent):
        if word['text'] in word2idf:
            prob_self = np.power(word2idf[word['text']]/base, power)
            # prob_self = np.power(min(word2idf.values())/(1.1*word2idf[word['text']]), c)
        else:
            prob_self = np.power(min(word2idf.values())/base, power)
            # prob_self = np.power(min(word2idf.values())/(1.1*min(word2idf.values())), c)
        prob_out = 1 - prob_self

        trM[d_word_idx,d_word_idx] = prob_self

        neighbor_all = [n for n in tree.neighbors(d_word_idx)]
        neighbor_all.remove(d_word_idx)
        neighbor_weight_all = [tree[d_word_idx][n]["weight"] for n in neighbor_all]
        total = sum(neighbor_weight_all)
        neighbor_weight_all = [w/total for w in neighbor_weight_all]
        for h_word_idx, w in zip(neighbor_all, neighbor_weight_all):
            trM[d_word_idx,h_word_idx] = prob_out * w
    return steady_state_prop(trM)

def mk_G(parsed_sent, word2idf, edge_penalty=2, base=10, power=1):
    tree = generate_graph(parsed_sent, word2idf, edge_penalty)
    trM = np.zeros((len(parsed_sent), len(parsed_sent)))

    for d_word_idx, word in enumerate(parsed_sent):
        if word['text'] in word2idf:
            prob_self = np.power(word2idf[word['text']]/base, power)
            # prob_self = np.power(min(word2idf.values())/(1.1*word2idf[word['text']]), c)
        else:
            prob_self = np.power(min(word2idf.values())/base, power)
            # prob_self = np.power(min(word2idf.values())/(1.1*min(word2idf.values())), c)
        prob_out = 1 - prob_self

        trM[d_word_idx,d_word_idx] = prob_self

        neighbor_all = [n for n in tree.neighbors(d_word_idx)]
        neighbor_all.remove(d_word_idx)
        neighbor_weight_all = [tree[d_word_idx][n]["weight"] for n in neighbor_all]
        total = sum(neighbor_weight_all)
        neighbor_weight_all = [w/total for w in neighbor_weight_all]
        for h_word_idx, w in zip(neighbor_all, neighbor_weight_all):
            trM[d_word_idx,h_word_idx] = prob_out * w
    return nx.from_numpy_matrix(trM)

def twograph2weight(parsed_s1, parsed_s2, edge_penalty, word2idf, alpha, stop_words):
    vocab_count = Counter([w['text'] for w in parsed_s1+parsed_s2])
    vocab = [w[0] for w in vocab_count.most_common()]
    word2id = {w:id for id,w in enumerate(vocab)}
    #%%
    G = nx.DiGraph()
    edge_count = defaultdict()
    total_count = {id:0 for id in range(len(vocab))}
    for sent in [parsed_s1, parsed_s2]:
        for word in sent:
            # if word['text'] in stop_words or sent[word["head"]-1]['text'] in stop_words:
            #     continue
            d_word_idx = word2id[word['text']]
            h_word_idx = word2id[sent[word["head"]-1]['text']]
            count = 1
            if word['text'] not in word2idf:
                d_idf = min(word2idf.values())
            else:
                d_idf = word2idf[word['text']]
            if sent[word["head"]-1]['text'] not in word2idf:
                h_idf = min(word2idf.values())
            else:
                h_idf = word2idf[sent[word["head"]-1]['text']]
            # count = count*d_idf*h_idf
            # if (d_word_idx, h_word_idx) in edge_count:
            #     edge_count[(d_word_idx, h_word_idx)] += 0
            #     total_count[d_word_idx] += 0
            # else: 
            #     edge_count[(d_word_idx, h_word_idx)] = count
            #     total_count[d_word_idx] += count
            if (d_word_idx, h_word_idx) in edge_count:
                edge_count[(d_word_idx, h_word_idx)] += 0
                total_count[d_word_idx] += 0
            else: 
                edge_count[(d_word_idx, h_word_idx)] = count*h_idf
                total_count[d_word_idx] += count*h_idf
            if (h_word_idx, d_word_idx) in edge_count:
                edge_count[(h_word_idx, d_word_idx)] += 0
                total_count[h_word_idx] += 0
            else: 
                edge_count[(h_word_idx, d_word_idx)] = count*d_idf
                total_count[h_word_idx] += count*d_idf
    # 9edge_count = {x:c/total_count[x[0]] for x, c in edge_count.items()}
    weight_edge_list = [x+tuple([c]) for x, c in edge_count.items()]
    G.add_weighted_edges_from(weight_edge_list)
    # connect higher-hops
    edge_new_list = []
    for s in G.nodes():
        for t in G.nodes():
            if not nx.has_path(G,source=s, target=t): continue
            path = nx.shortest_path(G, source=s, target=t)
            if len(path) <= 2: continue
            count = []
            for cur in range(len(path)-1):
                count.append(G.get_edge_data(path[cur], path[cur+1])['weight'])
            count = sum(count)/((len(path)-1)**edge_penalty)
            edge_new_list.append((s, t, count))
    G.add_weighted_edges_from(edge_new_list)
    #%%
    pr = nx.pagerank(G, alpha=alpha)
    weight1 = []
    for word in parsed_s1:
        if word2id[word['text']] in pr: weight1.append(pr[word2id[word['text']]])
        else: weight1.append(0)
    weight2 = []
    for word in parsed_s2:
        if word2id[word['text']] in pr: weight2.append(pr[word2id[word['text']]])
        else: weight2.append(0)

    # for k in range(len(parsed_s1)):
    #     print(parsed_s1[k]['text'], weight1[k])
    # print('--')
    # for k in range(len(parsed_s2)):
    #     print(parsed_s2[k]['text'], weight2[k])
    return weight1, weight2