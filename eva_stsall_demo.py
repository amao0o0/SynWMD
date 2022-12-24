#%% Essential packages
import numpy as np
from nltk.corpus import stopwords
import string 
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval, STSBun_Eval
from collections import Counter, defaultdict
from scipy.stats import spearmanr, pearsonr
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['SynWMD_MODEL'] = 'bert-base-uncased' #'princeton-nlp/unsup-simcse-bert-base-uncased'  #"sentence-transformers/bert-base-nli-mean-tokens"
#'bert-base-uncased' #
from Lib.SynWMD import SynWMD
from Lib import whiten as pcr
from Lib import lib_dependency as lib_dep
import networkx as nx
import stanza
import warnings
warnings.filterwarnings("ignore")

stop_words = stopwords.words('english')
punct = [i for i in string.punctuation ]
stop_words = stop_words + punct
# Parser
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',
                    tokenize_pretokenized=True,
                    tokenize_no_ssplit=True,
                    verbose = True, use_gpu= True)
#%% functions
def build_graph(parsing_data, word2id, hop_size=3):
    num_pair = 0
    G = nx.Graph()
    edge_count = defaultdict(float)
    total_count = {id:0 for id in range(len(word2id))}
    for sent in parsing_data:
        tree = lib_dep.sentdic2undicgraph(sent)
        for word_idx, word in enumerate(sent):
            d_word_idx = word2id[word['text']]
            neighb_dict = nx.single_source_shortest_path_length(tree, word_idx, cutoff=hop_size)
            # adding co-occurrence time
            for neighb_idx, hop in neighb_dict.items():
                if hop == 0: continue # avoid the word itself
                h_word_idx = word2id[sent[neighb_idx]['text']]
                edge_count[(d_word_idx, h_word_idx)] += 1 / hop   
                total_count[d_word_idx] += 1 / hop
                num_pair += 1
    # normalize
    # edge_count = {x:c/total_count[x[0]] for x, c in edge_count.items()}
    weight_edge_list = [x+tuple([c]) for x, c in edge_count.items()]
    G.add_weighted_edges_from(weight_edge_list)
    # print('num_pair:', num_pair)
    return G, num_pair

def data_all_set(data_name, data_loader):
    refs = []
    cands = []
    gs_scores = []
    vocab_count = Counter()
    if data_name =='STSB':
        evaluation = data_loader('./data/downstream/STS/STSBenchmark')
        data_test = evaluation.data['sts-test']
        rf, cd, gs = data_test
        for sent in rf+cd:
            vocab_count.update(sent)
        rf = [' '.join(x) for x in rf]
        cd = [' '.join(x) for x in cd]
        refs.extend(rf)
        cands.extend(cd)
        gs_scores.extend(gs)
    else:
        tpath = './data'
        fpath = data_name + '-en-test'
        evaluation = data_loader(tpath + '/downstream/STS/'+fpath)
        for dataset in evaluation.datasets:
            rf, cd, gs = evaluation.data[dataset]
            for sent in rf+cd:
                vocab_count.update(sent)
            rf = [' '.join(x) for x in rf]
            cd = [' '.join(x) for x in cd]
            refs.extend(rf)
            cands.extend(cd)
            gs_scores.extend(gs)  
    return refs, cands, gs_scores, vocab_count
# %% all-setting STS evaluation

# pre-trained kernel and bias for whitening
path_whiten = './data/whiten/bert-base-stsall-rmsw-first_last.pkl'
kernel, bias = pcr.load_whiten(path_whiten)

# parameters. The following is the setting for SynWMD_dwf+dwd using BERT (first_last)
param = {'batch_size': 64, 
        'l2_dist': False, # distance metric: False -> cosine distance, True -> l2 distance
        'tree': 's', # 'n' -> without DWD, that is SynWMD_dwf, 's'  -> use DWD, that is SynWMD_dwf+dwd
        'a': 1, # float, parameter a in DWD, controlling how much contextual and structual infor DWD considers
        'hop_num': 3, # int, subtree size in DWD
        'layer': 'first_last', # embedding layer: 'first_last', 'last', 'last2' or int
        'whiten_flag': True, # whitening pre-processing
        'pre_whiten': True, # use pre-trained kernel and bias, only works when 'whiten_flag': True
        'pre_kernel': kernel, # pre-trained kernel for whitening, otherwise set to []
        'pre_bias': bias } # pre-trained bias for whitening, otherwise set to []

# evaluation
task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSB']
task_func_list = [STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBun_Eval]
result_all = []
for task, task_func in zip(task_list, task_func_list):

    refs, cands, gs_scores, vocab_count = data_all_set(task, task_func)

    vocab = [w[0] for w in vocab_count.most_common()]
    word2id = {w:id for id,w in enumerate(vocab)}

    ## IDF weighting
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # tf = TfidfVectorizer(use_idf=True)
    # tf.fit_transform(refs+cands)
    # word_weight = dict(zip(tf.get_feature_names(), tf.idf_))

    ## None weigting
    # word_weight = {k:1 for k,v in word_weight.items()}

    ## DWF weighting
    parsing_batch = refs+cands
    parsing_batch= [s if not (s.isspace() or len(s)==0) else 'good' for s in parsing_batch]
    parsing_data= nlp('\n\n'.join(parsing_batch))
    parsing_data = lib_dep.stanza2dic(parsing_data)

    G, num_pair = build_graph(parsing_data, word2id, hop_size=3)
    word_weight = {}
    pr = nx.pagerank(G, alpha=0.2)
    for k in range(len(vocab)):
        if k in pr:
            word_weight[vocab[k]] = 1/(pr[k])
    ## 
    sys_scores = SynWMD(refs, cands, 
                word_weight, **param)

    all_pearson = pearsonr(sys_scores, gs_scores)[0]
    all_spearman = spearmanr(sys_scores, gs_scores)[0]
    print('ALL (weighted average) : Pearson = %.4f, \
        Spearman = %.4f' % (all_pearson, all_spearman))
    result_all.append(all_spearman)

print('\nFinal (weighted average):Spearman = %.4f\n' % np.average(result_all))
result_all = [str(round(x*100, 2)) for x in result_all]
print(' '.join(result_all))
# %%
