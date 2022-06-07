#%%
import os
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
import stanza

from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBun_Eval
from Lib import lib_dependency as lib_dep

os.environ["CUDA_VISIBLE_DEVICES"]="1"
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True, verbose = True)
#%% #################### STS Benchmark
# read data
tpath = './data'
evaluation = STSBun_Eval(tpath + '/downstream/STS/STSBenchmark')
data = []
for set_name in ['sts-train', 'sts-dev', 'sts-test']:
    data_set = evaluation.data[set_name]
    input1, input2, gs_scores = data_set
    data.extend(input1+input2)
# parse
doc= nlp(data)
doc = lib_dep.stanza2dic(doc)
# save
output_path = './data/parsed_data/sts_benchmark/'
Path(output_path).mkdir(parents=True, exist_ok=True)
with open(output_path+'data.pickle', mode='wb') as fp:
    pickle.dump(doc, fp)

#%% #################### STS12-16
tpath = './data'
for name, func in zip(['STS12', 'STS13', 'STS14', 'STS15', 'STS16'], [STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval]):
    fpath = name + '-en-test'
    evaluation = func(tpath + '/downstream/STS/' + fpath)
    
    data = []
    for dataset in evaluation.datasets:
        sys_scores = []
        input1, input2, gs_scores = evaluation.data[dataset]
        data.extend(input1+input2)
    print('-----')
    print(name, '# sentence: ', len(data))
    # parse
    doc= nlp(data)
    doc = lib_dep.stanza2dic(doc)

    # save
    output_path = './data/parsed_data/'+name+'/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(output_path+'data.pickle', mode='wb') as fp:
        pickle.dump(doc, fp)