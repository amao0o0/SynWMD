#%%
import string
import os
os.environ['MOVERSCORE_MODEL'] = 'bert-base-uncased' #"sentence-transformers/bert-base-nli-mean-tokens"
#'bert-base-uncased' #
import warnings
warnings.filterwarnings("ignore")
from Lib import whiten as whiten
from nltk.corpus import stopwords
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval, STSBun_Eval
from Lib.SynWMD import whitening_prepare

stop_words = stopwords.words('english')
punct = [i for i in string.punctuation ]
stop_words = stop_words + punct
# %% all sts sentences
task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B']
task_func_list = [STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBun_Eval]

refs = []
cands = []
for task, task_func in zip(task_list, task_func_list):
    if task == 'STS-B':
        tpath = './data'
        evaluation = STSBun_Eval(tpath + '/downstream/STS/STSBenchmark')
        for dataset in evaluation.datasets:
            rf, cd, gs = evaluation.data[dataset]
            rf = [' '.join(x) for x in rf]
            cd = [' '.join(x) for x in cd]
            refs.extend(rf)
            cands.extend(cd)
    else:
        tpath = './data'
        fpath = task + '-en-test'
        evaluation = task_func(tpath + '/downstream/STS/'+fpath)
        for dataset in evaluation.datasets:
            rf, cd, gs = evaluation.data[dataset]
            rf = [' '.join(x) for x in rf]
            cd = [' '.join(x) for x in cd]
            refs.extend(rf)
            cands.extend(cd)
#
kernel, bias = whitening_prepare(refs, cands, layer='first_last',
                                batch_size=64, stop_words=stop_words)
filename = 'bert-base-stsall-rmsw-first_last.pkl'
path = './data/whiten/'
if not os.path.exists(path):
  os.makedirs(path)
whiten.save_whiten(path+filename, kernel, bias)
# %%
