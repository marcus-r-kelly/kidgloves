import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--rules',
    action='store',
    help='Folder with cohort rules, including signature tables, logit data tables, and logit transformers.')
parser.add_argument('--hierarchy',
    action='store',
    help='pickled hierarchy, storing a dict where keys are strings identifing systems and values are sets of entrez IDs as strings')
parser.add_argument('--outpath',
    action='store',
    default='output',
    help='folder in which to dump run results. Will be created at execution if necessary.')
parser.add_argument('--n_repeats',
    action='store',
    default=30,
    help='Number of cohort bootstraps to conduct')

parser.add_argument('--genes',
    action='store',
    default=30,
    help='Number of cohort bootstraps to conduct')

ns=parser.parse_args()



import pickle
def qunpickle(fn) : 
    with open(fn,'rb') as f : 
        return pickle.load(f)

import os
import sys
def msg(*args,**kwargs) : 
    print(*args,**kwargs) ;
    sys.stdout.flush() ;



import kidgloves as kg
opj=kg.opj
import pandas as pd
import numpy as np

kg.read_config()
hier=kg.hier
s2eid=hier.s2eid
eid2s=hier.eid2s_current

rules=ns.rules
hpath=ns.hierarchy
hname='.'.join(hpath.split(os.path.sep)[-1].split('.')[:-1])
outpath=ns.outpath

#~~~~~~~~Read in logit transformer and the accompanying logit data~~~~~~~~~~~~~~
msg(f"Reading in omics models from folder {rules}...",end='')
lt=qunpickle(kg.opj(rules,'logittransformer.pickle'))
from sklearn.preprocessing import MaxAbsScaler
patients=pd.DataFrame(index=lt.patients.index,columns=lt.patients.columns,data=MaxAbsScaler().fit_transform(lt.patients))
lesions=lt.training_data
ld=pd.read_csv(kg.opj(rules,'logit_data.csv'),index_col=0)
msg('Done.')

#~~~~~~~~Read in hierarchy and adapt to lesions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
msg(f"Reading in hierarchy from {hpath}...",end='')
hierarchy=qunpickle(hpath)
hn=hpath.split(os.sep)[-1].split('.')[0]
ogenes={ g for v in hierarchy.values() for g in v}
lesions=lesions[[ c for c in lesions.columns if c.split('_')[0] in ogenes ]]
nmo_h=kg.mask_nest_systems_from_omics(hierarchy,lesions)
nma_h=kg.arrayify_nest_mask(nmo_h,lesions.columns)
nkeys=sorted(nmo_h.keys())
msg('Done.')

#~~~~~~~~Sync up indices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
common_patients=np.intersect1d(lesions.index,patients.index)
patients=patients.reindex(common_patients)
lesions=lesions.reindex(common_patients)


import multiprocessing as mp
import warnings
from tqdm.auto import tqdm

msg("Finding Spearman correlations...",end='')
def mycw(x) : 
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        return patients.corrwith(lesions[x],method='spearman')

with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
    cw=pd.concat([ x for x in p.imap(mycw,lesions.columns) ],axis=1)
    
cw.columns=lesions.columns
cw=cw.transpose()
msg("Done.")

ogenes={ c.split('_')[0] for c in lesions.columns }
aslo=np.array(sorted(list(ogenes)))

msg("Assembling features...",end='')
bigI=np.array([ ( c.split('_')[0] == aslo ) for c in lesions.columns])
bigX=np.concatenate([bigI,nma_h,cw],axis=1)
bigxcols=np.array(list(aslo) + nkeys + list(cw.columns))
msg("Done.")

origy=np.log10(lesions.sum(axis=0)+1)

import time
starttime=time.time()
def lmsg(x,lrtime) : 
    now=time.time()
    print('Fitting {: <12}, round {: >8}. Last round took {} ({} total)'.format(
        hname,x,
        time.strftime("%H:%M:%S",time.gmtime(now-lrtime)),
        time.strftime("%H:%M:%S",time.gmtime(now-starttime)),
    ),end='\n')
    sys.stdout.flush()

os.makedirs(outpath,exist_ok=True)

y=np.log10(lesions.sum(axis=0)+1)

lastroundstart=starttime
from sklearn.linear_model import LassoLarsIC
outdata=list()
with open(opj(outpath,'all_models_pickled.pickle'),'wb') as f : 
    for x in ['orig']+['{:0>4}'.format(x) for x in range(1,int(ns.n_repeats)+1) ] : 
        if x == 'orig' : 
            thisy=y
        else: 
            thisy=np.log10(lesions.sample(frac=1,replace=True).sum(axis=0)+1)

        lmsg(x,lastroundstart) ;
        lastroundstart=time.time()


        with warnings.catch_warnings() :
            warnings.simplefilter('ignore')
            os.environ['PYTHONWARNINGS']='ignore'
            mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1)
            mod.fit(bigX,thisy) ; 
            
        aic=mod.criterion_.min()
        nnz=( mod.coef_ != 0).sum()
        nnz_systems=len(np.intersect1d(nkeys,np.array(bigxcols)[ mod.coef_ != 0]))

        odd={ 'hierarchy' : hname ,
              'run_kind'  : x,
              'aic' : aic,
              'nnz' : nnz ,
              'nnz_systems' : nnz_systems }
        outdata.append(odd)

        pickle.dump(mod,f)

odf=pd.DataFrame(outdata)
odf.to_csv(opj(outpath,'models_summary_data.csv'))
