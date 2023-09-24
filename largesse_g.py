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

parser.add_argument('--sieve',
    action='store',
    default=-1,
    help='Drop out all-zero features over several subsequent runs to improve performance and estimation.'+
    'Every n iterations, coefficients that never produced a nonzero value. NOT YET IMPLEMENTED')

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

#~~~~~~~~Read in hierarchy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
msg(f"Reading in hierarchy from {hpath}...",end='')
hierarchy=qunpickle(hpath)
hn=hpath.split(os.sep)[-1].split('.')[0]
msg('Done.')

#ogenes={ g for v in hierarchy.values() for g in v } # commented out 230920

#~~~~~~~~Read signatures and patient omics~~~~~~~~~~~~~~~~~
msg(f"Reading in signatures from folder {rules}...",end='')
#lt=qunpickle(kg.opj(rules,'logittransformer.pickle'))
from sklearn.preprocessing import MaxAbsScaler
msigframe=pd.read_csv(kg.opj(rules,'mutation_signatures.csv'),index_col=0)
from sklearn.preprocessing import MaxAbsScaler
msigscale=pd.DataFrame(index=msigframe.index,columns=msigframe.columns,data=MaxAbsScaler().fit_transform(msigframe))
msg('Done.')

msg(f"Reading in omics from LUAD COHORT...",end='')
omics=kg.autoload_events(opj(kg._config['cbioportal_folder_prefix'],'luad_tcga_pan_can_atlas_2018'),heuristic=False)
#omics=kg.autoload_events(opj(kg._config['cbioportal_folder_prefix'],'luad_tcga_pan_can_atlas_2018'),heuristic=False,gene_set=ogenes) # commented out 230920-- this makes the comparison unfair!!!
omics.index=[ '-'.join(x.split('-')[:3]) for x in omics.index ]

#patients=pd.DataFrame(index=lt.patients.index,columns=lt.patients.columns,data=MaxAbsScaler().fit_transform(lt.patients))
#lesions=lt.training_data
#ld=pd.read_csv(kg.opj(rules,'logit_data.csv'),index_col=0)

nmo_h=kg.mask_nest_systems_from_omics(hierarchy,omics)
nma_h=kg.arrayify_nest_mask(nmo_h,omics.columns)
nma_h=nma_h.numpy()
nkeys=sorted(nmo_h.keys())

#~~~~~~~~Sync up indices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
common_patients=np.intersect1d(omics.index,msigscale.index)
msigscale=msigscale.reindex(common_patients)
omics=omics.reindex(common_patients)

import multiprocessing as mp
import warnings
from tqdm.auto import tqdm

msg("Finding Spearman correlations...",end='')
def mycw(x) : 
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        return msigscale.corrwith(omics[x],method='spearman')

with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
    cw=pd.concat([ x for x in p.imap(mycw,omics.columns) ],axis=1)
    
cw.columns=omics.columns
cw=cw.transpose()
msg("Done.")

ogenes={ c.split('_')[0] for c in omics.columns } # uncommented 230920
aslo=np.array(sorted(list(ogenes)))

msg("Assembling features...",end='')
bigI=np.array([ ( c.split('_')[0] == aslo ) for c in omics.columns])
bigX=np.concatenate([bigI,nma_h,cw],axis=1)
bigxcols=np.array(list(aslo) + nkeys + list(cw.columns))
msg("Done.")

origy=np.log(omics.sum(axis=0)+1)#  230920
#origy=np.log10(omics.sum(axis=0)+1) 

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

y=np.log10(omics.sum(axis=0)+1)

lastroundstart=starttime
from sklearn.linear_model import LassoLarsIC
outdata=list()
with open(opj(outpath,'all_models_pickled.pickle'),'wb') as f : 
    for x in ['orig']+['{:0>4}'.format(x) for x in range(1,int(ns.n_repeats)+1) ] : 
        if x == 'orig' : 
            thisy=y
        else: 
            thisy=np.log(omics.sample(frac=0.95,replace=False).sum(axis=0)+1) #230920
            #thisy=np.log10(omics.sample(frac=0.95,replace=False).sum(axis=0)+1) #230910
            #thisy=np.log10(omics.sample(frac=1,replace=True).sum(axis=0)+1)

        lmsg(x,lastroundstart) ;
        lastroundstart=time.time()


        with warnings.catch_warnings() :
            warnings.simplefilter('ignore')
            os.environ['PYTHONWARNINGS']='ignore'
            #mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1) #early
            #mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1,fit_intercept=False) #230910
            #mod=LassoLarsIC(criterion='aic',positive=False,noise_variance=0.1,fit_intercept=False) #230920
            mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1,fit_intercept=False) #230921
            mod.fit(bigX,thisy) ; 
            
        aic=mod.criterion_.min()
        nnz=( mod.coef_ != 0).sum()
        nnz_systems=len(np.intersect1d(nkeys,np.array(bigxcols)[ mod.coef_ != 0]))

        odd={ 'hierarchy' : hname ,
              'run_kind'  : x,
              'aic' : aic,
              'nnz' : nnz ,
              'nnz_systems' : nnz_systems,
              'n_iter' : mod.n_iter_ }
        outdata.append(odd)

        pickle.dump(mod,f)

odf=pd.DataFrame(outdata)
odf.to_csv(opj(outpath,'models_summary_data.csv'))
with open(opj(outpath,'bigX.pickle'),'wb') as f : 
    pickle.dump((bigxcols,omics.columns,bigX),f)
