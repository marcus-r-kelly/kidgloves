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

ns=parser.parse_args()

HIERARCHY_SYSTEM_MAX=2000

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
hierarchy={ k  : hierarchy[k] for k in hierarchy.keys() if len(hierarchy[k]) < HIERARCHY_SYSTEM_MAX and len(hierarchy[k]) > 1 }
hn=hpath.split(os.sep)[-1].split('.')[0]
msg('Done.')

#ogenes={ g for v in hierarchy.values() for g in v } # commented out 230920

#~~~~~~~~Read signatures and patient omics~~~~~~~~~~~~~~~~~
msg(f"Reading in omics from LUAD COHORT...",end='')
#omics=kg.autoload_events(opj(kg._config['cbioportal_folder_prefix'],'luad_tcga_pan_can_atlas_2018'),heuristic=False)
lt=kg.qp(kg.opj(rules,'logittransformer.pickle'))
omics=lt.training_data
#omics=kg.autoload_events(opj(kg._config['cbioportal_folder_prefix'],'luad_tcga_pan_can_atlas_2018'),heuristic=False,gene_set=ogenes) # commented out 230920-- this makes the comparison unfair!!!
omics.index=[ '-'.join(x.split('-')[:3]) for x in omics.index ]

#patients=pd.DataFrame(index=lt.patients.index,columns=lt.patients.columns,data=MaxAbsScaler().fit_transform(lt.patients))
#ld=pd.read_csv(kg.opj(rules,'logit_data.csv'),index_col=0)

nmo_h=kg.mask_nest_systems_from_omics(hierarchy,omics)
nma_h=kg.arrayify_nest_mask(nmo_h,omics.columns)
nma_h=nma_h.numpy()
nkeys=sorted(nmo_h.keys())

msg("Fetching event:signature relationships...",end='')

from sklearn.preprocessing import MaxAbsScaler
import warnings

with warnings.catch_warnings() :
    warnings.simplefilter('ignore')

    msigframe=pd.read_csv(kg.opj(rules,'mutation_signatures.csv'),index_col=0)
    armdata=pd.read_csv('/cellar/users/mrkelly/Data/largesse_paper/notebooks/positive_bigJ_comparison/arms_nmf.csv',index_col=0)

    msigframe=msigframe[[ c for c in msigframe.columns if not 'arm' in c ]]
    msigframe=msigframe.join(armdata)

    from sklearn.preprocessing import MaxAbsScaler
    msigscale=pd.DataFrame(index=msigframe.index,columns=msigframe.columns,data=MaxAbsScaler().fit_transform(msigframe))

msg("Done.")

#~~~~~~~~Sync up indices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
common_patients=np.intersect1d(omics.index,msigscale.index)
msigscale=msigscale.reindex(common_patients)
omics=omics.reindex(common_patients)

import multiprocessing as mp
from tqdm.auto import tqdm

def getJ(omics,jmat) : 
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        thisprecorr=np.concatenate([omics,jmat],axis=1)
        cc=np.corrcoef(thisprecorr.transpose())
        subcc=cc[:omics.shape[1],(-1*jmat.shape[1]):]
        subcc[ (subcc < 0) | np.isnan(subcc) ]=0.0

        return subcc

ogenes={ c.split('_')[0] for c in omics.columns } # uncommented 230920
aslo=np.array(sorted(list(ogenes)))

msg("Assembling features...",end='')
bigI=np.array([ ( c.split('_')[0] == aslo ) for c in omics.columns])
bigX=np.concatenate([bigI,nma_h,getJ(omics,msigscale)],axis=1)
bigxcols=np.array(list(aslo) + nkeys + list(msigscale.columns))
msg("Done.")

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


SAMPLEFACTOR=0.95
k=int(omics.shape[0]*SAMPLEFACTOR//1.0)
gen=np.random.RandomState(np.random.MT19937(int('0xc0ffee',16)))
oindices=np.arange(omics.shape[0],dtype=int)

def resample() :
    global gen
    luckies =   gen.choice(oindices,size=k)
    thisomics=   omics.values[luckies]
    thisy   =   np.log10(1/SAMPLEFACTOR*thisomics.sum(axis=0)+1)
    thissig =   msigscale.values[luckies]
    subcc=getJ(thisomics,thissig)
    return thisy,subcc

os.makedirs(outpath,exist_ok=True)

y=np.log10(omics.sum(axis=0)+1)

lastroundstart=starttime
from sklearn.linear_model import LassoLarsIC
outdata=list()
NV=1e-1


with open(opj(outpath,'all_models_pickled.pickle'),'wb') as f : 
    for x in ['orig']+['{:0>4}'.format(x) for x in range(1,int(ns.n_repeats)+1) ] : 
        if x == 'orig' : 
            thisy=y
            thisj=getJ(omics,msigscale)
        else: 
            #thisy=np.log(1/SAMPLEFACTOR*omics.sample(frac=SAMPLEFACTOR,replace=False).sum(axis=0)+1)
            thisy,thisj=resample()

        thisx=np.concatenate([bigI,nma_h,thisj],axis=1)

        lmsg(x,lastroundstart) ;
        lastroundstart=time.time()

        with warnings.catch_warnings() :
            warnings.simplefilter('ignore')
            os.environ['PYTHONWARNINGS']='ignore'
            mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=NV,fit_intercept=False) 
            mod.fit(thisx,thisy) ; 
            
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
