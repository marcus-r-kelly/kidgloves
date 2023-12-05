if __name__ == '__main__' : 
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
        default=500,
        help='Number of cohort bootstraps to conduct')
    parser.add_argument('--burn_in_repeats',
        action='store',
        default=20,
        help='Number of repeats for coefficient exploration during burn-in.')
    parser.add_argument('--sample_factor',
        action='store',
        default=0.5,
        help='Size of the training set during cross-validated fitting.')
    parser.add_argument('--post_hoc_power',
        action='store',
        default=50,
        help='Factor by which to resample coefficients to estimate significance, as a multiple of the number of coefficients.')
    parser.add_argument('--j_stringency',
        action='store',
        default=2,
        help='Negative power of ten by which to select nonzero elements of J. i.e. if j_stringency=2, p=0.01')

    ns=parser.parse_args()


import os
import sys
import time
import pickle
def lmsg(lg,x,lrtime,starttime) : 
    now=time.time()
    print('Fitting {: <12}, round {: >8}. Last round took {} ({} total)'.format(
        lg.hierarchy_name,x,
        time.strftime("%H:%M:%S",time.gmtime(now-lrtime)),
        time.strftime("%H:%M:%S",time.gmtime(now-starttime)),
    ),end='\n')
    sys.stdout.flush()

import pandas as pd
import numpy as np
import warnings
import kidgloves as kg
opj=kg.opj
msg=kg.msg
kg.read_config()
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import lars_path,Lasso,LassoLars
from sklearn.model_selection import RepeatedKFold
from collections import namedtuple
import multiprocessing as mp
import time
from tqdm.auto import tqdm
from functools import reduce
from statsmodels.stats.multitest import multipletests
from scipy.stats import bootstrap
from scipy.stats.distributions import t
import torch
#~~~~~~~~LARGeSSE-G core functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getJ(omics,jmat,correlation_p=1e-2) : 
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        thisprecorr=np.concatenate([omics,jmat],axis=1)
        cc=np.corrcoef(thisprecorr.transpose())
        subcc=cc[:omics.shape[1],(-1*jmat.shape[1]):]
        subcc[ (subcc < 0) | np.isnan(subcc) ]=0.0

        # this addition added 11/12/2023, following
        # https://courses.lumenlearning.com/introstats1/chapter/testing-the-significance-of-the-correlation-coefficient/
        n=omics.shape[0]
        tstats=subcc*np.sqrt(n-2)/np.sqrt(1-subcc*subcc)
        mydt=t(df=n-2)
        cdfs=mydt.cdf(tstats)
        sigmask=cdfs>(1-correlation_p)
        subcc[ ~sigmask ]=0

        return subcc

def qunpickle(path) : 
    with open(path,'rb') as f : 
        return pickle.load(f)

def _fix_overlong_identifiers(df,index=True) :
    if index: 
        df.index=[ '-'.join(x.split('-')[:3]) for x in df.index ]
        assert len(df.index) == len(set(df.index))
    else : 
        df.columns=[ '-'.join(x.split('-')[:3]) for x in df.columns ]
        assert len(df.columns) == len(set(df.columns))

    return df

def lt_to_lg(hierarchyfilename,ltpicklefilename,**kwargs) : 

    hierarchy=kg.qunpickle(hierarchyfilename) 
    lt=kg.qunpickle(ltpicklefilename)
    hname=hierarchyfilename.split(os.sep)[-1].split('.')[0]
    return LARGeSSE_G(hierarchy,lt.training_data,lt.patients,hierarchy_name=hname,**kwargs)

class LARGeSSE_G(object) : 
    
    def __init__(self,
            hierarchy, # filename, dict, array, tensor
            omics, # dataframe
            signatures, # dataframe
            loglengths, # Series - by EID
            timing_coordinates, #Series - by EID
            samplefactor=0.95,
            hierarchy_max_members=2000,
            hierarchy_name=None,
            hierarchy_features=None,
            j_stringency=1e-2
            ) :
        super(LARGeSSE_G,self).__init__()

        self.hierarchy_max_members  =   hierarchy_max_members 
        self.hierarchy_name         =   hierarchy_name if hierarchy_name is not None else 'plumbus'
        self.omics                  =   _fix_overlong_identifiers(omics)
        self.signatures             =   _fix_overlong_identifiers(signatures)
        self.samplefactor           =   samplefactor
        self.j_stringency           =   j_stringency
        assert type(self.omics) == pd.DataFrame
        assert self.omics.columns.dtype == object
        assert type(self.signatures) == pd.DataFrame
        assert self.signatures.columns.dtype == object


        _old_omics_index_length=self.omics.shape[0]
        _old_signatures_index_length=self.signatures.shape[1]
        common_patients=np.intersect1d(self.omics.index,self.signatures.index)
        self.omics=self.omics.reindex(common_patients)
        self.signatures=self.signatures.reindex(common_patients)
        if len(common_patients) < _old_signatures_index_length  or len(common_patients) < _old_omics_index_length  :
            warnings.warn(f"input patients {_old_omics_index_length},{_old_signatures_index_length} -> common patients {len(common_patients)}")

        self.sigscale=MaxAbsScaler().fit_transform(self.signatures) 
        # this should tbh be unnecessary when loaded from a pickled logittransformer
        _jbase=getJ(self.omics,self.sigscale,correlation_p=self.j_stringency).astype(np.float32)

        lld=dict(zip(loglengths.index,loglengths.values))
        llmean=loglengths.mean()
        lls=pd.Series(
                        data=MaxAbsScaler().fit_transform(
                            np.array([ lld.get(e.split('_')[0],llmean) for e in self.omics.columns ]).reshape(-1,1)).ravel() ,
                        index=self.omics.columns,
                        )
            
        rcd=dict(zip(timing_coordinates.index,timing_coordinates.values))

        timing_coordinates= pd.Series(
                                    data=MaxAbsScaler().fit_transform(np.array([ rcd.get(e.split('_')[0],llmean) for e in self.omics.columns ]).reshape(-1,1)).ravel(),
                                    index=self.omics.columns,
                            )

        intercept=-1*np.ones((omics.shape[1],))

        _j=np.c_[_jbase,lls,timing_coordinates,intercept]
        
        self.y=np.log10(omics.sum(axis=0)+1)

        self.getfeaturetype=np.vectorize(self._gft)

        if type(hierarchy) in (dict,str): 
            # "I have been given a file path containing a pickled dictionary"
            # "I have been given a dictionary
            if type(hierarchy) == str : 
                _hdict=kg.qunpickle(hierarchy)
                if not self.hierarchy_name : 
                    self.hierarchy_name=hierarchy.split(os.sep)[-1].split('.')[0]

            else : 
                _hdict=hierarchy

            if len(_hdict) > 0 : 
                _nmo=kg.mask_nest_systems_from_omics(_hdict,omics)
                _h=kg.arrayify_nest_mask(_nmo,omics.columns).numpy() 
            else : 
                _h=np.zeros((omics.columns.shape[0],0))
            self.hierarchy_features=sorted(list(_hdict.keys()))
            if hierarchy_features is not None : 
                warnings.warn('Since features are provided as dictionary keys, hierarchy names passed via __init__() are ignored')


        elif type(hierarchy) in (np.ndarray,torch.Tensor)  : 
            # "I have been given a hierarchy membership array"
            # "I have been given a hierarchy membership tensor"
            if type(hierarchy) == np.ndarray : 
                _h=hierarchy
            else:
                _h=hierarchy.detach().cpu().numpy()

            try : 
                assert hierarchy_features is not None
            except AssertionError as e : 
                raise e('if hierarchies are passed as numeric/logical types then keys must also be provided')
            self.hierarchy_features=hierarchy_features
        else : 
            raise ValueError(f"Unknown hierarchy data type {type(hierarchy)}")


        _hsum=_h.sum(axis=0)
        hcols_inbounds=np.argwhere((_hsum>1) & (_hsum<self.hierarchy_max_members)).ravel()
        _h=_h[:,hcols_inbounds].astype(np.float32)
        self.hierarchy_features=np.array(self.hierarchy_features)[hcols_inbounds]

        self.genes=np.array(sorted(list({ c.split('_')[0] for c in omics.columns })))
        _i=np.array([ ( c.split('_')[0] == self.genes ) for c in omics.columns]).astype(np.float32)


        self.random_seed=None
        self.set_random_seed('0xc0ffee')
        self.oindices=np.arange(self.omics.shape[0],dtype=np.uint32)

        self.X=np.concatenate([ _i,_h,_j ],axis=1)
        self.J=self.X.view()[:,(-1*_j.shape[1]):]
        self.I=self.X.view()[:,:_i.shape[1]]
        self.H=self.X.view()[:,_i.shape[1]:(-1*_j.shape[1])]

        self.features=np.array( list(self.genes)+
                                list(self.hierarchy_features)+
                                list(self.signatures.columns)+
                                ['log_length','timing_coordinate','intercept']
                                )

    def set_random_seed(self,seed) :
        if type(seed) == str : 
            self.random_gen=np.random.RandomState(np.random.MT19937(int(seed,16)))
        else : # better be int
            self.random_gen=np.random.RandomState(np.random.MT19937(seed))

    def shuf_oindices(self) : 
        k=int(self.samplefactor*self.omics.shape[0])
        return self.random_gen.choice(self.oindices,size=k)

    def resample(self,indices=None,recalc_J=False,feature_indices=None) : 
        if indices is None:
            indices=self.shuf_oindices()

        if feature_indices is None :
            fslice=slice(None,None)
        else : 
            fslice=feature_indices

        thisomics=   self.omics.values.view()[indices]
        #thisomics=   self.omics.values[indices].astype(np.float32)

        defactosamplefactor=len(indices)/self.omics.shape[0]
        
        thisy   =   np.log10(1/defactosamplefactor*thisomics.sum(axis=0)+1)

        if recalc_J : 
            thissig =   self.signatures.values.view()[indices,:]
            #thissig =   self.signatures.values[indices,:].astype(np.float32)
            subcc=getJ(thisomics,thissig)
            return thisy,np.concatenate([self.I,self.J,subcc],axis=1)[:,fslice]
        else : 
            return thisy,self.X.view()[:,fslice]


    def _gft(self,s) : 
        if s in self.genes : 
            return 'gene'
        if s in self.hierarchy_features : 
            return 'system'
        return 'signature'

    def copy(self) : 
        return LARGeSSE_G(
                        hierarchy=self.H, # filename, dict, array, tensor
                        omics=self.omics, # dataframe
                        signatures=self.signatures, # dataframe
                        samplefactor=self.samplefactor,
                        hierarchy_max_members=self.hierarchy_max_members,
                        hierarchy_name=self.hierarchy_name,
                        hierarchy_features=self.hierarchy_features,
                        j_stringency=self.j_stringency
                        ) ;


_default_alpharange=10**np.linspace(-5,0,40)


BurnInResult=namedtuple('BurnInResult',['coefs','dataframe'])

def burn_in(lg,folds=5,n_repeats=15,log=True,alpharange=_default_alpharange) : 

    if log : msg("Doing burn-in analyses to scout for alpha and meaningful features...",end='\n')
    starttime=time.time()
    lrtime=starttime
    mses=list()
    coefs=list()
    alphas=list()

    kf=RepeatedKFold(n_splits=folds,random_state=int('0xc0ffee',16),n_repeats=n_repeats)
    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')  
        for i,(tr,te) in enumerate(kf.split(lg.X)) : 
            if log : lmsg(lg,i,lrtime,starttime)
            lrtime=time.time()
            mse,thisalphas,thiscoefs=_do_burnin_burning(lg.X,lg.y,tr,te)
            if mse is not None : 
              mses.append(mse)
              alphas.append(thisalphas)
              coefs.append(thiscoefs)

        return _do_burnin_wrapup(mses,alphas,coefs)


def _do_burnin_burning(X,y,tr,te) : 
   thistrx=X.view()[tr,:]
   thistex=X.view()[te,:]
   thistry=y.values.view()[tr]
   thistey=y.values.view()[te]
   try : 
       #thisalphas,_,thiscoefs=lars_path(thistrx,thistry,positive=True,method='lasso',alpha_min=0,copy_X=True)
       thisalphas,_,thiscoefs=lars_path(thistrx,
                                       thistry,
                                       positive=True,
                                       method='lasso',
                                       alpha_min=0,
                                       copy_X=True,
                                       Gram=np.dot(thistrx.T,thistrx),
                                       eps=10*np.finfo(np.float32).eps,
                                       max_iter=1500
                                       )
   except ValueError as e : 
       print("Encountered issue # 9603 [https://github.com/scikit-learn/scikit-learn/issues/9603] Restarting.")
       print(e)
       return (None,None,None)
       
   mse=np.square(np.matmul(thistex,thiscoefs)-thistey.reshape(-1,1)).sum(axis=0)/len(thistey)
   del thistrx,thistry,thistey,thistex
   return mse,thisalphas,thiscoefs

def _do_burnin_wrapup(mses,alphas,coefs) :
        burninca=np.concatenate(coefs,axis=1)
        burninalphas=np.concatenate(alphas,axis=0)
        burninmses=np.concatenate(mses,axis=0)
        burninsplits=np.array([ str(x) for x,c in enumerate(alphas) for _ in c ])

        burnindf=pd.DataFrame(index=np.arange(len(burninsplits))).assign(
            alpha=burninalphas,
            lalpha=np.log10(burninalphas),
            mse=burninmses, # fixed 11/17/2023, was "rmse" but was not in fact root
            split=burninsplits)

        return BurnInResult(
                burninca,
                burnindf)

import threadpoolctl

def qpickle(obj,fname) : 
    with open(fname,'wb') as f : 
        pickle.dump(obj,f)

def burninpost(burninresult) : 

    walphamean=np.exp(
                    (np.log(burninresult.dataframe.alpha)/burninresult.dataframe.mse).sum() /
                    (1/burninresult.dataframe.mse).sum()
                )

    ofinterest_i=np.argwhere((burninresult.coefs > 0 ).any(axis=1)).ravel()
    return walphamean,ofinterest_i

def burnin_dissection(burninresult) : 

    bird=burninresult.dataframe.copy()
    subcycle=list()

    # figure out where within its split each cycle occurred
    subcycle=list()
    last_split=-1
    sc=0
    for x in bird.split.values : 
        if int(x) > last_split : 
            last_split=int(x)
            sc=0
        else :
            sc += 1
        subcycle.append(sc)
    bird['subcycle']=subcycle 
    bird=bird.sort_values(['subcycle','split'])

    ucoefs=set()
    crundata=pd.DataFrame(columns=['cumulative_unique_nz_coefs','unique_nz_coefs_this_run','unique_nz_coefs_this_split','collated_cycle'])
    lastsplit=-1
    ucoefs_by_split=dict()

    #for x,r in bird.iterrows(): 
    for x,r in enumerate(bird.itertuples()) :
        ucoefs_this_run=set(np.argwhere(burninresult.coefs[:,x] != 0).ravel())
        ucoefs |= ucoefs_this_run
        ucoefs_by_split.update({ x : ucoefs_by_split.get(x,set()) | ucoefs_this_run })
        crundata.loc[x]=pd.Series({ 'cumulative_unique_nz_coefs'  : len(ucoefs),
                          'unique_nz_coefs_this_run'   : len(ucoefs_this_run),
                          'unique_nz_coefs_this_split'   : len(ucoefs_by_split[x]),
                          'collated_cycle' : x,
                        })

    ofinterest_i=np.argwhere(burninresult.coefs != 0)

    crundata=crundata.join(bird)

    birnz=burninresult.coefs[(burninresult.coefs > 0).any(axis=1),:].T
    rr,cc=np.where(birnz > 0)

    ofinterest_i=np.argwhere((burninresult.coefs != 0).any(axis=1)).ravel()

    earliest_discovery = dict() ; 
    for i in np.arange(birnz.shape[1]) : 
        mask=( cc == i ) 
        earliest_discovery.update({ ofinterest_i[i]  : rr[ mask].min() })
    
    return crundata


MainResult=namedtuple('MainResult',['coefs','mses'])

def mainrun(lg,alpha,ofinterest_i,n_runs=500,fold_threadpool_limit=10) : 

    outdata=list()
    import gc

    cca=np.zeros((n_runs,len(ofinterest_i)))
    mses=np.zeros((n_runs,))

    import time
    starttime=time.time()
    lastroundstart=starttime

    import warnings
    with threadpoolctl.threadpool_limits(limits=int(fold_threadpool_limit*len(os.sched_getaffinity(0)))) : 
        with warnings.catch_warnings() : 
            warnings.simplefilter('ignore')  
            for x in range(n_runs) : 
                rhi=lg.shuf_oindices()
                allindices=np.arange(lg.omics.shape[0])
                lhi=np.setdiff1d(allindices,rhi)

                ytrain,xtrain=lg.resample(indices=rhi,feature_indices=ofinterest_i)
                ytest,xtest=lg.resample(indices=lhi,feature_indices=ofinterest_i)

                mod=LassoLars(alpha=alpha,fit_intercept=False,positive=True)
                mod.fit(xtrain,ytrain)

                mse=np.square(mod.predict(xtest)-ytest).sum()/len(ytest)

                cca[x,:]=mod.coef_
                mses[x]=mse

    return MainResult(cca,mses)


import multiprocessing as mp
class bootThrall(mp.Process) : 
    def __init__(self,seed,core,outQ,sourceData,weights,alpha,maxIters,ct=(1-5e-2),bootstraps_per_strapping=100) : 
        super(bootThrall,self).__init__()
        self.gen=np.random.RandomState(np.random.MT19937(seed))
        self.outQ=outQ
        self.sourceData=sourceData
        self.maxIters=maxIters
        self.core=core
        self.weights=weights
        self.alpha=alpha
        self.bootstraps_per_strapping=bootstraps_per_strapping
        
    def run(self) :

        ct=0.95

        os.sched_setaffinity(os.getpid(),[self.core,])
        register=pd.DataFrame(index=self.sourceData.columns,
                               columns=['above','total'],
                               data=np.zeros((len(self.sourceData.columns),2)))
        
        livecols=np.array(self.sourceData.columns)
        bootstraps=0
        
        for x in range(self.maxIters) : 
            bootbatch=pd.DataFrame(index=list(range(self.bootstraps_per_strapping)),
                                   columns=livecols,
                                   data=np.stack([ do_boot(self.sourceData,self.weights,self.gen,cols=livecols) for x in range(self.bootstraps_per_strapping) ],axis=0))

            bootstraps += self.bootstraps_per_strapping
            aboves=(bootbatch > self.alpha).sum(axis=0)
            register['above']=register.above.add(aboves,fill_value=0)
            register.loc[livecols,'total'] = bootstraps 

            quantiles=register.loc[livecols,'above']/register.loc[livecols,'total']
            livecols=quantiles[ quantiles > ct ].index
            
            if len(livecols) < 1 : break
            
        self.outQ.put(register)
    
def do_boot(dfcca,weights,gen,cols=None) :
    if cols is None : 
        cols=slice(None,None)
        
    sampled=gen.choice(np.arange(dfcca.shape[0]),size=int(dfcca.shape[0]),replace=True)
    means=(weights[sampled,:]*dfcca.loc[sampled,cols]).sum(axis=0)/(weights[sampled,:]).sum()
    return means

def post_run_analysis(lg,ofinterest_i,mr,alpha,php=100) :

    cca=mr.coefs
    mses=mr.mses

    ofinterest_c=lg.features[ofinterest_i]

    ct=1-5e-2

    weights=(1/mses).reshape(-1,1)
    gen=np.random.RandomState(np.random.MT19937(int('0xdeadbeef',16)))
    dfcca=pd.DataFrame(data=cca,columns=ofinterest_c,index=list(range(cca.shape[0])))

    outQ=mp.Queue()
    nthralls=len(os.sched_getaffinity(0))

    power_sampling_requirement=php*lg.X.shape[1]//nthralls+1

    thralls=[ bootThrall(gen.randint(1000),core,outQ,dfcca,weights,alpha,(power_sampling_requirement//nthralls)+1) for core in os.sched_getaffinity(0) ]
    for t in thralls : t.start()
    resslices=[ outQ.get() for x in range(nthralls) ]
    register=reduce(pd.DataFrame.add,resslices)

    complete_register=register.reindex(lg.features).fillna({'above' : 0 , 'total' : 1})
    complete_register['p']=1-complete_register['above']/(complete_register['total'])
    bxdf=pd.DataFrame(data=lg.X,columns=lg.features,index=lg.omics.columns)

    complete_register['FDR']=multipletests(complete_register.p,method='fdr_bh')[1]
    complete_register['empir_mean']=(dfcca*weights).sum(axis=0)/weights.sum()
    complete_register['frac_nz']=(dfcca > 0).sum(axis=0)/dfcca.shape[0]
    complete_register=complete_register.fillna({ 'empir_mean' : 0.0 , 'frac_nz' : 0.0})
    complete_register['nlFDR']=[ -1*np.log10(max([1e-10,f])) for f in complete_register.FDR.values ]
    complete_register['is_hit']=complete_register.FDR.lt(5e-2)
    complete_register['feature_type']=lg.getfeaturetype(complete_register.index)
    complete_register['n_members']=(bxdf > 0).sum(axis=0)

    bsresult=bootstrap([cca,],np.mean,axis=0,confidence_level=(1-5e-2),n_resamples=int(1e3))

    bslowdata=list()
    bshidata=list()
    bssedata=list()
    for x,c in enumerate(ofinterest_c) : 
        bslowdata.append(bsresult.confidence_interval.low[x])
        bshidata.append(bsresult.confidence_interval.high[x])
        bssedata.append(bsresult.standard_error[x])
        
    complete_register=complete_register.assign(
        bslow=pd.Series(index=ofinterest_c,data=bslowdata),
        bshi=pd.Series(index=ofinterest_c,data=bshidata),
        bsse=pd.Series(index=ofinterest_c,data=bssedata),
    ).fillna(0)

    return complete_register


def get_earliest_discovery(bir) :
    birnz=bir.coefs[(bir.coefs > 0).any(axis=1),:].T
    rr,cc=np.where(birnz > 0)

    earliest_discovery = list() ; 
    for i in np.arange(birnz.shape[1]) : 
        mask=( cc == i ) 
        earliest_discovery.append( rr[ mask].min() )

    return np.array(earliest_discovery)

acconly=np.vectorize(lambda x : x.split('.')[0])
eidonly=np.vectorize(lambda x : x.split('_')[0])

def load_protein_datas() : 
    #TODO : customize script to accommodate different timings

    if kg._s2e is None : 
        kg._get_geneinfo()

    acconly=np.vectorize(lambda x : x.split('.')[0])
    eidonly=np.vectorize(lambda x : x.split('_')[0])

    grh=pd.read_csv('/cellar/users/mrkelly/Data/canon/ncbi_reference/gene2refseq_human.tsv',
            names=['taxid','GeneID','status','msg_acc','msg_gi','protein_acc','protein_gi','genomic','genoimc_gi','gstart','gend','strand','assembly','matpep_acc','matpep_gi','Symbol'],
            sep='\t')
    pi=pd.read_csv('/cellar/users/mrkelly/Data/canon/ncbi_reference/genpept/pept_info',sep='\t',index_col=0)
    pi=pi.assign(acc=acconly(pi['id'].values))
    grh=grh.assign(GeneID=grh.GeneID.astype(str),acc=acconly(grh.protein_acc))
    fr=grh[['GeneID','acc']].merge(pi[['acc','length']],on='acc')
    fgb=fr.groupby('GeneID').length.max()

    rt=pd.read_csv('/cellar/users/mrkelly/Data/canon/replication_timing/Gene_Replication_Timing_Normalized_Density.txt',sep='\t')
    rt['GeneID']=rt.gene_name.apply(kg._s2e.get).astype(str)
    bins=rt[ rt.columns[rt.columns.str.startswith('bin')] ]
    _,cc=np.indices(bins.shape)
    wbins=cc*bins
    waverage=np.sum(wbins,axis=1)/np.sum(bins,axis=1)
    rt['coord']=waverage

    return np.log10(fgb),rt.set_index('GeneID').coord


    
if __name__ == '__main__' : 

    import os
    import sys
    rules=ns.rules
    hpath=ns.hierarchy

    n_repeats=int(ns.n_repeats)
    bir_repeats=int(ns.burn_in_repeats)
    sample_factor=float(ns.sample_factor)
    php=int(ns.post_hoc_power)
    j_stringency=10**(-1*float(ns.j_stringency))

    hname='.'.join(hpath.split(os.path.sep)[-1].split('.')[:-1])
    outpath=ns.outpath
    if not os.path.exists(outpath) : 
        os.mkdir(outpath)

    msg('Reading in gene data',end='')
    lengths,coord=load_protein_datas()
    msg('Done.')

    #~~~~~~~~Read in hierarchy and omics~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    msg(f"Reading in hierarchy from {hpath}...",end='')
    msg(f"Reading in omics from LUAD COHORT...",end='')
    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')
        lg=lt_to_lg(hpath,
            kg.opj(rules,'logittransformer.pickle'),
            loglengths=lengths,
            timing_coordinates=coord,
            j_stringency=j_stringency,
            
)
    lg.samplefactor=sample_factor
    lg.j_stringency=j_stringency
    msg('Done.')

    with open(opj(outpath,'lg.pickle'),'wb') as f : 
        pickle.dump(lg,f)

    bxdf=pd.DataFrame(data=lg.X,index=lg.omics.columns,columns=lg.features)



    #~~~~~~~~BURN-IN analysis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # the goal is to get an estimate of alpha  for this hierarchy
    # as well as to define the space of coefficients actually being tested


    msg("Running burn-in analysis...",end='')
    bir=burn_in(lg,n_repeats=bir_repeats)
    with open(opj(outpath,'burnin.pickle'),'wb') as f: 
        pickle.dump(bir,f)

    alpha_empir,ofinterest_i=burninpost(bir)
    alpha_theor=np.log10(lg.omics.shape[0]/(lg.omics.shape[0]-1))
    msg('Empirical alpha (regularization strength) of {:0>4.2e} '.format(alpha_empir))
    msg('Theoretical alpha (single-coefficient limit) of {:0>4.2e} '.format(alpha_theor))
    msg('Done.')

    msg("Dissecting burn-in...",end='')
    crundata=burnin_dissection(bir)
    crundata.to_csv(opj(outpath,'burnin_dissection.csv'))
    msg("Done.")

    np.savez(opj(outpath,'ofinterest_i.npz'),ofinterest_i)
    #.....................................................................

    #~~~~~~~~Main analysis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    msg("Performing main run...",end='')
    coremodel=LassoLars(alpha=alpha_empir,positive=True,fit_intercept=False)
    coremodel.fit(lg.X[:,ofinterest_i],lg.y)
    #ofinterest_i2=ofinterest_i[coremodel.coef_ > 0]

    mr=mainrun(lg,alpha_empir,ofinterest_i,n_runs=n_repeats)
    with open(opj(outpath,'mainrun.pickle'),'wb') as f : 
        pickle.dump(mr,f)
    with open(opj(outpath,'coremodel.pickle'),'wb') as f : 
        pickle.dump(coremodel,f)
    msg("Done.")
    #.....................................................................

    #~~~~~~~~Bootstrapping for nonzero value and confidence interval~~~~~~
    msg("Skipping post-run analysis [WIP]. Done.")
    #msg("Performing post-run analysis...",end='')
    #pra=post_run_analysis(lg,ofinterest_i,mr,alpha_theor,php=php)
    #pra.to_csv(opj(outpath,'post_run_analysis.csv'))
    #msg("Done.")
    #.....................................................................

