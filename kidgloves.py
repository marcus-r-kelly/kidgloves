import sys
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#%matplotlib inline
#import seaborn as sb
#import palettable
#sb.set(context='notebook',style='whitegrid',palette=palettable.cartocolors.qualitative.Safe_10.hex_colors)
import os
opj=os.path.join
NPROCESSORS=len(os.sched_getaffinity(0))
import scipy.sparse
from scipy.sparse import dok_matrix,issparse
import torch
#import tensorflow as tf
from scipy.stats import percentileofscore
import dill
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import re
import warnings
import yaml
import multiprocessing as mp
from scipy.sparse import csc_matrix
from functools import reduce,partial

import hier
N_PCS=3

_config=None

import pickle
def qunpickle(fn) : 
    with open(fn,'rb') as f : 
        return pickle.load(f)

def read_config(path=opj(os.getenv('HOME'),'.config','kgconfig.yaml')): 
    global _config
    with open(path,'r') as y : 
        _config=yaml.safe_load(y)

def msg(*args,**kwargs) : 
    print(*args,**kwargs) ;
    sys.stdout.flush() ;


#~~~~~~~~Omics data reader functions are now in cohort_preprocessing~~~~~~~~~~~~
@np.vectorize
def get_ensembl_xref(dbxrefs) : 
    for xref in dbxrefs.split('|') : 
        subfields=xref.split(':')
        if subfields[0] == 'Ensembl' : 
            return subfields[1]  
    else :
        return None
@np.vectorize
def trim_ensembl_version(ensembl) :
    return ensembl.split('.')[0]

@np.vectorize
def trim_gz(filename) : 
    if filename.endswith('.gz'): 
        return '.'.join(filename.split('.')[:-1])
    return filename

    
@np.vectorize(otypes=[str])
def fix_tsb(tsb) : 
    return '-'.join(tsb.split('-')[:3])

#~~~~~~~~Tree reader functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_nest_systems(nestdf) :
    """ creates an annotation: {gene,} dict from nest table """
    return { r['shared name'] : set(r.genes.split(' ')) - {'',}
            for x,r in nestdf.iterrows ()} 

def mask_nest_systems(nest_dict,logit_df) : 
    """ creates an annotation: {lesionclass,} dict from nest table """
    return { k : logit_df[ logit_df.gene.isin(v) ].lesion_class.values for k,v in 
            nest_dict.items() if logit_df.gene.isin(v).any() }

def mask_nest_systems_from_omics(nest_dict,omics_df) : 
    nmo=dict()
    omics_columns_as_genes=np.array([ s.split('_')[0] for s in omics_df.columns ])
    for k in nest_dict : 
        nmo.update({ k : set(omics_df.columns[
                                    np.argwhere(
                                        np.isin(omics_columns_as_genes,list(nest_dict[k]))
                                    ).ravel()]) })

    return nmo
        
    #getgene=lambda s : s.split('_')[0]
    #return { k : { c for c in omics_df.columns
                   #if getgene(c) in nest_dict[k] } for k in nest_dict }

def arrayify_nest_mask(nest_mask,event_order,tensorize=True) : 
    """ creates an lc x s dok matrix mapping events to systems.
        to avoid weirdness,keys of the nest mask are sorted. """ 
    events=np.array([ e for e in event_order ])
    systems=np.array([ s for s in sorted(nest_mask.keys()) ])

    nma=np.zeros((len(systems),len(events)),dtype=np.int32)
    indices_to_change=np.argwhere([ np.isin(events,list(nest_mask[s]))
                    for s in systems ])
    nma[indices_to_change[:,0],indices_to_change[:,1]]=1
    nma=nma.transpose()

    if tensorize :
        return torch.tensor(nma) 
    else : return nma
    

#~~~~~~~~Things actually about mutational epistasis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_coincident_observation_names(singlet_names) :
    """
    from a DataFrame, get combinations of events in the order they would 
    be indexed by get_coincident_observations (for keying coincidence matrices).
    if column is None, use the column names themselves, (it should be "event_name"
    for logit data frames), otherwise use values in the indicated column.
    """
    for i,ei in enumerate(singlet_names) : 
        for ej in vals[i+1:] : 
            yield (ei,ej) ;
    

def get_coincident_observations(lc_data) : 
    """
    given a tuple <t> where the first item is a list of event names
    and a sparse matrix with len(t) columns (as retuned by a logit polyfunction),
    get the number of coincident observations.
    """

    lcdf=lc_data.astype(bool).astype(np.uint8)
    nsymobs=np.dot(lcdf.transpose(),lcdf)
    ti=np.triu_indices(nsymobs.shape[0],k=1)
    #torch
    return ( dok_matrix(nsymobs)[ti] ).transpose()

def systematize_eventdf(event_df,nest_mask) :
    """
        using a pregenerated event DataFrame, determine whether each patient
        has an event in any system defined by nest_mask
    """
    
    outdf=pd.DataFrame(data=[ event_df[nm_events].any(axis=1) 
    for nmk,nm_events in nest_mask.items() ],index=list(nest_mask.keys())).transpose()
    
    return outdf

def systematize(event_sm,nest_mask_array) :
    """
        using a pregenerated event sparse matrix, determine whether each patient
        has an event in any system defined by nest_mask
    """

    #torch
    return scipy.sparse.dok_matrix(event_sm).dot(nest_mask_array).astype(bool)

#~~~~~~~~develop model primitives-- mutation loads and logit models~~~~~~~~~~~~~

def load_transformer(filename) : 
    with open(filename,'rb') as f : 
        outdata=dill.load(f)
    return outdata

_metacols=['intercept','bigc','accuracy','n_events','lesion_class','lesion_overclass','gene']

def logit_aic_scorer(estimator,x,ytru) : 
    probs=estimator.predict_proba(x=x)[:,1] # a(x) in your formulation
    lls=ytru*np.log(probs)+(1-ytru)*np.log(1-probs)
    bigll=lls.sum()
    nparams=((estimator.coef_ != 0.0).sum()+1)
    aic=2*nparams-2*bigll
    return -1*aic

import multiprocessing as mp
#from sklearn.model_selection import gridsearchcv,shufflesplit
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

class LogitHandler(mp.Process) : 
    def __init__(self,patients,training_data,taskq,resultq) : 
        super().__init__()
        self.patients=patients
        self.training_data=training_data
        self.taskq=taskq
        self.resultq=resultq

    def run(self) : 
        while True : 
            lesionclass=self.taskq.get()
            if lesionclass is None :
                break

            result=_fit_single_lesionclass(self.patients,self.training_data[lesionclass],lesionclass)
            self.resultq.put(result)
        return

def _fit_single_lesionclass(patients,y,lesionclass) : 
    gene,lesion_overclass=lesionclass.split('_')

    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')
        os.environ['pythonwarnings']='ignore'

        bigcs=10**np.arange(-2.0,4.01,0.5)
       #if sum(y) >= 3 : 
       #    search_params=dict(c=bigcs)
       #    gscv=gridsearchcv(
       #            LogisticRegression(solver='saga',penalty='l1',max_iter=100)
       #            ,search_params,
       #            n_jobs=1,
       #            cv=stratifiedshufflesplit(n_splits=3,train_size=0.8),
       #            scoring=logit_aic_scorer)
       #    gscv.fit(patients,y)
       #    bestmodel=gscv.best_estimator_
       #else: 
        bestnaic=0
        bestmodel=None
        for bigc in bigcs : 
            estimator=LogisticRegression(penalty='l1',solver='saga',c=bigc,max_iter=100)
            estimator.fit(patients,y)
            naic=logit_aic_scorer(estimator,patients,y)
            if naic > bestnaic or bestnaic == 0 : 
                bestnaic=naic
                bestmodel=estimator

    proto_outser=dict(zip(patients.columns,bestmodel.coef_[0]))
    proto_outser.update({
        'intercept' : bestmodel.intercept_[0] ,
        'bigc'      : bestmodel.c,
        'accuracy' : bestmodel.score(patients,y) , 
        'lesion_class' : lesionclass, 
        'lesion_overclass' : lesion_overclass,
        'n_events' : (y != 0).sum(),
        'gene' : gene })

    return pd.series(proto_outser)
        

    print('alpha :',1/estimator.c,'k:',nparams,'ll:',bigll)
    return -1*aic

class LogitTransformer(object) : 
    """
    generates a function to simulate events from a (patients x burdens) array 
    by applying a probabilistic (logit) model for a given number of events
    """
    def __init__(self,training_data=None) : 
        super().__init__() ;
        self.hasbeenfit=False
        self.training_data=training_data
        self.patients=None

    def __call__(self,patient_lburdens) : 
        if not self.hasbeenfit : 
            print('has not yet been fit!')
            return None
        eventprobs=np.stack([ lm.predict_proba(patient_lburdens)[:,1] for lm in self.logit_models ],axis=-1)
        #eventdice=_rng.random(eventprobs.shape)
        eventdice=np.random.default_rng().random(eventprobs.shape)
        sim_events= eventprobs > eventdice
        idxs=np.where(sim_events)
        return (np.c_[idxs[0],idxs[1],np.ones(idxs[0].shape,dtype=np.uint16)]).transpose()

    #def analytical_freq(self,patient_lburdens=self.patients) :
        #eventprobs=np.stack([ lm.predict_proba(patient_lburdens)[:,1] for lm in self.logit_models ],axis=-1)
        #return 
    def _assemble_logit_model_from_params(self,logit_data_series):
        from sklearn.linear_model import LogisticRegression
        lr=LogisticRegression(c=logit_data_series.bigc)
        lr.classes_=np.array([0,1])
        lr.coef_=np.array([[ logit_data_series[c] for c in logit_data_series.index if c not in _metacols ]])
        lr.intercept_=np.array([logit_data_series.intercept])
        return lr

    def fit(self,patients,parallel=True) : 
        self.patients=patients

        print('fitting logit models ',end='')
        sys.stdout.flush()

        if parallel and nprocessors > 2 : 
                print('in parallel mode with {} processors '.format(nprocessors),end='\n')
                sys.stdout.flush()

                taskq=mp.queue()
                resultq=mp.queue()
                processpool=[ LogitHandler(self.patients,self.training_data,taskq,resultq) for x in range(nprocessors) ]
                for p in processpool : p.start() ;
                for c in self.training_data.columns : taskq.put(c)
                for p in processpool : taskq.put(None)

                ldd=list()
                for x,c in enumerate(self.training_data.columns) : 
                    ldd.append(resultq.get())
                    print('{: >8} of {: >8} events fitted.'.format(x,self.training_data.shape[1]),end='\r')
                    sys.stdout.flush()
                print('{: >8} of {: >8} events fitted.'.format(x+1,self.training_data.shape[1]),end='\n')

                logit_data=pd.DataFrame(ldd)
                #logit_data=pd.DataFrame([ resultq.get() for c in self.training_data.columns ])
                for p in processpool : p.join()

        else :
            print('in linear mode with {} processors '.format(nprocessors),end='')
            sys.stdout.flush()
            logit_data=pd.DataFrame([ _fit_single_lesionclass(self.patients,self.training_data[c],c) for c in self.training_data.columns])

        print('done.')
        sys.stdout.flush()

        print('assembling storable logit models...',end='')
        sys.stdout.flush()
        self.logit_models=[ self._assemble_logit_model_from_params(r) for x,r in logit_data.iterrows() ]
        print('done')
        sys.stdout.flush()

        self.hasbeenfit=True
        ctable=logit_data[logit_data.columns[ ~logit_data.columns.isin(_metacols) ]]
        nnz_relationships=( ctable !=0 ).sum().sum()
        pnz_relationships=100*nnz_relationships/np.prod(ctable.shape)
        hit_features=( ctable !=0 ).any().sum()
        discarded_features=( ctable ==0 ).all().sum()
        print('# nonzero relationships {}\n% nonzero relationships {}\n# used features {}\n# discarded features {}'.format(
                nnz_relationships,pnz_relationships,hit_features,discarded_features))

        return logit_data

    def save(self,filename) : 
        with open(filename,'wb') as f  : 
            dill.dump(self,f) ;




            





#~~~~~~~~helpful annotations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
geneinfopath=os.sep.join([os.getenv('HOME'),'Data','canon','ncbi_reference','Homo_sapiens.gene_info'])
_gi=None
_s2e=None
_e2s=None
_ens2e=None
_e2ens=None

BADGENES=None

@np.vectorize
def get_ensembl_xref(dbxrefs) : 
    for xref in dbxrefs.split('|') : 
        subfields=xref.split(':')
        if subfields[0] == 'Ensembl' : 
            return subfields[1]  
    else :
        return None

def _get_geneinfo() : 
    """
    generate event pairs for logit field analysis, excluding those where 
    a single gene is associated with both events.
    """
    global _gi 
    global _s2e
    global _e2s
    global BADGENES

    _gi=pd.read_csv(geneinfopath,sep='\t')[::-1] 
    _gi['Ensembl']=get_ensembl_xref(_gi.dbXrefs)
    _gi['GeneID']=_gi.GeneID.astype(str)
    # [::-1] this means that for items iterating through, "older"/more canonical entries will be last and supersede shakier ones


    _e2s=dict()
    _s2e=dict()
    _ens2e=dict()
    _e2ens=dict()
    for r in _gi.itertuples() :
        _e2s.update({ r.GeneID : r.Symbol })
        _e2ens.update({ r.GeneID : r.Ensembl})
        _ens2e.update({ r.Ensembl : r.GeneID})
        _s2e.update({ r.Symbol : r.GeneID })

    bad_gene_categories={'other','pseudo','biological-region','unknown'}
    BADGENES=set(list(_gi.query("type_of_gene in @bad_gene_categories or (type_of_gene == 'ncRNA' and Ensembl == 'None')").GeneID.unique()))

def annotate_map_locations(raw_model_stats_df) :
    """
    add chromosomal map locations from the ncbi's homo_sapiens.gene_info file
    to a DataFrame with columns 'gene_a' and 'gene_b', and annotate whether
    those genes are found on the same chromosome arm.
    """
    
    gi=_get_geneinfo()[['Symbol','map_location']];
    df=pd.DataFrame(raw_model_stats_df)
    df['gene_a']=raw_model_stats_df.event_a.apply(lambda s : s.split('_')[0])
    df['gene_b']=raw_model_stats_df.event_b.apply(lambda s : s.split('_')[0])
    df=df.merge(gi,left_on='gene_a',right_on='Symbol',how='left')
    df=df.merge(gi,left_on='gene_b',right_on='Symbol',suffixes=['_a','_b'],how='left')
    df.loc[df.map_location_a.isnull(),'map_location_a']='unk'
    df.loc[df.map_location_b.isnull(),'map_location_b']='unk'
    import re
    df['arm_a']=[ ''.join(re.split(r'([pq])',r.map_location_a)[:2]) for x,r in df.iterrows() ]
    df['arm_b']=[ ''.join(re.split(r'([pq])',r.map_location_b)[:2]) for x,r in df.iterrows() ]
    df['share_arm']=df.arm_a.eq(df.arm_b)
        
    return df

def annotate_system_sharing(frame,nest_mask,columns=['event_a','event_b']) : 
       return  [ len(np.intersect1d(nest_mask[r.event_a],nest_mask[r.event_b]))>0 for x,r in frame.iterrows()]

#~~~~~~~~"pipeline" components functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# the "portable item" between these functions should be a tuple 
# where the first item is a list of strings describing the events in each column
# and the second is a scipy sparse matrix of events  


import scipy.sparse
import numpy.random
_rng=numpy.random.default_rng()

#save
def _BGf_generator(patient_lburdens,multiverse_size) :
    from sklearn.mixture import bayesiangaussianmixture
    bu_a=patient_lburdens.values
    bgm=bayesiangaussianmixture(n_components=10)
    bgm.fit(patient_lburdens.values)
    for x in range(multiverse_size) : 
        i=bgm.sample(patient_lburdens.shape[0])[0]
        _rng.shuffle(i,axis=0)
        yield i

#save
def _SHc_generator(patient_lburdens,multiverse_size) :
    """
    returns a generator that generates <multiverse_size> cohorts
    of patents resampled from rows of <patient_lburdens> (an array of patient
    mutation loads in log-space)
    """
    bu_a=patient_lburdens.values
    for x in range(multiverse_size) :
        i=_rng.choice(np.indices((bu_a.shape[0],))[0],replace=True,size=bu_a.shape[0])
        yield bu_a[i]

#~~~~~~~~high-level functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def reconstitute_codf(coframe,namesframe,column=None) : 
    """
    given a tuple <t> where the first item is a list of event names
    and a sparse matrix with len(t) columns (as retuned by a logit polyfunction),
    get the number of coincident observations.
    """

    na,nb   =   zip(*get_coincident_observation_names(namesframe,column=column))
    cs      =   get_coincident_observations(coframe) ;

    if issparse(cs) : 
        return pd.DataFrame().assign(event_a=na,event_b=nb,observed=cs.toarray()) ;
    else : 
        return pd.DataFrame().assign(event_a=na,event_b=nb,observed=cs) ;
