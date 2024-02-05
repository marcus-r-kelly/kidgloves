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


#~~~~~~~~Input reader functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#TODO : alter this so that it can read a folder tree's worth of MAF files

def _fix_overlong_tcga(tcga) : 
    return '-'.join(tcga.split('-')[:3])

def read_mutation_file(fn,drop_null_eids=True) : 
    """ Load mutation file (TCGA via cbioportal) """
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        muts=pd.read_csv(fn,sep='\t')
        if drop_null_eids :  
            muts.dropna(subset=['Entrez_Gene_Id'],inplace=True)
        else : 
            muts['Entrez_Gene_Id']=muts['Entrez_Gene_Id'].replace(np.nan,0)

        muts['Entrez_Gene_Id']=muts['Entrez_Gene_Id'].astype(int).astype(str)
        muts['Tumor_Sample_Barcode']=np.vectorize(_fix_overlong_tcga)(muts.Tumor_Sample_Barcode)
        return muts

def read_cna_file(fn) : 
    """ Load CNA file (TCGA via cbioportal) """
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        cnas=pd.read_csv(fn,sep='\t').groupby('Entrez_Gene_Id').mean(numeric_only=True)
        cnas=cnas.reindex(cnas.index.dropna())
        cnas.index=cnas.index.astype(int).astype(str)
        cnas.columns.name='sample'
        cnas.index.name='Entrez_Gene_Id'
        cnas.columns=np.vectorize(_fix_overlong_tcga)(cnas.columns)
        return cnas.transpose()

def read_rna_file(fn) : 
    """ Load HiSeqV2 file and normalize (TCGA via cbioportal) """
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        hiseq=pd.read_csv(fn,sep='\t').groupby('Entrez_Gene_Id').mean(numeric_only=True)
        hiseq=hiseq.reindex(hiseq.index.dropna())
        hiseq.index=hiseq.index.astype(int).astype(str)
        hiseq.index.name='Entrez_Gene_Id'
        hiseq.columns.name='sample'
        #hiseq=hiseq.drop(columns=['Hugo_Symbol'])
        from sklearn.preprocessing import StandardScaler
        hiseq=hiseq.reset_index().groupby('Entrez_Gene_Id').mean(numeric_only=True)
        hsss=pd.DataFrame(data=StandardScaler().fit_transform(hiseq.transpose()),index=hiseq.columns,columns=hiseq.index)
        hsss.index=np.vectorize(_fix_overlong_tcga)(hsss.index)
        return hsss

def read_fusion_file(fn) : 
    """ Load gene fusion calls (TCGA via cbioportal) """
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        fus=pd.read_csv(fn,sep='\t',index_col=False)
        pats=fus.Sample_Id.unique()
        patindices=dict(zip(pats,range(len(pats))))
        syms=np.union1d(fus.Site1_Hugo_Symbol.unique(),fus.Site2_Hugo_Symbol.unique())
        symindices=dict(zip(syms,range(len(syms))))

        if _s2e is None : _get_geneinfo()

        eids=np.array([ _s2e.get(s,'0') for s in syms ])

        fgrid=np.zeros(shape=(len(pats),len(syms)),dtype=np.uint32)
        for x,r in fus.iterrows() :
            pi=patindices[r.Sample_Id]
            si1=symindices[r.Site1_Hugo_Symbol]
            si2=symindices[r.Site2_Hugo_Symbol]
            fgrid[pi,si1]=1
            fgrid[pi,si2]=1

        dffus=pd.DataFrame(index=pats,columns=eids,data=fgrid).drop(columns=['0'])
        dffus.index=np.vectorize(_fix_overlong_tcga)(dffus.index)
            
        return dffus

def autoload(tcga_directory,gene_set=None,mutations_from=None): 
    if mutations_from is None : 
        muts=read_mutation_file(os.path.join(tcga_directory,'data_mutations.txt'))
    else : 
        muts=read_mutation_file(mutations_from)
    #mpiv=pivot_mutation_events(muts).rename(columns=lambda x : x+'_mut').drop(columns=['0_mut'])
    cnas=read_cna_file(os.path.join(tcga_directory,'data_log2_cna.txt'))
    #rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt'))
    #rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt'))
    rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt')) # changed back 12/7/2023
    fus=read_fusion_file(os.path.join(tcga_directory,'data_sv.txt'))
    if not gene_set : 
        omics=sync_omics(muts,cnas,rnas,fus,logic='intersection',gene_set=gene_set)
    else : 
        omics=sync_omics(muts,cnas,rnas,fus,logic='force',gene_set=gene_set)

    return omics

def autoload_events(tcga_directory,gene_set=None,heuristic=3,n2keep=2,mutations_from=None) : 
    omics=autoload(tcga_directory,gene_set=gene_set,mutations_from=mutations_from)
    mpiv=pivot_mutation_events(omics['muts']).rename(columns=lambda x : x+'_mut')

    ups=define_lesionclass(
            [
                omics['rna'],
                omics['cnas'],
            ],
            [
                lambda x : x > 1.6 , # formerly both 1.6
                lambda x : x > 1.6 ,
            ],'up',min_events_to_keep=n2keep)

    dns=define_lesionclass(
            [
                omics['rna'],
                omics['cnas'],
            ],
            [
                lambda x : x < -0.75 , # formerly both 1.6, this is after inspection of CDK2NA in lung cancer
                lambda x : x < -0.75 ,
            ],'dn',min_events_to_keep=n2keep)

    fus=define_lesionclass(
        [ omics['fus'],],
        [ lambda x : x > 0,],
        'fus',
        min_events_to_keep=1)

    td=pd.concat([mpiv,ups,dns,fus],axis=1).fillna(0).astype(int)

    # the heuristic is only invoked if there are 3 times as many features as patients
    if heuristic and td.shape[1] >= heuristic*td.shape[0]:  
        omic_incidence=td.sum().sort_values(ascending=False)
        omicfloor=omic_incidence.iloc[heuristic*td.shape[0]]
        # the heuristic sets a floor at the event count the 3xn ranked event
        # all events with that many or greater counts are taken
        print('Omic floor set at',omicfloor)
        td=td[ td.columns[td.sum().ge(omicfloor)]]
    
    return td


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
    

   ##tf
   ##return tf.convert_to_tensor([
   #        #[ int(e in nest_mask[s]) for s in systems ]
   #        #for e in events ])
   #return torch.tensor([
   #        [ np.int32(e in nest_mask[s]) for s in systems ]
   #        for e in events ])

#~~~~~~~~transitioning from input data to lesionclasses and burdens~~~~~~~~~~~~~

def sync_omics(muts,cnas,rna,fus,gene_set=None,patients=None,logic='union') :
    """ return datasets such that all refer to the same genes and patients.
        if gene_set or patients not provided, use the largest set represented
        in all 3 datasets.

        muts, cnas, rna refer to the outputs of read_mutation_file,
        read_cna_file, and read_hiseq_file respectively.

        the mutations are *not* pivoted first
    """
    
    theop={'union' : np.union1d ,'intersection' : np.intersect1d ,
           'force' : 'force' }.get(logic.lower())
    
    from functools import reduce
    if theop == 'force' : 
        if gene_set is None : 
            raise valueerror('if logic is "force", then a gene_set must be provided')
        pass ;
    elif gene_set is None: 
        gene_set=reduce(theop,[np.unique(muts['Entrez_Gene_Id']),cnas.columns,rna.columns])
        #resume
    else: 
        inargeneset=gene_set
        gene_set=reduce(theop,[gene_set,np.unique(muts['Entrez_Gene_Id']),cnas.columns,rna.columns])
        #import sys
        #print('the following items in gene_set were not found:\n',
              #np.setdiff1d(inargeneset,gene_set),file=sys.stderr)
        
    if patients is None : 
        
        patients=reduce(np.intersect1d,[np.unique(muts['Tumor_Sample_Barcode']),cnas.index,rna.index])
        print(len(patients),'patients')
        
    mutsout=muts.query('Tumor_Sample_Barcode in @patients and Entrez_Gene_Id in @gene_set')
    #attention to these lines--- can cause dropping of some dozens of patients
    with warnings.catch_warnings() :
        warnings.simplefilter("error")

        patients=np.intersect1d(patients,mutsout['Tumor_Sample_Barcode'].unique())
        cnasout=cnas.reindex(index=patients,columns=gene_set).fillna(0)
        rnaout=rna.reindex(index=patients,columns=gene_set).fillna(0)
        fusout=fus.reindex(index=patients,columns=gene_set).fillna(0)
    
    return dict(zip(['muts','cnas','rna','fus'],[mutsout,cnasout,rnaout,fusout]))

def _default_mutation_filter(r) : 
    """
    test whether series from mutation file refers to amino acid change
    """
    return ( not pd.isnull(r.HGVSp) ) and ( not '=' in r.HGVSp )

def pivot_mutation_events(muts,f=_default_mutation_filter,min_events_to_keep=1) :
    """
    return a binarized pivot table from mutation DataFrame,
    filtering first using function f. if f is None, do not filter
    """

    if not f :
        _muts=pd.DataFrame(muts)
    else: 
        _muts=pd.DataFrame(muts[ muts.apply(f,axis=1) ])
        
    _muts['theval']=1
        
    _muts=_muts.drop_duplicates(subset=['Entrez_Gene_Id','Tumor_Sample_Barcode'],keep='first')\
            .pivot(index='Tumor_Sample_Barcode',columns='Entrez_Gene_Id',values='theval').fillna(0)

    _muts.index.name='sample'
    _muts.columns.name='gene_id'
    _muts.columns=_muts.columns.astype(str)

    totals=_muts.sum()
    _muts=_muts[ totals[ totals >= min_events_to_keep ].index ]

    return _muts.reindex(index=muts['Tumor_Sample_Barcode'].unique()).fillna(0)
    
def define_lesionclass(frames,filters,suffix,logic='and',min_events_to_keep=1) : 
    """
    define lesionclasses by applying filter functions to *pivoted* DataFrames
    and returning a boolean DataFrame defined by the given logical relationship
    between all filtered frames.
    """ 
    
    assert len(frames)==len(filters)
    assert logic.upper() in {'AND','OR'}
    
    ma=filters[0](frames[0])
    for i in range(1,len(filters)) : 
        if logic.upper() == 'AND' : 
            ma=ma & filters[i](frames[i])
        elif logic.upper() == 'OR' : 
            ma=ma | filters[i](frames[i])
                   
    ma=ma.rename(columns= lambda s : str(s)+'_'+suffix)
    if min_events_to_keep :
        ma=ma.loc[:,ma.sum(axis=0).gt(min_events_to_keep)]
    return ma

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
def _get_geneinfo() : 
    """
    generate event pairs for logit field analysis, excluding those where 
    a single gene is associated with both events.
    """
    global _gi 
    global _s2e
    global _e2s

    _gi=pd.read_csv(geneinfopath,sep='\t')
    _s2e=dict(zip(_gi.Symbol.values,_gi.GeneID.astype(int).astype(str).values))
    _e2s=dict(zip(_gi.GeneID.astype(int).astype(str).values,_gi.Symbol.values))#BOLO: can this be dropped?

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
