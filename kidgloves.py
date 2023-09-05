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
import re
import warnings
import yaml
N_PCS=3

_config=None

def read_config(path=opj(os.getenv('HOME'),'.config','kgconfig.yaml')): 
    global _config
    with open(path,'r') as y : 
        _config=yaml.safe_load(y)


#~~~~~~~~Input reader functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_mutation_file(fn,drop_null_eids=True) : 
    """ Load mutation file (TCGA via cbioportal) """
    import warnings
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        muts=pd.read_csv(fn,sep='\t')
        if drop_null_eids :  
            muts.dropna(subset=['Entrez_Gene_Id'],inplace=True)
        else : 
            muts['Entrez_Gene_Id']=muts['Entrez_Gene_Id'].replace(np.nan,0)

        muts['Entrez_Gene_Id']=muts['Entrez_Gene_Id'].astype(int).astype(str)
        return muts

def read_cna_file(fn) : 
    """ Load CNA file (TCGA via cbioportal) """
    import warnings
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        cnas=pd.read_csv(fn,sep='\t').groupby('Entrez_Gene_Id').mean()
        cnas=cnas.reindex(cnas.index.dropna())
        cnas.index=cnas.index.astype(int).astype(str)
        cnas.columns.name='sample'
        cnas.index.name='Entrez_Gene_Id'
        return cnas.transpose()

def read_rna_file(fn) : 
    """ Load HiSeqV2 file and normalize (TCGA via cbioportal) """
    import warnings
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        hiseq=pd.read_csv(fn,sep='\t').groupby('Entrez_Gene_Id').mean()
        hiseq=hiseq.reindex(hiseq.index.dropna())
        hiseq.index=hiseq.index.astype(int).astype(str)
        hiseq.index.name='Entrez_Gene_Id'
        hiseq.columns.name='sample'
        #hiseq=hiseq.drop(columns=['Hugo_Symbol'])
        from sklearn.preprocessing import StandardScaler
        hiseq=hiseq.reset_index().groupby('Entrez_Gene_Id').mean()
        hsss=pd.DataFrame(data=StandardScaler().fit_transform(hiseq.transpose()),index=hiseq.columns,columns=hiseq.index)
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

        eids=np.array([ _s2e.get(s,'0') for s in syms ])

        fgrid=np.zeros(shape=(len(pats),len(syms)),dtype=np.uint32)
        for x,r in fus.iterrows() :
            pi=patindices[r.Sample_Id]
            si1=symindices[r.Site1_Hugo_Symbol]
            si2=symindices[r.Site2_Hugo_Symbol]
            fgrid[pi,si1]=1
            fgrid[pi,si2]=1

        dffus=pd.DataFrame(index=pats,columns=eids,data=fgrid).drop(columns=['0'])
            
        return dffus

def autoload(tcga_directory,gene_set=None): 
    muts=read_mutation_file(os.path.join(tcga_directory,'data_mutations.txt'))
    #mpiv=pivot_mutation_events(muts).rename(columns=lambda x : x+'_mut').drop(columns=['0_mut'])
    cnas=read_cna_file(os.path.join(tcga_directory,'data_log2_cna.txt'))
    #rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt'))
    rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt'))
    #rnas=read_rna_file(os.path.join(tcga_directory,'data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'))
    fus=read_fusion_file(os.path.join(tcga_directory,'data_sv.txt'))
    if not gene_set : 
        omics=sync_omics(muts,cnas,rnas,fus,logic='intersection',gene_set=gene_set)
    else : 
        omics=sync_omics(muts,cnas,rnas,fus,logic='force',gene_set=gene_set)

    return omics

def autoload_events(tcga_directory,gene_set=None,heuristic=True) : 
    omics=autoload(tcga_directory,gene_set=gene_set)
    mpiv=pivot_mutation_events(omics['muts']).rename(columns=lambda x : x+'_mut')

    ups=define_lesionclass(
            [
                omics['rna'],
                omics['cnas'],
            ],
            [
                lambda x : x > 1.6 , # formerly both 1.6
                lambda x : x > 1.6 ,
            ],'up',min_events_to_keep=2)

    dns=define_lesionclass(
            [
                omics['rna'],
                omics['cnas'],
            ],
            [
                lambda x : x < -1 , # formerly both 1.6, this is after inspection of CDK2NA in lung cancer
                lambda x : x < -1 ,
            ],'dn',min_events_to_keep=2)

    fus=define_lesionclass(
        [ omics['fus'],],
        [ lambda x : x > 0,],
        'fus',
        min_events_to_keep=1)

    td=pd.concat([mpiv,ups,dns,fus],axis=1).fillna(0).astype(int)

    if heuristic and td.shape[1] >= 3*td.shape[0]:  
        omic_incidence=td.sum().sort_values(ascending=False)
        omicfloor=omic_incidence.iloc[3*td.shape[0]]
        td=td[ td.columns[td.sum().ge(omicfloor)]]
    
    return td

def load_nest(nestfile) :
    """ Load NeST from the csv exported from cytoscape """
    df=pd.read_csv(nestfile).fillna('') ;
    return df

def extract_nest_systems(nestdf) :
    """ Creates an annotation: {gene,} dict from nest table """
    return { r['shared name'] : set(r.Genes.split(' ')) - {'',}
            for x,r in nestdf.iterrows ()} 

def mask_nest_systems(nest_dict,logit_df) : 
    """ Creates an annotation: {lesionclass,} dict from nest table """
    return { k : logit_df[ logit_df.gene.isin(v) ].lesion_class.values for k,v in 
            nest_dict.items() if logit_df.gene.isin(v).any() }

def mask_nest_systems_from_omics(nest_dict,omics_df) : 
    getgene=lambda s : s.split('_')[0]
    return { k : { c for c in omics_df.columns
                   if getgene(c) in nest_dict[k] } for k in nest_dict }

def arrayify_nest_mask(nest_mask,event_order) : 
    """ Creates an lc x s dok matrix mapping events to systems.
        To avoid weirdness,keys of the nest mask are sorted. """ 
    events=[ e for e in event_order ]
    systems=[ s for s in sorted(nest_mask.keys()) ]

    #tf
    #return tf.convert_to_tensor([
            #[ int(e in nest_mask[s]) for s in systems ]
            #for e in events ])
    return torch.tensor([
            [ np.int32(e in nest_mask[s]) for s in systems ]
            for e in events ])

#~~~~~~~~Transitioning from input data to lesionclasses and burdens~~~~~~~~~~~~~

def sync_omics(muts,cnas,rna,fus,gene_set=None,patients=None,logic='union') :
    """ Return datasets such that all refer to the same genes and patients.
        If gene_set or patients not provided, use the largest set represented
        in all 3 datasets.

        muts, cnas, rna refer to the outputs of read_mutation_file,
        read_cna_file, and read_hiseq_file respectively.

        The mutations are *not* pivoted first
    """
    
    theop={'union' : np.union1d ,'intersection' : np.intersect1d ,
           'force' : 'force' }.get(logic.lower())
    
    from functools import reduce
    if theop == 'force' : 
        if gene_set is None : 
            raise ValueError('if logic is "force", then a gene_set must be provided')
        pass ;
    elif gene_set is None: 
        gene_set=reduce(theop,[np.unique(muts['Entrez_Gene_Id']),cnas.columns,rna.columns])
        #RESUME
    else: 
        inargeneset=gene_set
        gene_set=reduce(theop,[gene_set,np.unique(muts['Entrez_Gene_Id']),cnas.columns,rna.columns])
        #import sys
        #print('The following items in gene_set were NOT found:\n',
              #np.setdiff1d(inargeneset,gene_set),file=sys.stderr)
        
    if patients is None : 
        
        patients=reduce(np.intersect1d,[np.unique(muts['Tumor_Sample_Barcode']),cnas.index,rna.index])
        print(len(patients),'patients')
        
    mutsout=muts.query('Tumor_Sample_Barcode in @patients and Entrez_Gene_Id in @gene_set')
    #ATTENTION TO THESE LINES--- CAN CAUSE DROPPING OF SOME DOZENS OF PATIENTS
    import warnings
    with warnings.catch_warnings() :
        warnings.simplefilter("error")

        patients=np.intersect1d(patients,mutsout['Tumor_Sample_Barcode'].unique())
        cnasout=cnas.reindex(index=patients,columns=gene_set).fillna(0)
        rnaout=rna.reindex(index=patients,columns=gene_set).fillna(0)
        fusout=fus.reindex(index=patients,columns=gene_set).fillna(0)
    
    return dict(zip(['muts','cnas','rna','fus'],[mutsout,cnasout,rnaout,fusout]))

def _default_mutation_filter(r) : 
    """
    Test whether series from mutation file refers to amino acid change
    """
    return ( not pd.isnull(r.HGVSp) ) and ( not '=' in r.HGVSp )

def pivot_mutation_events(muts,f=_default_mutation_filter,min_events_to_keep=1) :
    """
    Return a binarized pivot table from mutation dataframe,
    filtering first using function f. If f is None, do not filter
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
    Define lesionclasses by applying filter functions to *pivoted* dataframes
    and returning a boolean dataframe defined by the given logical relationship
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
    from a dataframe, get combinations of events in the order they would 
    be indexed by get_coincident_observations (for keying coincidence matrices).
    If column is None, use the column names themselves, (it should be "event_name"
    for logit data frames), otherwise use values in the indicated column.
    """
    for i,ei in enumerate(singlet_names) : 
        for ej in vals[i+1:] : 
            yield (ei,ej) ;
    

def get_coincident_observations(lc_data) : 
    """
    Given a tuple <t> where the first item is a list of event names
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
        Using a pregenerated event dataframe, determine whether each patient
        has an event in any system defined by nest_mask
    """
    
    outdf=pd.DataFrame(data=[ event_df[nm_events].any(axis=1) 
    for nmk,nm_events in nest_mask.items() ],index=list(nest_mask.keys())).transpose()
    
    return outdf

def systematize(event_sm,nest_mask_array) :
    """
        Using a pregenerated event sparse matrix, determine whether each patient
        has an event in any system defined by nest_mask
    """

    #torch
    return scipy.sparse.dok_matrix(event_sm).dot(nest_mask_array).astype(bool)

#~~~~~~~~Develop model primitives-- mutation loads and logit models~~~~~~~~~~~~~

def load_transformer(filename) : 
    with open(filename,'rb') as f : 
        outdata=dill.load(f)
    return outdata

from sklearn.pipeline import make_pipeline

class CohortTransformer(object) : 

    def __init__(self,n_pcs=N_PCS)  :
        self.pca=PCA(n_pcs)
        self.ss=StandardScaler()
        self.pl=make_pipeline(self.ss,self.pca)
        self.hasbeenfit=False

    def fit(self,signatures) :
        self.pl.fit(signatures)
        self.hasbeenfit=True
        #self.ss.fit(comboburden) ;
        #self.pca.fit(omics) ;

    def save(self,filename) : 
        with open(filename,'wb') as f  : 
            dill.dump(self,f) ;

    def transform(self,signatures) :
        if not self.hasbeenfit : 
            print('CohortTransformer has not been fit.')
            return

        return self.pl.transform(signatures)

class _CohortTransformer_arm(object) : 

    def __init__(self)  :
        self.hsgi=_get_geneinfo()[['Symbol','map_location','type_of_gene']]
        self.hsgi['arm']=self.hsgi.map_location.apply(lambda x : ''.join(re.split(r'([pq])',x)[:2]))
        self.hsgi=self.hsgi.query('arm != "-" and type_of_gene == "protein-coding"')
        self.hsgi=self.hsgi.set_index('Symbol')
        self.pca=None
        self.ss=None

    def _group_cnas_by_arm(self,cnas) : 
        cnas_T_by_arm=cnas.transpose().assign(arm=self.hsgi.arm).groupby('arm').mean()
        cnas_by_arm=cnas_T_by_arm.transpose()
        return cnas_by_arm

    def _mburden_from_muts(self,muts) : 

        return np.log10(muts.groupby(['Tumor_Sample_Barcode']).Consequence.count()+1) ;

    def fit(self,muts,cnas) :
        cnas_by_arm=self._group_cnas_by_arm(cnas) ;
        self.pca=PCA(N_PCS) ;
        self.pca.fit(cnas_by_arm) ;
        pca_transframe=pd.DataFrame(index=cnas_by_arm.index,data=self.pca.transform(cnas_by_arm),columns=['PC'+str(x) for x in range(N_PCS) ]) ;
        mburden=self._mburden_from_muts(muts) ;
        mburden.name='mburden'
        comboburden=pd.DataFrame(mburden).join(pca_transframe) ;

        self.ss=StandardScaler()
        self.ss.fit(comboburden) ;

    def save(self,filename) : 
        with open(filename,'wb') as f  : 
            dill.dump(self,f) ;

    def transform(self,muts,cnas) :
        if self.pca is None :
            print('CohortTransformer has not been fit.')
            return

        cnas_by_arm=self._group_cnas_by_arm(cnas) ;
        pca_transframe=pd.DataFrame(index=cnas_by_arm.index,data=self.pca.transform(cnas_by_arm),columns=['PC'+str(x) for x in range(N_PCS) ]) ;
        mburden=self._mburden_from_muts(muts) ;
        mburden.name='mburden'
        comboburden=pd.DataFrame(mburden).join(pca_transframe) ;
        combonormburden=pd.DataFrame(data=self.ss.transform(comboburden),index=comboburden.index,columns=comboburden.columns) ;
        return combonormburden
        
_metacols=['intercept','bigC','accuracy','n_events','lesion_class','lesion_overclass','gene']

def logit_aic_scorer(estimator,X,ytru) : 
    probs=estimator.predict_proba(X=X)[:,1] # a(x) in your formulation
    lls=ytru*np.log(probs)+(1-ytru)*np.log(1-probs)
    bigll=lls.sum()
    nparams=((estimator.coef_ != 0.0).sum()+1)
    aic=2*nparams-2*bigll
    return -1*aic

import multiprocessing as mp
#from sklearn.model_selection import GridSearchCV,ShuffleSplit
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

class LogitHandler(mp.Process) : 
    def __init__(self,patients,training_data,taskQ,resultQ) : 
        super().__init__()
        self.patients=patients
        self.training_data=training_data
        self.taskQ=taskQ
        self.resultQ=resultQ

    def run(self) : 
        while True : 
            lesionclass=self.taskQ.get()
            if lesionclass is None :
                break

            result=_fit_single_lesionclass(self.patients,self.training_data[lesionclass],lesionclass)
            self.resultQ.put(result)
        return

def _fit_single_lesionclass(patients,y,lesionclass) : 
    gene,lesion_overclass=lesionclass.split('_')

    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')
        os.environ['PYTHONWARNINGS']='ignore'

        bigCs=10**np.arange(-2.0,4.01,0.5)
       #if sum(y) >= 3 : 
       #    search_params=dict(C=bigCs)
       #    gscv=GridSearchCV(
       #            LogisticRegression(solver='saga',penalty='l1',max_iter=100)
       #            ,search_params,
       #            n_jobs=1,
       #            cv=StratifiedShuffleSplit(n_splits=3,train_size=0.8),
       #            scoring=logit_aic_scorer)
       #    gscv.fit(patients,y)
       #    bestmodel=gscv.best_estimator_
       #else: 
        bestnaic=0
        bestmodel=None
        for bigC in bigCs : 
            estimator=LogisticRegression(penalty='l1',solver='saga',C=bigC,max_iter=100)
            estimator.fit(patients,y)
            naic=logit_aic_scorer(estimator,patients,y)
            if naic > bestnaic or bestnaic == 0 : 
                bestnaic=naic
                bestmodel=estimator

    proto_outser=dict(zip(patients.columns,bestmodel.coef_[0]))
    proto_outser.update({
        'intercept' : bestmodel.intercept_[0] ,
        'bigC'      : bestmodel.C,
        'accuracy' : bestmodel.score(patients,y) , 
        'lesion_class' : lesionclass, 
        'lesion_overclass' : lesion_overclass,
        'n_events' : (y != 0).sum(),
        'gene' : gene })

    return pd.Series(proto_outser)
        

    print('alpha :',1/estimator.C,'k:',nparams,'ll:',bigll)
    return -1*aic

class LogitTransformer(object) : 
    """
    Generates a function to simulate events from a (patients x burdens) array 
    by applying a probabilistic (logit) model for a given number of events
    """
    def __init__(self,training_data=None) : 
        super().__init__() ;
        self.hasbeenfit=False
        self.training_data=training_data
        self.patients=None

    def __call__(self,patient_lburdens) : 
        if not self.hasbeenfit : 
            print('Has not yet been fit!')
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
        lr=LogisticRegression(C=logit_data_series.bigC)
        lr.classes_=np.array([0,1])
        lr.coef_=np.array([[ logit_data_series[c] for c in logit_data_series.index if c not in _metacols ]])
        lr.intercept_=np.array([logit_data_series.intercept])
        return lr

    def fit(self,patients,parallel=True) : 
        self.patients=patients

        print('Fitting logit models ',end='')
        sys.stdout.flush()

        if parallel and NPROCESSORS > 2 : 
                print('in parallel mode with {} processors '.format(NPROCESSORS),end='\n')
                sys.stdout.flush()

                taskQ=mp.Queue()
                resultQ=mp.Queue()
                processpool=[ LogitHandler(self.patients,self.training_data,taskQ,resultQ) for x in range(NPROCESSORS) ]
                for p in processpool : p.start() ;
                for c in self.training_data.columns : taskQ.put(c)
                for p in processpool : taskQ.put(None)

                ldd=list()
                for x,c in enumerate(self.training_data.columns) : 
                    ldd.append(resultQ.get())
                    print('{: >8} of {: >8} events fitted.'.format(x,self.training_data.shape[1]),end='\r')
                    sys.stdout.flush()
                print('{: >8} of {: >8} events fitted.'.format(x+1,self.training_data.shape[1]),end='\n')

                logit_data=pd.DataFrame(ldd)
                #logit_data=pd.DataFrame([ resultQ.get() for c in self.training_data.columns ])
                for p in processpool : p.join()

        else :
            print('in linear mode with {} processors '.format(NPROCESSORS),end='')
            sys.stdout.flush()
            logit_data=pd.DataFrame([ _fit_single_lesionclass(self.patients,self.training_data[c],c) for c in self.training_data.columns])

        print('Done.')
        sys.stdout.flush()

        print('Assembling storable logit models...',end='')
        sys.stdout.flush()
        self.logit_models=[ self._assemble_logit_model_from_params(r) for x,r in logit_data.iterrows() ]
        print('Done')
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

   #def _fit_single_lesionclass(self,lesionclass) : 

   #    gene,lesion_overclass=lesionclass.split('_')

   #    bigCs=10**np.arange(-3.0,3.01,0.25)
   #    bestnaic=0
   #    bestmodel=None
   #    with warnings.catch_warnings() : 
   #        warnings.simplefilter('ignore')
   #        os.environ['PYTHONWARNINGS']='ignore'

   #        for bigC in bigCs : 
   #            estimator=LogisticRegression(penalty='l1',solver='saga',C=bigC,max_iter=1000)
   #            estimator.fit(self.patients,self.training_data[lesionclass])
   #            naic=logit_aic_scorer(estimator,self.patients,self.training_data[lesionclass])
   #            if naic > bestnaic or bestnaic == 0 : 
   #                bestnaic=naic
   #                bestmodel=estimator

   #    coefs=bestmodel.coef_[0,:]

   #    proto_outser=dict(zip(self.patients.columns,coefs))
   #    proto_outser.update({
   #        'intercept' : bestmodel.intercept_[0] ,
   #        'bigC'      : bestmodel.C,
   #        'accuracy' : bestmodel.score(self.patients,self.training_data[lesionclass]) , 
   #        #   remember to change this!
   #        'lesion_class' : lesionclass, 
   #        'lesion_overclass' : lesion_overclass,
   #        'n_events' : (self.training_data[lesionclass] != 0).sum(),
   #        #TODO why is this not showing up
   #        'gene' : gene })

   #    return pd.Series(proto_outser)

#   def _fit_single_lesionclass(self,lesionclass) :
#       #l1_ratio=0.15 if self.penalty=='elasticnet' else None
#       #lr=LogisticRegression(solver='saga',penalty=self.penalty,max_iter=1000,C=self.bigC,l1_ratio=l1_ratio)
#       # above was used 5/15
#       #lr=LogisticRegression(solver='saga',penalty='l2',max_iter=1000,C=None)
#       lr=LogisticRegression(solver='saga',penalty='l1',max_iter=1000,C=0.3)
#       lr.fit(self.patients,self.training_data[lesionclass])
#       gene,lesion_overclass=lesionclass.split('_')

#       coefs=lr.coef_[0,:]

#       #proto_outser=dict(zip(['coef'+str(x) for x in range(len(coefs))],coefs))
#       proto_outser=dict(zip(self.patients.columns,coefs))
#       proto_outser.update({
#           'intercept' : lr.intercept_[0] ,
#           'accuracy' : lr.score(self.patients,self.training_data[lesionclass]) , 
#           #   remember to change this!
#           'lesion_class' : lesionclass, 
#           'lesion_overclass' : lesion_overclass,
#           'gene' : gene })

#       outser=pd.Series(proto_outser)
#       
#       return outser

    def save(self,filename) : 
        with open(filename,'wb') as f  : 
            dill.dump(self,f) ;

# integral models (formerly found here) are no longer going to apply because of
# the necessity of using patient resampling

#~~~~~~~~Helpful annotations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GENEINFOPATH=os.sep.join([os.getenv('HOME'),'Data','canon','ncbi_reference','Homo_sapiens.gene_info'])
def _get_geneinfo() : 
    """
    Generate event pairs for logit field analysis, excluding those where 
    a single gene is associated with both events.
    """
    gi=pd.read_csv(GENEINFOPATH,sep='\t')
    return gi

_gi=_get_geneinfo()
_s2e=dict(zip(_gi.Symbol.values,_gi.GeneID.astype(int).astype(str).values))
_e2s=dict(zip(_gi.GeneID.astype(int).astype(str).values,_gi.Symbol.values))

def annotate_map_locations(raw_model_stats_df) :
    """
    Add chromosomal map locations from the NCBI's Homo_sapiens.gene_info file
    to a dataframe with columns 'gene_a' and 'gene_b', and annotate whether
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

#~~~~~~~~"Pipeline" components functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# the "portable item" between these functions should be a tuple 
# where the first item is a list of strings describing the events in each column
# and the second is a scipy sparse matrix of events  


import scipy.sparse
import numpy.random
_rng=numpy.random.default_rng()

def _BGf_generator(patient_lburdens,multiverse_size) :
    from sklearn.mixture import BayesianGaussianMixture
    bu_a=patient_lburdens.values
    bgm=BayesianGaussianMixture(n_components=10)
    bgm.fit(patient_lburdens.values)
    for x in range(multiverse_size) : 
        I=bgm.sample(patient_lburdens.shape[0])[0]
        _rng.shuffle(I,axis=0)
        yield I

def _ShC_generator(patient_lburdens,multiverse_size) :
    """
    Returns a generator that generates <multiverse_size> cohorts
    of patents resampled from rows of <patient_lburdens> (an array of patient
    mutation loads in log-space)
    """
    bu_a=patient_lburdens.values
    for x in range(multiverse_size) :
        I=_rng.choice(np.indices((bu_a.shape[0],))[0],replace=True,size=bu_a.shape[0])
        yield bu_a[I]

import multiprocessing as mp

class _MultiverseTransformer(mp.Process) : 
    """
    Instantiates a process for taking items from inQ, 
    transforming them with a list of functions, and putting them in outQ.
    """
    def __init__(self,inQ,outQ,transformer_functions) :
        super().__init__()
        self.inQ=inQ
        self.outQ=outQ
        self.transformer_functions=transformer_functions

    def workfunc(self,inmatrix) : 
        outmatrix=inmatrix
        for f in self.transformer_functions : 
            outmatrix=f(outmatrix)
        return outmatrix

    def run(self) : 
        while True : 

            inmatrix = self.inQ.get() ;
            if inmatrix is None :  
                myname=mp.current_process().name
                print(f'Shutting down {myname}.')
                return  

            outmatrix=self.workfunc(inmatrix)
            self.outQ.put(outmatrix) ;


class _MultiverseStatter(mp.Process) : 
    """
    Instantiates a process for taking items from inQ, 
    comparing them with reference_array, and putting a matrix
    of quantiles with <output_shape> in outQ
    """
    def __init__(self,reference_array,output_shape,inQ,outQ) : 
        super().__init__()
        self.reference_array=reference_array
        self.output_shape=output_shape
        self.inQ=inQ 
        self.outQ=outQ

        self.quantiles=np.zeros(self.output_shape)
        self.means=np.zeros(self.output_shape)

    def workfunc(self,inmatrix) : 
        if issparse(inmatrix) : 
            inmatrix=inmatrix.toarray()

        equalses=( self.reference_array == inmatrix).astype(np.uint16) 
        greaters=( self.reference_array > inmatrix).astype(np.uint16)

        self.quantiles  = self.quantiles+equalses+greaters+greaters
        self.means      = self.means+inmatrix

    def run(self) : 
        try: 
            n=0
            while True : 
                inmatrix = self.inQ.get() ;
                if inmatrix is None : break
                self.workfunc(inmatrix)
                n += 1
                print(f'Statter has processed {n} cohorts.',end='\r')

            print(f'\n\n\n\n\nStatter wrapping up...',end='\n')
            self.means=self.means/n
            self.quantiles=self.quantiles/n/2
            self.outQ.put((self.means,self.quantiles,n))

            myname=mp.current_process().name
            print(f'Shutting down {myname}.')
        except : 
            print(self.reference_array,inmatrix)
            print(type(self.reference_array),type(inmatrix))
            print(self.reference_array.shape,inmatrix.shape)
            raise ; 

def eos_massive(reference_df,transformer_functions,multiverse_gen,outindex) : 
    """
    A parallel form of <enrichment_over_simulation>, which applies
    transformer_functions to a multiverse of patients from multiverse_gen
    and generates a dataframe with the quantile of each reference event (labeled withoutindex)
    from the multiverse generator
    """

    tra=reference_df.values
    for fn in transformer_functions:
    # transformer functions *must* produce a 1d vector
        tra=fn(tra)

    if issparse(tra) : 
        tra=tra.toarray()

    readerQ=mp.Queue(40)
    feederQ=mp.Queue(40)
    resultQ=mp.Queue(40)

    try : 
        transformerpool=[ 
            _MultiverseTransformer(
                inQ=readerQ,
                outQ=feederQ,
                transformer_functions=transformer_functions,)
            for x in range(NPROCESSORS-1) ]

        statter=_MultiverseStatter(
            inQ=feederQ,
            outQ=resultQ,
            reference_array=tra,
            output_shape=tra.shape,
            )

        for tp in transformerpool : tp.start()
        statter.start()
        for mv in multiverse_gen : readerQ.put(mv)

        print('Poisoning reader queue...                    ')
        for tp in transformerpool : readerQ.put(None)
        for tp in transformerpool : tp.join() ; 
        print('Joined transformers.')

        print('Poisoning feeder queue...                    ')
        feederQ.put(None)
        means,quantiles,n=resultQ.get()
        print(f'Fetched final output data (n={n})...               ')


        statter.join()
        print('Closed and joined statters.')

        feederQ.close() ;
        resultQ.close() ;
    finally:
        print('Exiting responsibly...')
        for tp in transformerpool : 
            if tp.is_alive() : tp.terminate()
        if statter.is_alive() : statter.terminate()
        readerQ.close()
        print('Reader queue closed.')
        feederQ.close()
        resultQ.close()
        print('Closed feeder and result queues.')

        
    
    print('[eos_massive] Finalizing output dataframe...')
    if outindex is not None : 
        outindex=list(outindex) 
    outdf=pd.DataFrame(index=outindex).assign(
        mean        =   np.array(means).squeeze(),
        quantile    =   np.array(quantiles).squeeze(),
        observed    =   np.array(tra).squeeze())

    outdf['two-sided_quantile']=outdf['quantile'].apply(lambda x : min([x,1-x]))
    from statsmodels.stats.multitest import multipletests
    outdf['q_fdr']=multipletests(outdf['two-sided_quantile'],method='fdr_bh')[1]
    outdf['l2fc']=np.log(outdf['observed']+1)/np.log(2) - np.log(outdf['mean']+1)/np.log(2)

    return outdf

def _eoss_helper(args) :
    a,val,cohort_size=args
    if issparse(a) : 
        aa=a.toarray().ravel()
    else :
        aa=a

    std=np.std(aa)
    
    mean=aa.mean()
    if std==0 : std=(mean+1)/np.sqrt(len(aa))
    
    q=percentileofscore(aa,val,kind='rank')/100
    min_q=min([q,1-q])
    
    return { 'mv_mean' : mean,
              'mv_std' : std,
              'mv_sem' : std/np.sqrt(len(aa)),
             'mv_median' : np.median(aa),
             'obs_vs_mv_quantile' : q,
            'two-sided_quantile' : min_q,
              }
        


def enrichment_over_simulation(observation_df,multiverse_dfs) : 
    """
        Finds enrichment of events in <observation_df> over empircal distribution
        from <multiverse_dfs>. Despite the name, <multiverse_dfs> should be a list
        of sparse matrices.
    """
    
    from scipy.sparse import hstack
    from tqdm.auto import tqdm

    assert observation_df.shape[0] == multiverse_dfs[0].shape[0]
    cohort_size=observation_df.shape[0]
    waspivoted=False
    
    if ( len(multiverse_dfs[0].shape) < 2 )  or (min(multiverse_dfs[0].shape)  > 1 ): 
        waspivoted=True
    # viz. if this is a pivoted count of events or system events rather than event coincidences
        mverse_piv=np.stack([ scipy.sparse.csr_matrix(e).toarray().sum(axis=0) for e in multiverse_dfs],axis=-1)
        observation_df=observation_df.sum() ;
        observation_df.name='observed'
        observation_df=observation_df.reset_index()
        def eoss_job_gen() :
            for x in range(observation_df.shape[0]) :
                # iterating over genes
                yield((mverse_piv[x],observation_df.iloc[x].observed,cohort_size))
    else : 
        mverse_piv=hstack(multiverse_dfs)
        def eoss_job_gen() :
            for x in range(observation_df.shape[0]) :
                # iterating over genes
                yield((mverse_piv.getrow(x),observation_df.iloc[x].observed,cohort_size))
            
            
    ejg=eoss_job_gen();
    
    import multiprocessing as mp
    with mp.Pool(processes=NPROCESSORS) as p : 
        quantiledf=pd.DataFrame([ q for q in tqdm(p.imap(_eoss_helper,ejg),
                                                  total=observation_df.shape[0]) ],
                                index=observation_df.index,
                                )
    
    quantiledf=quantiledf.join(observation_df)
    quantiledf=quantiledf.assign(z=(quantiledf['observed']-quantiledf['mv_mean'])/
                                 quantiledf['mv_std']
                                ).replace(
                                    { np.inf : np.nan, -1*np.inf : np.nan })
    quantiledf['zabs']=np.abs(quantiledf.z)
    from statsmodels.stats.multitest import multipletests
    quantiledf['q_fdr']=multipletests(quantiledf['two-sided_quantile'],method='fdr_bh')[1]
    
    return quantiledf

import json
from scipy.sparse import hstack
from scipy.sparse import save_npz,load_npz
import os

def save_events_ST(dirname,econtainer,nevents,chunksize=1000,ekeys=None,cleanup=True) : 
    """
    Saves simulated events in simulated patients to a folder <dirname>.
    It breaks events from econtainer (totaling nevents) into chunks of chunksize,
    saving each chunk to its own npz file
    """
    if not os.path.isdir(dirname) : 
        os.mkdir(dirname) ; 

    if cleanup and os.path.exists(dirname+os.sep+'meta.json') : 
        os.remove(dirname+os.sep+'meta.json')

    if cleanup : 
        for f in os.listdir(dirname) : 
            if f.endswith('.npz') : os.remove(dirname+os.sep+f) ;
    
    with open(dirname+os.sep+'meta.json','w') as f : 
        json.dump({ 'chunksize' : chunksize , 
                    'ekeys' : ekeys ,
                     'nevents' : nevents },f)

    cohort_this_file=0
    total_cohort_files=0
    chunk_subcontainer=list()
    fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'

    for x,e in enumerate(econtainer) : 

        chunk_subcontainer.append(e)
        cohort_this_file += 1

        if len(chunk_subcontainer) == chunksize : 
            save_npz(fname,hstack(chunk_subcontainer).tocsc())
            chunk_subcontainer=list()
            total_cohort_files += 1
            fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'

        print(f'Saved cohort {x:06}',end='\r')

    if chunk_subcontainer : 
        save_npz(fname,hstack(chunk_subcontainer))

def _parallel_event_saver(args) : 

    fname,data=args
    save_npz(fname,hstack(data).tocsc())

    return

def _parallel_event_saver_gen(econtainer,chunksize,dirname) : 

    cohort_this_file=0
    total_cohort_files=0
    chunk_subcontainer=list()
    fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'

    for x,e in enumerate(econtainer) : 

        chunk_subcontainer.append(e)
        cohort_this_file += 1

        if len(chunk_subcontainer) == chunksize : 
            yield (fname,chunk_subcontainer)
            chunk_subcontainer=list()
            total_cohort_files += 1
            fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'

        print(f'Saved cohort {x:06}',end='\r')

    if chunk_subcontainer : 
        yield (fname,chunk_subcontainer)

def save_events_MP(dirname,econtainer,nevents,chunksize=1000,ekeys=None,cleanup=True) : 
    if not os.path.isdir(dirname) : 
        os.mkdir(dirname) ; 

    if cleanup and os.path.exists(dirname+os.sep+'meta.json') : 
        os.remove(dirname+os.sep+'meta.json')

    if cleanup : 
        for f in os.listdir(dirname) : 
            if f.endswith('.npz') : os.remove(dirname+os.sep+f) ;
    
    with open(dirname+os.sep+'meta.json','w') as f : 
        json.dump({ 'chunksize' : chunksize , 
                    'ekeys' : ekeys ,
                     'nevents' : nevents },f)

    import multiprocessing as mp
    from tqdm.auto import tqdm
    gen=_parallel_event_saver_gen(econtainer,chunksize,dirname) ;
    with mp.Pool(processes=NPROCESSORS) as p :
        for x in tqdm(p.imap(_parallel_event_saver,gen)) :
            pass ; 

import multiprocessing as mp 

class _Piper(mp.Process) : 

    def __init__(self,logit_data,jobQ) : 
        super().__init__()
        self.lt=LogitTransformer(logit_data) ;
        self.jobQ=jobQ

    def run(self) : 
        while True : 
            qresult=self.jobQ.get()
            if qresult is None : 
                break ; 
            fname,patients=qresult
            pevents=hstack([ self.lt(p) for p in patients]).tocsc() ;
            save_npz(fname,pevents)
                
def cohort_gen_to_saved_events_MP(patient_gen,logit_data,chunksize,dirname,rngseed=None,cleanup=True) : 
    """
    This helper function directly pipes a patient generator <patient_gen> to 
    a LogitTransformer derived from <logit_data> that operates on multiple cores
    and directly saves data to avoid pickling steps.
    Otherwise behaves as other save functions.
    """
    if not os.path.isdir(dirname) : 
        os.mkdir(dirname) ; 

    if cleanup and os.path.exists(dirname+os.sep+'meta.json') : 
        os.remove(dirname+os.sep+'meta.json')

    if cleanup : 
        for f in os.listdir(dirname) : 
            if f.endswith('.npz') : os.remove(dirname+os.sep+f) ;
    
    with open(dirname+os.sep+'meta.json','w') as f : 
        json.dump({ 'chunksize' : chunksize , 
                    'ekeys' : list(logit_data.event_name.values) ,
                     'nevents' : logit_data.shape[0] },f)

    import multiprocessing as mp

    jobQ=mp.Queue() ;
    servitors=[ _Piper(logit_data,jobQ) for x in range(NPROCESSORS-1) ] ; 
    try : 
        [ s.start() for s in servitors] ;

        jobchunk=list()
        total_cohort_files=0
        fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'
        while 1 : 

            try : 
                jobchunk.append(next(patient_gen))
            except StopIteration:
                break ; 

            if len(jobchunk) >= chunksize: 
                jobQ.put((fname,jobchunk))
                jobchunk=list() ;
                total_cohort_files += 1
                fname=dirname+os.sep+f'events{total_cohort_files:04}.npz'
                print(f'Readied cohort file {total_cohort_files:06}',end='\r')

        if jobchunk : 
            jobQ.put((fname,jobchunk))
            print(f'Readied cohort file {total_cohort_files:06}',end='\r')
        print()

        [ jobQ.put(None) for s in servitors ] ; 
        [ s.join() for s in servitors ]  ; 


    finally : 
        for s in servitors : 
            if s.is_alive() : s.terminate() ;
        jobQ.close() ;

from scipy.sparse import csc_matrix
def load_events(dirname) :

    with open(dirname+os.sep+'meta.json','r') as f : 
        meta=json.load(f)

    nevents=meta['nevents']

    files2load=[ dirname+os.sep+f for f in os.listdir(dirname) if f.endswith('.npz') ]
    for fname in files2load : 
        hs=load_npz(fname).tocsc() 
        for x in range(int(hs.shape[1]/nevents)) :
            cohort=hs[:,x*nevents:(x+1)*nevents]
            yield cohort
            
#~~~~~~~~High-level functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def reconstitute_codf(coframe,namesframe,column=None) : 
    """
    Given a tuple <t> where the first item is a list of event names
    and a sparse matrix with len(t) columns (as retuned by a logit polyfunction),
    get the number of coincident observations.
    """

    na,nb   =   zip(*get_coincident_observation_names(namesframe,column=column))
    cs      =   get_coincident_observations(coframe) ;

    if issparse(cs) : 
        return pd.DataFrame().assign(event_a=na,event_b=nb,observed=cs.toarray()) ;
    else : 
        return pd.DataFrame().assign(event_a=na,event_b=nb,observed=cs) ;
