import json
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
import numpy.typing as npt
import typing
import warnings
import kidgloves as kg
opj=kg.opj
msg=kg.msg
kg.read_config()
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import lars_path,Lasso,LassoLars
from sklearn.model_selection import RepeatedKFold
import multiprocessing as mp
import time
elapsed=lambda t : time.time()-t
from tqdm.auto import tqdm
from itertools import product
from functools import reduce
from statsmodels.stats.multitest import multipletests
from scipy.stats import bootstrap
from scipy.stats.distributions import t
import torch
import torch.sparse
import sptops
from scipy.stats import binom
from scipy.special import expit
from dataclasses import dataclass
from sklearn.model_selection import KFold
CPU=torch.device('cpu')
if torch.cuda.is_available() : 
    DEVICE=torch.device('cuda:0')
    print("Detected GPU.")
else : 
    DEVICE=CPU

def msg(*args,**kwargs) : 
    print(time.strftime("%m%d %H:%M:%S",time.localtime())+'|',*args,**kwargs)
    sys.stdout.flush()

#~~~~~~~~LARGeSSE-G core functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def spoof(hier,preserve_rowsums=True) : 
    systemsizes=[ len(v) for k,v in hier.items() ]
    memberships=[ vi for k,v in hier.items() for vi in v ]
    if preserve_rowsums : 
        mq=list(np.random.choice(memberships,size=len(memberships),replace=False))
    else: 
        mq=list(np.random.choice(list(set(memberships)),size=sum(systemsizes)*2,replace=True))
    
    outhier=dict()
    sysind=0
    
    while len(mq) > 0 and len(systemsizes) > 0 : 
        
        syssize=systemsizes.pop()
        newsyscontents=set()
        for i in range(syssize) : 
            if len(mq) < 1 : break
            newsyscontents.add(mq.pop())
                
        outhier.update({'sys{:0>5}'.format(sysind) : newsyscontents })
        sysind += 1
        
    return outhier
            
    
@np.vectorize
def _v_fix_tcga(barcode) :     
    return '-'.join(barcode.split('-')[:3])

def _fix_overlong_identifiers(df,index=True) :
    if index: 
        df.index=[ '-'.join(x.split('-')[:3]) for x in df.index ]
        assert len(df.index) == len(set(df.index))
    else : 
        df.columns=[ '-'.join(x.split('-')[:3]) for x in df.columns ]
        assert len(df.columns) == len(set(df.columns))

    return df

def prep_guts(omics_table,mutation_signature_path) :
    omics=_fix_overlong_identifiers(pd.read_csv(omics_table,index_col=0))
    msig=_fix_overlong_identifiers(pd.read_csv(mutation_signature_path,index_col=0))
    pats=np.intersect1d(omics.index,msig.index)
    #FUTURE : this might not stricly be necessary? What matters is that you know the correlations for each gene
    #Therefore, assuming those are not invariant for patients where you just have regular and not allele-specific CNA data
    #You are probably OK to use more patients
    # AS LONG AS you compile omics from other things
    return omics.reindex(pats),msig.reindex(pats)
    

def _prep_guts_legacy(tcga_directory,mutation_signature_path) :
    omics=kg.autoload_events(tcga_directory,heuristic=False,n2keep=0,mutations_from=opj(os.path.split(mutation_signature_path)[0],'mutations.maf'))
    msig=pd.read_csv(mutation_signature_path,index_col=0)
    omics=_fix_overlong_identifiers(omics)
    omics=omics.reindex(msig.index)

    return omics,msig

def load_protein_datas() : 
    #FUTURE : customize script to accommodate different timings

    if kg._s2e is None : 
        kg._get_geneinfo()

    acconly=np.vectorize(lambda x : x.split('.')[0])
    eidonly=np.vectorize(lambda x : x.split('_')[0])

    grh=pd.read_csv('/cellar/users/mrkelly/Data/canon/ncbi_reference/gene2refseq_human.tsv',
            names=['taxid','GeneID','status','msg_acc','msg_gi','protein_acc','protein_gi','genomic','genoimc_gi','gstart','gend','strand','assembly','matpep_acc','matpep_gi','Symbol'],
            sep='\t',low_memory=False)
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
    rto= rt.set_index('GeneID').coord
    rto=rto.sort_values(ascending=False)
    rto=rto[ ~rto.index.duplicated(keep='first') & ~rto.index.isnull() ].dropna()

    return np.log10(fgb),rto


from scipy.stats.distributions import t
def regularize_cc(ccmat,n,correlation_p=1e-3) : 
    tstats=ccmat*np.sqrt(n-2)/np.sqrt(1-ccmat*ccmat)
    tthresh=t.isf(correlation_p,df=n-2)
    sigmask=tstats > tthresh
    out=ccmat.copy()
    out[ ~sigmask ]=0
    
    return out

from scipy.stats.distributions import t
def regularize_cc_torch(ccten,n,correlation_p=1e-3) : 
    tstats=ccten*torch.sqrt(torch.tensor(n,device=ccten.device)-2)/torch.sqrt(1-ccten*ccten)
    tthresh=t.isf(correlation_p,df=n-2)
    sigmask=tstats > tthresh
    out=ccten.clone()
    out[ ~sigmask ]=0
    
    return out

def cc2mats(m1,m2) :
    """
    This function finds the correlation coefficient between all **columns** of two matrices m1 and m2.
    (The rows are the observations measured across all variables in both columns).
    This does _not_ show the correlation coefficient between any two variables that are both columns of the 
    same matrix.
    """

    m1bar=m1.mean(axis=0)
    m2bar=m2.mean(axis=0)
    m1sig=m1.std(axis=0)
    m2sig=m2.std(axis=0)
    n=m1.shape[0]

    cov=np.dot(
            (m1-m1bar).transpose(),
            (m2-m2bar)
            )
    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')  
        cov1=cov/m2sig
        cov1[ np.isnan(cov1) ]=0
        cov2=cov1.transpose()/m1sig
        cov2[ np.isnan(cov2) ]=0
        out=cov2.transpose()/n

    return out

def cc2tens(t1,t2) : 
    """
    This function finds the correlation coefficient between all **columns** of two dense tensors t1 and t2.
    (The rows are the observations measured across all variables in both columns).
    This does _not_ show the correlation coefficient between any two variables that are both columns of the 
    same matrix.
    """
    if t1.is_sparse: 
        t1=t1.clone().to_dense()
    if t2.is_sparse: 
        t2=t2.clone().to_dense()


    t1bar=t1.mean(axis=0)
    t2bar=t2.mean(axis=0)
    t1sig=t1.std(axis=0)
    t2sig=t2.std(axis=0)
    n=t1.shape[0]

    cov=torch.matmul(
            (t1-t1bar).transpose(0,1),
            (t2-t2bar)
            )
    cov1=cov/t2sig
    cov1[ torch.isnan(cov1) ]=0
    cov2=cov1.transpose(0,1)/t1sig
    cov2[ torch.isnan(cov2) ]=0
    out=cov2.transpose(0,1)/n
    return out



event2eid=lambda c: c.split('_')[0]

class LARGeSSE_G(object) : 

    def __init__(self,**kwargs) : 
        super(LARGeSSE_G,self).__init__()
        self.device=kwargs.get('device',DEVICE)

        self.gen=np.random.RandomState(np.random.MT19937(seed=int('0xc0ffee',16))) 

        omics=kwargs.get('omics')
        signatures=kwargs.get('signatures')
        if omics is not None : 
            self._assign_omics(omics)

        if signatures is not None: 
            self._assign_signatures(signatures)

        if hasattr(self,'omics') and hasattr(self,'signatures') : 
            self._sync_feeder_indices()

        self.correlation_p=kwargs.get('correlation_p',1e-3)

        lengths=kwargs.get('lengths')
        if lengths is not None : 
            self._assign_lengths(lengths)

        timings=kwargs.get('timings')
        if timings is not None : 
            self._assign_timings(timings)

        hierarchy=kwargs.get('hierarchy')

        if hierarchy is not None : 
            self._assign_hierarchy(hierarchy)

    def _assign_lengths(self,lengths) : 
        assert hasattr(self,'omics')
        self.lengths=lengths.copy()
        self.lengths.index=np.vectorize(lambda s : s+'_mut')(self.lengths.index)
        self.lengths=self.lengths.reindex(self.full_index).fillna(0)
        self.t_lengths=torch.tensor(self.lengths.values,dtype=DTYPE,device=CPU)

    def _assign_timings(self,timings) : 
        assert hasattr(self,'omics')
        self.timings=timings.copy()
        self.timings.index=np.vectorize(lambda s : s+'_mut')(self.timings.index)
        self.timings=self.timings.reindex(self.full_index).fillna(0)
        self.t_timings=torch.tensor(self.timings.values,dtype=DTYPE,device=CPU)

    def _assign_hierarchy(self,hierarchy,system_limit_upper=2000,system_limit_lower=4) : 
        assert hasattr(self,'omics')
        assert hasattr(self,'y')

        self.hierarchy={ k : v for k,v in hierarchy.items()
                            if len(v) < system_limit_upper and len(v) >= system_limit_lower }
        if len(self.hierarchy) < 1 : 
            self.nmo=None
            self.systems=None
            return

            
        gd={ g : i for i,g in enumerate(self.genes) }
        ssk=sorted(self.hierarchy.keys())
        #sd={ s : j for j,s in enumerate(ssk) }

        row_indices=list()
        col_indices=list()
        for j,sk in enumerate(ssk) : 
            ri=[ gd[g] for g in self.hierarchy[sk] if g in gd]
            row_indices.extend(ri)
            col_indices.extend([j]*len(ri))
            
        sphier=torch.sparse_coo_tensor(
            indices=np.c_[col_indices,row_indices].transpose(),
            values=np.ones((len(row_indices,))),
            size=(len(ssk),len(self.genes)),
            device=CPU
        ).coalesce().transpose(0,1).float()

        self.nmo=torch.sparse.mm(self.ii,sphier)

        self.systems=ssk

    def _assign_omics(self,omics,resolve_cn_conflicts=True) : 
        """
        omics should be a pd.DataFrame as generated by kidgloves.autoload_events
        """


        self.genes=np.unique(np.vectorize(event2eid)(omics.columns)) 
        self.full_index=np.array([ '_'.join([eid,suf]) for suf,eid in product(['mut','fus','up','dn'],self.genes) ])
        self.omics=_fix_overlong_identifiers(omics).reindex(columns=self.full_index).fillna(0).astype(np.float32)

        ltypes=np.r_[*[[lt]*len(self.genes) for lt in ['mut','fus','up','dn']]]
        if resolve_cn_conflicts: 
            ltypes=np.r_[*[[lt]*len(self.genes) for lt in ['mut','fus','up','dn']]]
            uptotals=self.omics.values[:,( ltypes == 'up' )].sum(axis=0)+1
            dntotals=self.omics.values[:,( ltypes == 'dn' )].sum(axis=0)+1
            lquot=np.log(uptotals/dntotals)/np.log(2)
            keepups=( lquot > -1 ) | (uptotals+dntotals < 5)
            keepdns=(lquot < 1 ) | (uptotals+dntotals < 5)
            omask=np.r_[[True]*2*len(self.genes),keepups,keepdns]

            self.omics=self.omics[ self.omics.columns[omask]]

        self.t_omics=torch.tensor(self.omics.values,
                                    dtype=DTYPE,
                                    device=self.device)

        self._c_omics_mut=torch.tensor(self.omics.columns.str.endswith('_mut'),device=CPU)
        self._c_omics_fus=torch.tensor(self.omics.columns.str.endswith('_fus'),device=CPU)
        self._c_omics_up=torch.tensor(self.omics.columns.str.endswith('_up'),device=CPU)
        self._c_omics_dn=torch.tensor(self.omics.columns.str.endswith('_dn'),device=CPU)
        self._c_omics_struct= self._c_omics_fus | self._c_omics_up | self._c_omics_dn

        self._r_genes=np.vectorize(event2eid)(omics.columns)
        # at each row of **omics**, what gene is being represented?

        self._t_row_indices=torch.arange(len(self._r_genes),device=CPU,dtype=torch.float)

        self.build_y()

        iivalues=np.ones((len(self._r_genes),))

        # at each row, what is the column of each gene being referenced
        cog=dict(zip(self.genes,list(range(len(self.genes)))))
        # eid -> column
        lam=np.vectorize(lambda x : cog[x])
        iirow_indices=torch.tensor(np.arange(len(self._r_genes)),device=CPU)
        # will be slot in on dimension 0 but then tranposed
        iicol_indices=torch.tensor(lam(self._r_genes),device=CPU)


        # the columns are genes
        # the rows are events
        self.ii=torch.sparse_coo_tensor(
                        indices=torch.stack([iicol_indices,iirow_indices],axis=-1).transpose(0,1),
                        values=iivalues,
                        size=(len(self.genes),len(self._r_genes)),
                        device=CPU,
                  ).coalesce().transpose(0,1).float().coalesce()



    def _assign_signatures(self,signatures) : 
        self._c_signature_sbs=torch.tensor(signatures.columns.str.startswith('SBS'),device=CPU)
        self._c_signature_dbs=torch.tensor(signatures.columns.str.startswith('DBS'),device=CPU)
        self._c_signature_id=torch.tensor(signatures.columns.str.startswith('ID'),device=CPU)
        self._c_signature_cn=torch.tensor(signatures.columns.str.startswith('CN'),device=CPU)
        self._c_signature_arm=torch.tensor(signatures.columns.str.startswith('arm_pc'),device=CPU)
        self._c_signature_point=self._c_signature_sbs | self._c_signature_dbs | self._c_signature_id
        self._c_signature_region=self._c_signature_cn | self._c_signature_arm
        self.signatures=_fix_overlong_identifiers(signatures).astype(np.float32)

        self.t_signatures=torch.tensor(self.signatures.values,device=self.device,dtype=DTYPE)

        self.signature_names=np.array(self.signatures.columns)

    def _sync_feeder_indices(self): 
        pats=np.intersect1d(self.omics.index,self.signatures.index)
        self.patients=pats
        self._t_pat_indices=torch.arange(len(pats),device=CPU,dtype=torch.int)
        self._assign_omics(self.omics.reindex(pats))
        self._assign_signatures(self.signatures.reindex(pats))

    def features(self,force=False) : 
        assert hasattr(self,'omics')
        assert hasattr(self,'nmo')
        assert hasattr(self,'signatures')

        systems=self.systems if self.systems is not None else list()

        if (not hasattr(self,'_features') or (self._features is None)) or force : 
            self._features=np.r_[self.genes,systems,self.signature_names,['max_prot_length','replication_timing']]
            #self._features=np.r_[self.genes,self.systems,self.signature_names,['max_prot_length','replication_timing','intercept']]
        return self._features

    def featuretypes(self,force=False) : 
        systems=self.systems if self.systems is not None else list()

        if (not hasattr(self,'_featuretypes') or (self._featuretypes is None)) or force : 
            self._featuretypes = np.r_[['gene']*len(self.genes),['system']*len(systems),['signature']*len(self.signature_names),['length','timing']]
            #self._featuretypes = np.r_[['gene']*len(self.genes),['system']*len(self.systems),['signature']*len(self.signature_names),['length','timing','intercept']]
        return self._featuretypes

    def build_J(self,patient_mask=None,gene_mask=None,correlation_p=None,inplace=False) :

        if correlation_p is None : 
            correlation_p=self.correlation_p 

        if not patient_mask is None : 
            omics=self.t_omics[patient_mask,:].to(self.device)
            signatures=self.t_signatures[patient_mask,:].to(self.device)
        else : 
            omics=self.t_omics
            signatures=self.t_signatures

        if gene_mask is None : 
            gcols=self._t_row_indices
            mutcols= self._c_omics_mut
            structcols= self._c_omics_struct
        else : 
            gcols=self._t_row_indices[gene_mask]
            mutcols=gene_mask | self._c_omics_mut
            structcols=gene_mask | self._c_omics_struct

        with warnings.catch_warnings() : 
            warnings.simplefilter('ignore')


            cc_mut_vs_point=regularize_cc_torch(
                cc2tens( omics[:,mutcols],
                    signatures[:,self._c_signature_point],
                ),
                n=omics.shape[0],
                correlation_p=correlation_p
                ).float()
            
            cc_str_vs_region=regularize_cc_torch(
                cc2tens( omics[:,structcols],
                    signatures[:,self._c_signature_region],
                ),
                n=omics.shape[0],
                correlation_p=correlation_p
                ).float()

            ccmvpnz=cc_mut_vs_point.nonzero().transpose(0,1)
            sp_mvp=torch.sparse_coo_tensor(
                    indices=ccmvpnz,
                    values=cc_mut_vs_point[*ccmvpnz].ravel(),
                    size=cc_mut_vs_point.shape,
                    device=self.device #new
                    ).coalesce()

            ccsvrnz=cc_str_vs_region.nonzero().transpose(0,1)
            sp_svr=torch.sparse_coo_tensor(
                    indices=ccsvrnz,
                    values=cc_str_vs_region[*ccsvrnz].ravel(),
                    size=cc_str_vs_region.shape,
                    device=self.device #new
                    ).coalesce()

            sp_svr_adj_indices=sp_svr.indices()+torch.tensor(sp_mvp.shape,device=self.device).reshape(-1,1)

            sp_J=torch.sparse_coo_tensor(
                    indices=torch.cat([sp_mvp.indices(),sp_svr_adj_indices],axis=1),
                    values=torch.cat([sp_mvp.values(),sp_svr.values()],axis=0),
                    size=(len(gcols),self._c_signature_point.sum().item()+self._c_signature_region.sum().item()),
                    device=self.device #new
                    ).coalesce()

        if inplace: 
            self.J=sp_J

        return sp_J

    def build_y(self) : 
        assert hasattr(self,'omics')
        self.y=self.omics.sum(axis=0)
        self.t_y=torch.tensor(self.y.values,device=self.device,dtype=DTYPE).reshape(-1,1)

    def build_IH(self,weight=False,gene_mask=None,inplace=False,patient_mask=None) : 
        assert hasattr(self,'omics')
        assert hasattr(self,'hierarchy')
        if not hasattr(self,'y'): 
            self.build_y()


        if gene_mask is not None : 
            ii=mask_sparse_rows(self.ii,gene_mask).coalesce()
            if self.nmo is not None :
                nmo=mask_sparse_rows(self.nmo,gene_mask).coalesce()
            else : 
                nmo=self.nmo
        else : 
            ii=self.ii
            nmo=self.nmo

        if nmo is None : 
            ih=ii.clone()
        else : 
            nmo_indices_adj=(torch.tensor([[0,ii.shape[1]]],device=CPU)+nmo.indices().transpose(0,1)).transpose(0,1)
            ih=torch.sparse_coo_tensor(
                values=torch.cat((ii.values(),nmo.values()),axis=0),
                indices=torch.cat((ii.indices(),nmo_indices_adj),axis=1),
                size=(ii.shape[0],ii.shape[1]+nmo.shape[1]),
                device=CPU,
            )
        #FLAG : you may need to backtrack through here if running with resampling becomes very slow

        if weight : 
            psty=torch.tensor(t_y+1,device=CPU).ravel()
            sppsty=psty.to_sparse_coo()
            desppsty=sptops.diag_embed(sppsty)

            ihw=torch.sparse.mm(desppsty.coalesce(),ih) # this multiplies across the values of y to each element

            ihw_colsums=sptops.diag_embed(torch.pow(ihw.sum(axis=0),-1)) # this is the inverse of the column sums

            ih=torch.sparse.mm(ihw,ihw_colsums) # this divides by the column sums

        self.IH=ih

        return ih

    def build_X(self,inplace=False,normalize=True,**kwargs) : 

        IH=kwargs.get('IH',self.IH).coalesce()
        J=kwargs.get('J',self.J).coalesce()
        timings=kwargs.get('timings',self.t_timings).to(self.device).reshape(-1,1)
        lengths=kwargs.get('lengths',self.t_lengths).to(self.device).reshape(-1,1)

        j_shift=IH.shape[1]
        length_shift=j_shift+J.shape[1]
        timing_shift=length_shift+1

        tnz=timings.nonzero().transpose(0,1)
        tv=torch.sparse_coo_tensor(
            indices=tnz,
            values=timings[tnz[0].ravel()].ravel(),
            size=timings.shape,
            device=CPU
            ).coalesce()

        lnz=lengths.nonzero().transpose(0,1)
        lv=torch.sparse_coo_tensor(
            indices=lnz,
            values=lengths[lnz[0].ravel()].ravel(),
            size=lengths.shape,
            device=CPU
            ).coalesce()

        j_adj_indices=J.indices().cpu()+torch.tensor([[0],[j_shift]],device=CPU)
        length_adj_indices=lv.indices()+torch.tensor([[0],[length_shift]],device=CPU)
        timing_adj_indices=tv.indices()+torch.tensor([[0],[timing_shift]],device=CPU)
        #intercept_indices=torch.tensor(np.c_[np.zeros((IH.shape[0],),dtype=float),(1+timing_shift)*np.ones((IH.shape[0],),dtype=float)].transpose())

        X=torch.sparse_coo_tensor(
                indices=torch.cat([IH.indices(),j_adj_indices,length_adj_indices,timing_adj_indices],axis=1),
                values=torch.cat([IH.values(),J.values().cpu(),lv.values(),tv.values()],axis=0),
                size=(IH.shape[0],IH.shape[1]+J.shape[1]+2),
                device=self.device #new
                ).float()

        #X=torch.sparse_coo_tensor(
                #indices=np.c_[IH.indices(),j_adj_indices,length_adj_indices,timing_adj_indices,intercept_indices],
                #values=np.r_[IH.values(),J.values(),lv.values(),tv.values(),-1*np.ones((IH.shape[0],))],
                #size=(IH.shape[0],IH.shape[1]+J.shape[1]+3)
                #).float()

        if normalize : 

            xdm=X.to_dense().max(axis=0).values
            xdmde=torch.sparse_coo_tensor(
                    indices=np.c_[np.arange(len(xdm)),np.arange(len(xdm))].transpose(),
                    values=xdm,
                    size=(len(xdm),len(xdm))
                    )
            X=torch.mm(X,xdmde)

        if inplace: 
            self.X=X
        return X

    def guess_weights(self) : 
        guesses=torch.clip(cc2tens(self.X,self.t_y),0,torch.inf).ravel()
        guess_intercept=torch.tensor(-5,device=DEVICE,dtype=torch.float)
        return guesses,guess_intercept


    def sample_patients(self,frac=None,n=1,replace=False) : 

        assert hasattr(self,'patients')

        if frac is not None : 
            n=int(frac*len(self.patients)//1.0)

        return torch.multinomial(input=torch.ones_like(self._t_pat_indices).float(),num_samples=n).int()

    def sample_rows(self,frac=None,n=1,replace=False) : 

        # right now this 

        assert hasattr(self,'genes')
        if frac is not None : 
            n=int(frac*len(self._r_genes)//1.0)

        return torch.multinomial(input=torch.ones_like(self._t_row_indices),num_samples=n).int()

    def sample_genes(self,frac=None,n=1,replace=False) : 

        assert hasattr(self,'genes')
        if frac is not None : 
            n=int(frac*len(self.genes)//1.0)

        picked_genes=torch.tensor(
                np.argwhere(
                        np.isin(
                            self._r_genes,
                            self.gen.choice(self.genes,n,replace=False)
                            )
                        ).ravel(),
                device=self.device,
                dtype=torch.int
            )
        # these are indices
        return picked_genes

    def sample_xy(self,gene_frac=None,gene_n=None,patient_frac=None,patient_n=None,recalc_J=False,renormalize_X=False,reweight_IH=False) : 

        if gene_frac is not None or gene_n is not None : 
            genes=self.sample_genes(frac=gene_frac,n=gene_n)
        else : 
            genes=self._t_row_indices.int()

        if patient_frac is not None or patient_n is not None : 
            patients=self.sample_patients(frac=patient_frac,n=patient_n)
        else : 
            patients=self._t_pat_indices.int()

        return self.build_xy_from_subset(genes,patients,recalc_J=recalc_J,renormalize_X=renormalize_X)


    def build_xy_from_subset(self,gene_indices,patient_indices,recalc_J=False,renormalize_X=False,reweight_IH=False) : 

        if gene_indices is None : 
            gene_mask=None
            timings=self.t_timings.clone()
            lengths=self.t_lengths.clone()
        else : 
            gene_mask=torch.isin(self._t_row_indices,gene_indices)
            timings=self.t_timings[ gene_indices ].clone()
            lengths=self.t_lengths[ gene_indices ].clone()

        patient_mask=torch.isin(self._t_pat_indices,patient_indices)
        sih=self.build_IH(gene_mask=gene_mask,patient_mask=patient_mask,weight=reweight_IH,inplace=False)

        if recalc_J : 
            sj=self.build_J(gene_mask=gene_mask,patient_mask=patient_mask)
        else : 
            if gene_mask is not None : 
                sj=mask_sparse_rows(self.J,gene_mask)
            else : 
                sj=self.J


        sX=self.build_X(IH=sih,J=sj,timings=timings,lengths=lengths,normalize=renormalize_X,inplace=False)
        if gene_indices is not None and patient_indices is not None : 
            sy=self.t_omics[patient_indices,:][:,gene_indices].sum(axis=0)
        elif patient_indices is not None : 
            sy=self.t_omics[patient_indices,:].sum(axis=0)
        elif gene_indices is not None : 
            sy=self.t_omics[gene_indices,:].sum(axis=0)
        else : 
            sy=self.t_omics.sum(axis=0)

        return (sX,sy)

def mask_sparse_rows(t,mask) :
    imi=torch.argwhere(mask).ravel()
    new_indices=torch.cumsum(mask,0)-1
    timask=torch.isin(t.indices()[0],imi)
    stvals=t.values()[timask]
    ti=t.indices()[:,timask]
    stindices=torch.stack([new_indices[ti[0]],ti[1]],axis=-1).transpose(0,1)
    st=torch.sparse_coo_tensor(
        values=stvals,
        indices=stindices,
        size=(mask.sum(),t.shape[1]),
        device=t.device,
    )

    return st

def mask_sparse_columns(t,mask) : 
    imi=torch.argwhere(mask).ravel()

    new_indices=torch.cumsum(mask,0)-1
    
    timask=torch.isin(t.indices()[1],imi)

    stvals=t.values()[timask]
    
    ti=t.indices()[:,timask]

    stindices=torch.stack([ti[0],new_indices[ti[1]]],axis=-1).transpose(0,1)
    
    st=torch.sparse_coo_tensor(
        values=stvals,
        indices=stindices,
        size=(t.shape[0],mask.sum()),
        device=t.device,
    )

    return st

def cast_tensor(t,device=DEVICE,**kwargs) : 
    if torch.is_tensor(t) : 
        return t
    else :
        return torch.tensor(t,device=device,**kwargs)

def copy_tensor(t,device=DEVICE) : 
    if torch.is_tensor(t) : 
        return t.clone()
    else :
        return torch.tensor(t,device=device)



class LARGeSSE_Logprob(torch.nn.Module) : 
    def __init__(self, nfeats,init_vals=None,device=DEVICE,init_intercept=None):
        super(LARGeSSE_Logprob, self).__init__()
        
        self.relu = torch.nn.ReLU()
        
        if init_vals is None : 
            self.weights=torch.nn.Parameter(torch.empty(size=(nfeats,),requires_grad=True,device=device,dtype=DTYPE))
            torch.nn.init.uniform_(self.weights,a=0,b=1)
        else : 
            self.weights = torch.nn.Parameter(init_vals.clone())
            
        if init_intercept is None : 
            self.intercept=torch.nn.Parameter(torch.tensor(0,dtype=DTYPE,device=device))
        else : 
            self.intercept=torch.nn.Parameter(copy_tensor(init_intercept).float().to(device))
            
    def forward(self,X,return_weights=False) :
        corrected_weight=self.relu(self.weights) ;
        bigdot=torch.matmul(X,corrected_weight)+self.intercept
        
        if not return_weights : 
            return bigdot
        else : 
            return bigdot,corrected_weight
    
class MultiPenaltyBinomialLoss(torch.nn.Module) : 
    def __init__(self,strengths,masks,device=DEVICE) : 
        # strengths is an iterable container of floats
        # masks is an iterable container of boolean numpy arrays spanning the feature space
        # the sum of coefficients masked by mask 0 is multiplied by strength 0, etc.
        
        super(MultiPenaltyBinomialLoss,self).__init__()
        
        self.strengths=[ cast_tensor(s,device=device,dtype=DTYPE,requires_grad=False)
                        for s in strengths ]
        self.masks=[ cast_tensor(m,device=device,dtype=torch.bool,requires_grad=False)
                        for m in masks ]

    def forward(self,output_log_odds,target_event_cts,weights,n) : 
        
        eps=torch.finfo(output_log_odds.dtype).eps
        
        output_probs=torch.clip(torch.special.expit(output_log_odds),eps,1-eps)
        log_output_probs=torch.log(output_probs)
        k=target_event_cts
        comb=torch.lgamma(n+1)-torch.lgamma(k+1)-torch.lgamma(n-k+1)
        indiv_ells=(log_output_probs*k + torch.log(1-output_probs)*(n-k))
        llterm=(comb+(indiv_ells)).sum()

        if len(k.shape) == 1 : 
            kshape=k.shape[0]
        else : 
            kshape=max(k.shape) 
        
        penalty=sum([ np.exp(s.cpu().numpy())*kshape*weights[m].sum() for s,m in zip (self.strengths,self.masks) ])
        #penalty=sum([ np.exp(s.cpu().numpy())*m.sum()*(weights[m]).sum() for s,m in zip (self.strengths,self.masks) ])
        # why was this? fiting different numbers of event types causes the penalty to scale improperly
        # len(weights) new as of 20240216

        return penalty-llterm

_default_lgbfs_kwargs=dict(
max_iter=20,
history_size=10
)

DTYPE=torch.float


@dataclass
class OptimizerResult : 
    """ Class for storing results from the `run_model` function"""
    loss : float
    weight : npt.NDArray[typing.Any]
    intercept : npt.NDArray[typing.Any] 
    strengths : list 
    masks : tuple 
    sampling_epochs : npt.NDArray[typing.Any]
    max_epochs : int 
    lr  : float
    lbgfs_kwargs : dict
    convergence_status : bool
    patient_indices_train :  typing.Optional[npt.NDArray[typing.Any]]  = None
    patient_indices_test :  typing.Optional[npt.NDArray[typing.Any]]    = None

@dataclass
class AICResult: 
    criterion : float
    k : int
    ll : float
    n : int

@dataclass
class Settings: 
    """ Class for storing run preferences"""

# expected to be different between them
#~~~~~~~~Process config~~~~~~~~~~~~~~~~~
    save_tmps :  bool
    tmpprefix : str
    jobprefix : str
    output_directory : str

#~~~~~~~~Initialization config~~~~~~~~~~
    hierarchy_path : str
    omics_path : str
    mutation_signature_path : str
    correlation_p : float
    j_regstrength : float
    hsystem_limit_lower : 3
    hsystem_limit_upper : 2000

#~~~~~~~~Burn-in config~~~~~~~~~~~~~~~~~
    burn_init_regstrength : int =10
    burn_max_epochs_per_run : int=2001
    burn_n_sampling_epochs_per_run : int=201
    burn_lr : float =5e-4
    burn_n_noh_runs : int = 10
    burn_grad_floor : float =0.005
    burn_convergence_timeout : int = 15
#~~~~~~~~Main run config~~~~~~~~~~~~~~~~
    main_max_epochs : int = 4001
    main_n_sampling_epochs : int = 401
    main_lr : float = 2e-4
#~~~~~~~~xval  config~~~~~~~~~~~~~~~~
    n_xval_repeats : int = 10

#~~~~~~~~Bootstrap config~~~~~~~~~~~~~~~
    n_boot_runs : int = 1000
    boot_patient_resampling_fraction  : float = 0.9
    boot_max_epochs : int = 2001 
    boot_n_sampling_epochs : int = 401
    boot_lr : float = 1e-2

#~~~~~~~~Spoof config~~~~~~~~~~~~~~~
    n_spoofs : int = 20
    spoof_lr  : float = 1e-4

#~~~~~~~~Legacy~~~~~~~~~~~~~~~~~~~~~
    tcga_directory : str=''
    
@dataclass
class BurnInResult : 
  frame : typing.Any
  ofinterest : npt.NDArray[typing.Any]
  strengths : typing.List[float]


class Logger(object) : 
    def __init__(self,
                 model,
                 losser,
                 sampling_epochs,
                 nfeats,
                 check_convergence=False,
                 sampling_epochs_to_check_convergence=10,
                 sampling_epochs_with_worse_loss_for_convergence=8,
                 device=DEVICE,
            ) : 
        
        super(Logger,self).__init__()
        self.model=model
        self.losser=losser
        self.sampling_epochs=sampling_epochs
        self.n_sampling_epochs=len(sampling_epochs)
        self.se=0
        
        self.wlog=torch.zeros(size=(self.n_sampling_epochs,nfeats),dtype=DTYPE,device=device)
        self.llog=torch.zeros(size=(self.n_sampling_epochs,),dtype=DTYPE,device=device)
        self.ilog=torch.zeros(size=(self.n_sampling_epochs,),dtype=DTYPE,device=device)
        
        
        self.check_convergence=check_convergence
        self.sampling_epochs_to_check_convergence=sampling_epochs_to_check_convergence
        self.sampling_epochs_with_worse_loss_for_convergence=sampling_epochs_with_worse_loss_for_convergence
        self.out_sampling_epochs=None
        
    def __call__(self,epoch,tX,ky,n) : 
        if epoch not in self.sampling_epochs: return False
    
        if epoch in self.sampling_epochs: 
            yhat,weights=self.model(tX,return_weights=True)
            loss=self.losser(yhat,ky.ravel(),weights,n)
            self.wlog[self.se,:]=weights.reshape(1,-1)
            self.llog[self.se]=loss
            self.ilog[self.se]=self.model.intercept.detach()
            
            if self.se > self.sampling_epochs_to_check_convergence and self.check_convergence : 
                concheck=(self.llog[self.se-self.sampling_epochs_to_check_convergence:self.se] < loss).sum()
                # this loss is greater(=worse) than this many of the previous epochs
                if concheck > self.sampling_epochs_with_worse_loss_for_convergence : 
                    convergence_status=True
                    self.wlog=self.wlog[:self.se+1,:]
                    self.llog=self.llog[:self.se+1]
                    self.ilog=self.ilog[:self.se+1]
                    self.out_sampling_epochs=self.sampling_epochs[:self.se+1]
                    return True
                    
            self.se +=1
            
        return False
            
            
    def wrap(self) : 
        if self.n_sampling_epochs > 1 : 
            npwlog=self.wlog.cpu().detach().numpy()
            npllog=self.llog.cpu().detach().numpy()
            npilog=self.ilog.cpu().detach().numpy()
        else : 
            yhat,weights=self.model(tX,return_weights=True)
            npllog=self.losser(yhat,ky.ravel(),weights,n).cpu().detach().numpy()
            npwlog=weights.cpu().detach().numpy()
            npilog=self.ilog.cpu().detach().numpy()
            
        if self.out_sampling_epochs is None :
            self.out_sampling_epochs =self.sampling_epochs
            
        return self.out_sampling_epochs,npllog,npwlog,npilog


def aic_w(w,b0,X,y,n) : 
    eps=np.finfo(w.dtype).eps
    ak=( w > eps).sum()
    phat=np.clip(
            expit(
                np.matmul(X,w)+b0
            ),
            eps,
            1-eps
    )
    ll=binom.logpmf(
        y,
        n,
        phat).sum()

    aicn=X.shape[0]
    
    criterion=2*ak - 2*ll + 2*ak*(ak+1)/max(aicn-1-ak,aicn/ak)
    
    return AICResult(criterion,ak,ll,n)

    
def run_model(lg,
                   strengths,
                   masks,
                   global_feature_mask=None,
                   patient_indices_train=None,
                   patient_indices_test=None,
                   max_epochs=2001,
                   n_sampling_epochs=401,
                   lr=1e-2,
                   lgbfs_kwargs=_default_lgbfs_kwargs,
                   device=DEVICE,
                   check_convergence=True,
                   recalc_J=False,
                   init_vals=None,
                   init_intercept=None,
                   savefile=None,
                  ) : 

    
    if  init_vals is None : 
        init_vals,init_intercept=lg.guess_weights()
        if global_feature_mask is not None : 
            init_vals=init_vals[
                        torch.tensor(global_feature_mask,device=DEVICE,dtype=torch.bool)
                        ]

    elif init_intercept is None : 
        init_intercept=-5

    if global_feature_mask is not None : 
        tX=mask_sparse_columns(lg.X.clone(),torch.tensor(global_feature_mask,device=DEVICE,dtype=torch.bool)).coalesce()
    else : 
        tX=lg.X

    if patient_indices_train is None :
        patient_indices_train=lg._t_pat_indices

    if patient_indices_test is None : 
        patient_indices_test=patient_indices_train

    ytr=lg.t_omics[patient_indices_train,:].sum(axis=0)
    ntr=torch.tensor(len(patient_indices_train),device=lg.device)
    yte=lg.t_omics[patient_indices_test,:].sum(axis=0)
    nte=torch.tensor(len(patient_indices_test),device=lg.device)

    assert tX.shape[1] == init_vals.shape[0]
        
    model=LARGeSSE_Logprob(tX.shape[1],init_vals=init_vals,init_intercept=init_intercept)
    losser=MultiPenaltyBinomialLoss(strengths=strengths,masks=masks)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=lr,**lgbfs_kwargs)
    sampling_epochs=np.cast['int'](np.linspace(0,max_epochs-1,n_sampling_epochs)//1.0)
    logger=Logger(model,losser,sampling_epochs,tX.shape[1],check_convergence=True)

    for epoch in range(max_epochs): 

        def closure() : 
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.zero_grad()
            yhat,weights=model(tX,return_weights=True)
            loss=losser(yhat,ytr,weights,ntr)
            loss.backward()
            return loss

        optimizer.step(closure)

        if convergence_status := logger(epoch,tX,yte,nte) : break

    else : 
        convergence_status=False

    ose,llog,wlog,ilog=logger.wrap()
    output=OptimizerResult(
            loss=llog,
            weight=wlog,
            intercept=ilog,
            strengths=strengths,
            masks=masks,
            sampling_epochs=ose,
            max_epochs=max_epochs,
            lr=lr,
            lbgfs_kwargs=lgbfs_kwargs,
            convergence_status=convergence_status,
            patient_indices_train=patient_indices_train,
            patient_indices_test=patient_indices_test,
            )
    if savefile : 
        os.makedirs(os.path.split(savefile)[0],exist_ok=True)
        torch.save(output,savefile)

    return output

def optimize_regstrenth_itergridsearch(
        lg,
        globalmask,
        masks,
        strengths,
        tweakwhichstrength,
        nkfsplits=5,
        lr=1e-3,
        grad_floor=0.1,
        max_epochs_per_run=2001,
        n_sampling_epochs=201,
        convergence_timeout=15,
        fold_generator_seed='0xcade2ce',
    ) : 
    xnp=lg.X.cpu().to_dense().numpy()
    sfxnp=xnp[:,globalmask]

    init_rs=strengths[tweakwhichstrength]
    lastcmean=np.inf

    init_vals,init_intercept=lg.guess_weights()

    criteria=np.zeros((nkfsplits,))


    ofinterest=np.zeros((0,),dtype=np.int32)

    starttime=time.time()
    #kf=KFold(n_splits=nkfsplits)

    fold_generator=np.random.Generator(np.random.MT19937(int(fold_generator_seed,16)))
    folds_labels=fold_generator.choice(
            a=np.cast['int'](np.arange(lg.omics.shape[0])),
            size=lg.omics.shape[0],
            replace=False,
            )

    nominal_best=strengths[tweakwhichstrength]
    ltchunks=list()

    for hme in range(4) :
        loggertable=pd.DataFrame(
                                columns=[
                                      'metaepoch',
                                      'fold',
                                      'regstrength',
                                      'criterion',
                                      'elapsed',
                                      'mean_vel',
                                      'nparams',
                                  ]
                                )

        range_amp=12/(2**hme)
        mestrrange=np.linspace(nominal_best-range_amp,nominal_best+range_amp,7)

        for metaepoch,strength_this_me in enumerate(mestrrange) : 
            strengths[tweakwhichstrength]=strength_this_me
            for e in range(5) : 
                pitr=np.argwhere(folds_labels != e)
                pite=np.argwhere(folds_labels == e)

                optres=run_model(lg,
                                 strengths=strengths,
                                 global_feature_mask=globalmask,
                                 masks=masks,
                                 lr=lr,
                                 max_epochs=max_epochs_per_run,
                                 patient_indices_train=pitr,
                                 #patient_indices_test=pival,
                                 patient_indices_test=pite,
                                )


                ynp=lg.t_omics[torch.tensor(pite,device=lg.device),:].sum(axis=0).cpu().numpy()

                aicres=aic_w(
                               optres.weight[optres.loss.argmin(),:],
                               optres.intercept[optres.loss.argmin()],
                               sfxnp,
                               ynp,
                               len(pite)
                          )

                best_epoch_this=optres.loss.argmin()
                best_weights_this=optres.weight[ best_epoch_this] 
                eps=np.finfo(np.float32).eps

                nparams=(best_weights_this> eps).sum()

                ofinterest=np.union1d(ofinterest,np.argwhere(best_weights_this > eps).ravel())

                last=pd.Series(dict(
                                   metaepoch=hme*7+metaepoch,
                                   fold=int(e),
                                   regstrength=strengths[tweakwhichstrength],
                                   criterion=aicres.criterion,
                                   elapsed=int(time.time()-starttime//1.0),
                                   nparams=nparams,
                                   convergence_status=optres.convergence_status,
                                   sumweights=best_weights_this.sum(),
                                   lynp=max(ynp.shape),
                                   effective_penalty=sum([ np.exp(s)*max(ynp.shape)*best_weights_this[m].sum() for s,m in zip(strengths,masks) ]),
                               ))

                last.name='{: >2}:{: >3}'.format(hme*7+metaepoch,e)
                if loggertable.shape[0]  < 1: 
                    loggertable=pd.DataFrame([last])
                else: 
                    loggertable=pd.concat([loggertable,pd.DataFrame([last])],axis=0) 

                import __main__ as main
                if hasattr(main,'__file__') : 
                    print(loggertable.tail(1))
                else: 
                    display(loggertable.tail(10),clear=True)

        ltchunks.append(loggertable.copy())
        loggertable=postprocess_burnintable(loggertable[ loggertable.convergence_status ])
                    
        nominal_best=(-1*loggertable.dc*loggertable.regstrength).sum()/(-1*loggertable.dc).sum()

    return pd.concat(ltchunks,axis=0),ofinterest 


def optimize_regstrength_gradient(
        lg,
        globalmask,
        masks,
        strengths,
        tweakwhichstrength,
        nkfsplits=5,
        lr=1e-3,
        grad_floor=0.1,
        max_epochs_per_run=2001,
        n_sampling_epochs=201,
        convergence_timeout=15,
        fold_generator_seed='0xcade2ce',
    ) : 

    xnp=lg.X.cpu().to_dense().numpy()
    sfxnp=xnp[:,globalmask]

    init_rs=strengths[tweakwhichstrength]
    lastcmean=np.inf

    init_vals,init_intercept=lg.guess_weights()

    criteria=np.zeros((nkfsplits,))

    loggertable=pd.DataFrame(
                            columns=[
                                  'metaepoch',
                                  'fold',
                                  'regstrength',
                                  'criterion',
                                  'dv',
                                  'elapsed',
                                  'mean_vel',
                                  'nparams',
                              ]
                            )


    ofinterest=np.zeros((0,),dtype=np.int32)

    metaepoch=0
    grad=0.3
    starttime=time.time()
    dv=1
    lastfivedvs=np.ones((5,))
    #kf=KFold(n_splits=nkfsplits)

    fold_generator=np.random.Generator(np.random.MT19937(int(fold_generator_seed,16)))
    folds_labels=fold_generator.choice(
            a=np.cast['int'](np.arange(lg.omics.shape[0])),
            size=lg.omics.shape[0],
            replace=False,
            )

    try : 
        while not (metaepoch > 3 and np.abs(np.mean(lastfivedvs)) < grad_floor ) : 
            for e in range(5) :

               #trval_indices=np.argwhere(folds_labels != e)
               #trval_indices=fold_generator.choice(
               #        a=trval_indices,
               #        size=trval_indices.shape[0],
               #        replace=False) #shuffle
               #trvalcut=int(trval_indices.shape[0]*4//5)
               #pitr=torch.tensor(trval_indices[:trvalcut],device=lg.device)
               #pival=torch.tensor(trval_indices[trvalcut:],device=lg.device)
               #pite=np.argwhere(folds_labels == e)

                pitr=np.argwhere(folds_labels != e)
                pite=np.argwhere(folds_labels == e)

                optres=run_model(lg,
                                 strengths=strengths,
                                 global_feature_mask=globalmask,
                                 masks=masks,
                                 lr=lr,
                                 max_epochs=max_epochs_per_run,
                                 patient_indices_train=pitr,
                                 #patient_indices_test=pival,
                                 patient_indices_test=pite,
                                )


                ynp=lg.t_omics[torch.tensor(pite,device=lg.device),:].sum(axis=0).cpu().numpy()

                aicres=aic_w(
                               optres.weight[optres.loss.argmin(),:],
                               optres.intercept[optres.loss.argmin()],
                               sfxnp,
                               ynp,
                               len(pite)
                          )

                if metaepoch <= 1: 
                    grad=1
                else : 
                   #dys=aicres.criterion - loggertable.query("fold == @e").criterion.values # overall improvement vs previous values
                   #dxs=strengths[tweakwhichstrength] - loggertable.query("fold == @e").regstrength.values # said previous values
                   #dyx=dys/dxs # slope
                   #dyx[ np.isnan(dyx) | (dyx == 0)]=0 #zero out infinite slopes
                   #weights=np.clip(np.abs(1/dxs/dxs),1e-7,1e2) # weights inverse of squared distance
                   #weights[ np.isnan(weights) | (weights == 0) | np.isinf(weights) ]=0 # fix zeros in weights
                   #grad=(dyx*weights).sum()/weights.sum()  #weighted gradient
                    last_this_fold=loggertable.query("fold == @e").iloc[-1]

                    drs=strengths[tweakwhichstrength] - last_this_fold.regstrength
                    if (drs == 0) :
                        grad=0
                    else : 
                        grad=(aicres.criterion - last_this_fold.criterion)/drs

                    assert not np.isnan(grad) and not np.isinf(grad) and not np.isinf(-1*grad)
                    # grad will be positive if positive shifts increase criteria
                    # this is the shift in AIC strength from a change in regularization strength of +1
                    #  dv used to be grad/2



                #factor=np.power(decel,metaepoch)

                if metaepoch  > 1 : 
                    #dv = (dv-3*np.sin(np.arctan(grad)))/np.log(metaepoch+5)

                    #if grad is negative, it means that increasing the regularization strength
                    # decreases the criterion
                    # in this case,  arctan(grad) will be negative (in radians)
                    # and sin(grad) will also be negative
                    #grad being negative means that positive shifts decrease criteria
                    # so the shift should be opposite signed to grad

                    dv = -1*np.arctan(grad/1e4)
                    assert not np.isnan(dv) and not np.isinf(dv) and not np.isinf(-1*dv)
                else : 
                    dv=-2

                best_epoch_this=optres.loss.argmin()
                best_weights_this=optres.weight[ best_epoch_this] 
                eps=np.finfo(np.float32).eps

                nparams=(best_weights_this> eps).sum()

                ofinterest=np.union1d(ofinterest,np.argwhere(best_weights_this > eps).ravel())

                last=pd.Series(dict(
                                   metaepoch=metaepoch,
                                   fold=int(e),
                                   regstrength=strengths[tweakwhichstrength],
                                   criterion=aicres.criterion,
                                   grad=grad,
                                   dv=dv,
                                   elapsed=int(time.time()-starttime//1.0),
                                   mean_vel=np.mean(lastfivedvs),
                                   nparams=nparams,
                                   convergence_status=optres.convergence_status,
                                   sumweights=best_weights_this.sum(),
                                   lynp=max(ynp.shape),
                                   effective_penalty=sum([ np.exp(s)*max(ynp.shape)*best_weights_this[m].sum() for s,m in zip(strengths,masks) ]),
                               ))

                last.name='{: >2}:{: >3}'.format(metaepoch,e)
                if loggertable.shape[0]  < 1: 
                    loggertable=pd.DataFrame([last])
                else: 
                    loggertable=pd.concat([loggertable,pd.DataFrame([last])],axis=0) 

                import __main__ as main
                if hasattr(main,'__file__') : 
                    print(loggertable.tail(1))
                else: 
                    display(loggertable.tail(10),clear=True)


                if metaepoch > 1 or ( e == 4 ) : 
                    if aicres.k == 0 : 
                        #strengths[tweakwhichstrength] = abs(strengths[tweakwhichstrength])/5
                        strengths[tweakwhichstrength] = strengths[tweakwhichstrength] - 10
                    else : 
                        strengths[tweakwhichstrength] = strengths[tweakwhichstrength] + dv 

                elif metaepoch > 0  : 
                    if aicres.k < globalmask.sum()/2 : 
                        strengths[tweakwhichstrength] = strengths[tweakwhichstrength] - np.exp(-1)
                    else: 
                        strengths[tweakwhichstrength] = strengths[tweakwhichstrength] + np.exp(-1)
                
                lastfivedvs[:-1]=lastfivedvs[1:]
                lastfivedvs[-1]=dv

            if metaepoch > convergence_timeout : break
            metaepoch += 1

        print('Done.')
    except KeyboardInterrupt : 
        #print(dys)
        #print(dxs)
        #print(dyx)
        print(weights)
        print(grad)
        print(dv)
        return loggertable,ofinterest

    return loggertable,ofinterest

def postprocess_burnintable(burnintable) :

    burnintable=burnintable.assign(fold=pd.Categorical(burnintable.fold.astype(int),categories=np.unique(np.cast['int'](burnintable.fold.values))))
    cfmax=burnintable.groupby('fold').criterion.max()
    cfmin=burnintable.groupby('fold').criterion.min()
    burnintable['cfmax']=cfmax.reindex(burnintable.fold).values
    burnintable['cfmin']=cfmin.reindex(burnintable.fold).values
    burnintable['cnorm']=(burnintable.criterion-burnintable.cfmin)/(burnintable.cfmax-burnintable.cfmin)
    burnintable['dc']=(burnintable.criterion-burnintable.cfmax)

    return burnintable

def do_burn_in(lg,settings,savefile=None) : 

    j_regstrength=settings.j_regstrength
    regstrength=settings.burn_init_regstrength

    fts=lg.featuretypes()
    notagene=( fts != 'gene')
    isagene= ~notagene
    isasystem=(fts == 'system')
    notasystem=~isasystem

    burnintable,ofinterest_nogenes=optimize_regstrenth_itergridsearch(
    #burnintable,ofinterest_nogenes=optimize_regstrength_gradient(
                                        lg,
                                        globalmask=notagene,
                                        masks=(isasystem[notagene],(~isasystem)[notagene]),
                                        strengths=[regstrength,j_regstrength],
                                        tweakwhichstrength=0,
                                        lr=settings.burn_lr,
                                        grad_floor=settings.burn_grad_floor,
                                        max_epochs_per_run=settings.burn_max_epochs_per_run,
                                        n_sampling_epochs=settings.burn_n_sampling_epochs_per_run,
                                        convergence_timeout=settings.burn_convergence_timeout,
                                    )

    # registering this back to the full shape, since ofinterest_nogenes has no genes
    full_feat_index=np.arange(lg.X.shape[1])
    nogeneindex=full_feat_index[notagene]
    ofinterest_full=nogeneindex[ofinterest_nogenes]

    burnintable=postprocess_burnintable(burnintable)

    consensus_regstrength=(-1*burnintable.dc*burnintable.regstrength).sum()/(-1*burnintable.dc).sum()
    ostrengths=[consensus_regstrength,settings.j_regstrength]
    # You need to do another model run without the hierarchy but **with** genes in order to figure out which
    # could possibly matter.

    initw,initi=lg.guess_weights()

    msg('Sampling no-hierarchy runs...')
    oi_genes=list()
    from tqdm.auto import tqdm

    start_randseed=1337

    for x in tqdm(range(settings.burn_n_noh_runs),total=settings.burn_n_noh_runs) : 

        bnkwargs=compile_kwargs(lg,settings,which='burn_null',randseed=int(start_randseed*(x+1)))

        aux_optres=run_model(
                    lg,
                    strengths=ostrengths,
                    **bnkwargs,
                   )

        oi_genes.append(np.argwhere(aux_optres.weight[aux_optres.loss.argmin()] > 0 ).ravel())

    for x in range(settings.burn_n_noh_runs) : 
        oi_genes[x]=full_feat_index[notasystem][oi_genes[x]]

    ofinterest=reduce(np.union1d,[ofinterest_full]+oi_genes)

    bir= BurnInResult(frame=burnintable,
                        ofinterest=ofinterest,
                        strengths=ostrengths)
    if savefile : 
        os.makedirs(os.path.split(savefile)[0],exist_ok=True)
        torch.save(bir,savefile)
                                    
    return bir

def do_bootstrapping(lg,
                     ofinterest,
                     strengths,
                     settings,
                    ):

    bootwlog=np.zeros((settings.n_boot_runs,len(ofinterest)))
    from tqdm.auto import tqdm

    isgeneorsystem  =   np.isin(lg.featuretypes(),np.array(['gene','system']))
    issig           =   ~isgeneorsystem
    wasofinterest   =   np.isin(np.arange(lg.X.shape[1]),ofinterest)

    feats=lg.features()

    for i in tqdm(range(settings.n_boot_runs)) : 

        patient_indices=lg.sample_patients(frac=settings.boot_patient_resampling_fraction)
        demo_bs=run_model(lg,
                          strengths=strengths,
                          masks=[ isgeneorsystem[wasofinterest] , issig[wasofinterest]  ],
                          global_feature_mask=wasofinterest,
                          patient_indices_train=patient_indices,
                          max_epochs=settings.boot_max_epochs,
                          n_sampling_epochs=settings.boot_n_sampling_epochs,
                          lr=settings.boot_lr,
                      )

        bootwlog[i,:]=demo_bs.weight[demo_bs.loss.argmin()]
    
        if settings.save_tmps : 
            torch.save(demo_bs,kg.opj(settings.tmpprefix,settings.jobprefix+'_boot_{:0>4}.pt'.format(i)))

    return bootwlog

def assemble_lg(settings) : 

    omics,msig=prep_guts(settings.omics_path,settings.mutation_signature_path)

    lengths,timings=load_protein_datas()

    os.makedirs(settings.output_directory,exist_ok=True)
    import pickle
    with open(settings.hierarchy_path,'rb') as f : 
        hier=pickle.load(f)

    lg=LARGeSSE_G(omics=omics,signatures=msig,lengths=lengths,timings=timings)
    lg._assign_hierarchy(hier,system_limit_upper=settings.hsystem_limit_upper,system_limit_lower=settings.hsystem_limit_lower)
    sp_IH=lg.build_IH(inplace=True,weight=False)
    sp_J=lg.build_J(inplace=True,correlation_p=settings.correlation_p)
    X=lg.build_X(normalize=True,inplace=True)

    return lg

def estimate_sigonly_regstrength(lg,settings,save=True) : 
    fts=lg.featuretypes()
    regstrength=settings.burn_init_regstrength

    isasig=( fts != 'gene') & (fts != 'system')

    burnintable_sigonly,ofinterest_sigonly=optimize_regstrenth_itergridsearch(
                                        lg,
                                        globalmask=isasig,
                                        masks=(isasig[isasig],),
                                        strengths=[settings.j_regstrength,],
                                        tweakwhichstrength=0,
                                        lr=settings.burn_lr,
                                        grad_floor=settings.burn_grad_floor,
                                        max_epochs_per_run=settings.burn_max_epochs_per_run,
                                        n_sampling_epochs=settings.burn_n_sampling_epochs_per_run,
                                        convergence_timeout=settings.burn_convergence_timeout,
                                    )

    burnintable_sigonly=postprocess_burnintable(burnintable_sigonly)

    consensus_regstrength=(-1*burnintable_sigonly.dc*burnintable_sigonly.regstrength).sum()/(-1*burnintable_sigonly.dc).sum()

    if save : 
        burnintable_sigonly.to_csv(opj(settings.output_directory,'sigonly_burn_in.csv'))

    return burnintable_sigonly


def compile_kwargs(lg,settings,which='main',randseed='0xc0ffee') : 
    if type(randseed) == str : 
        randseed=int(randseed,16)

    tg=torch.Generator(device=lg.device).manual_seed(randseed)

    if which  in {'main','null','spoof'} : 
        folds_labels=torch.multinomial(torch.tensor([4,1]).float().to(lg.device),lg.t_omics.shape[0],generator=tg,replacement=True)
        itr=torch.argwhere(folds_labels == 0).ravel()
        ite=torch.argwhere(folds_labels == 1).ravel()
    elif which in {'burn_null'} : 
        #folds_labels=torch.multinomial(torch.tensor([16,4,5]).to(lg.device),lg.t_omics.shape[0],generator=tg,replacement=True)
        folds_labels=torch.multinomial(torch.tensor([16,4,5]).float().to(lg.device),lg.t_omics.shape[0],generator=tg,replacement=True)
        itr=torch.argwhere(folds_labels == 0).ravel()
        ite=torch.argwhere(folds_labels == 1).ravel()

    fts=lg.featuretypes()
    gene_mask= (fts == 'gene')
    system_mask= (fts == 'system') 
    sig_mask = ~gene_mask & ~system_mask
    nonsig_mask =  gene_mask | system_mask

    savefile=None

    match which : 
        case 'main' : 
            global_feature_mask=np.cast['bool'](np.ones_like(nonsig_mask))
            masks=[nonsig_mask,sig_mask]
            savefile=opj(settings.output_directory,'main.pt')
        case 'null' : 
            global_feature_mask=~system_mask
            masks=[nonsig_mask[~system_mask],sig_mask[~system_mask]]
            savefile=opj(settings.output_directory,'matched_null.pt')
        case 'burn_null' : 
            global_feature_mask=~system_mask
            masks=[nonsig_mask[~system_mask],sig_mask[~system_mask]]
        case 'spoof' : 
            global_feature_mask=np.cast['bool'](np.ones_like(nonsig_mask))
            masks=[nonsig_mask,sig_mask]


    return dict(
                masks                   =   masks ,
                global_feature_mask     =   global_feature_mask,
                patient_indices_train   =   itr,
                patient_indices_test    =   ite,
                max_epochs              =   settings.main_max_epochs,
                n_sampling_epochs       =   settings.main_n_sampling_epochs,
                lr                      =   settings.main_lr,
                savefile                =   savefile
            )


def main_script_process(settings): 

    lg=assemble_lg(settings) 
    # interpretation-- you are trying to find the signature regularization strengths
    if len(lg.hierarchy) < 1 : 
        estimate_sigonly_regstrength(lg,settings)  
    else : 
        msg('Burn-in:')
        bir=do_burn_in(lg,settings,)

        msg('Running full main model...')
        main_kws=compile_kwargs(lg,settings,which='main')
        mainout=run_model(lg,strengths=bir.strengths,**main_kws) 
        
        msg('Running full matched null model...')
        null_kws=compile_kwargs(lg,settings,which='null')
        matchnullout=run_model(lg,strengths=bir.strengths,**null_kws)

        msg('Running cross-validation...')
        os.makedirs(opj(settings.output_directory,'xval'),exist_ok=True)


        fts=lg.featuretypes()
        gene_mask= (fts == 'gene')
        system_mask= (fts == 'system') 
        sig_mask = ~gene_mask & ~system_mask
        nonsig_mask =  gene_mask | system_mask


        fold_generator=np.random.Generator(np.random.MT19937(int('0xc0ffee',16)))
        for xvrepeat in range(settings.n_xval_repeats) : 

            folds_labels=fold_generator.choice(
                                a=np.cast['int'](np.arange(lg.omics.shape[0]) // (lg.omics.shape[0]/5)),
                                size=lg.omics.shape[0],
                                replace=False
                            )

            # if you use floor division **within** the denomiator term, you can give values fold labels of '5'
            # if for some reason it is necessary that all folds are trained on the same number of patients

            for fold in range(5) : 

                rflabel='rep{:0>3}fold{:0>3}'.format(xvrepeat,fold)

                pite=torch.tensor(np.argwhere(folds_labels == fold).ravel(),device=lg.device)

                trval_indices=np.argwhere(folds_labels != fold).ravel()
                trval_indices=fold_generator.choice(a=trval_indices,size=trval_indices.shape[0],replace=False)
                trvalcut=int(trval_indices.shape[0]*4//5)
                pitr=torch.tensor(trval_indices[:trvalcut],device=lg.device)
                pival=torch.tensor(trval_indices[trvalcut:],device=lg.device)


                optres_full=run_model(
                                        lg,
                                        strengths=bir.strengths,
                                        masks=(nonsig_mask,sig_mask),
                                        lr=settings.main_lr,
                                        max_epochs=settings.main_max_epochs,
                                        n_sampling_epochs=settings.main_n_sampling_epochs,
                                        patient_indices_train=pitr,
                                        patient_indices_test=pival,
                                    )
                torch.save((pitr,pival,pite),opj(settings.output_directory,'xval','splits_{:}.pt'.format(rflabel)))
                torch.save(optres_full,opj(settings.output_directory,'xval','xval_full_{:}.pt'.format(rflabel)))

        #msg('Getting bootstrap values..')

        full_feat_index=np.arange(lg.X.shape[1])

        ofinterest=reduce(np.union1d,
                [bir.ofinterest,
                 np.argwhere(mainout.weight[mainout.loss.argmin()] > 0).ravel(),
                 full_feat_index[np.argwhere(matchnullout.weight[matchnullout.loss.argmin()] > 0).ravel()],
                 ])

        #bootout=do_bootstrapping(lg,ofinterest,bir.strengths,settings)
        #torch.save(bootout,opj(settings.output_directory,'boot.pt'))

        #msg('Saving most matrices...')
        #if settings.save_tmps :
        #    tocollect=sorted([ kg.opj(settings.tmpprefix,fp) for fp in os.listdir(settings.tmpprefix) 
        #                        if fp.startswith(settings.jobprefix+'_boot') and fp.endswith('.pt')])
        #    tmps=list()
        #    for tc in tocollect : 
        #        tmps.append(torch.load(tc))
        #    torch.save(tmps,opj(settings.output_directory,'boot_individuals.pt'))

        torch.save(settings,opj(settings.output_directory,'settings.pt'))

        torch.save(lg.X.cpu(),opj(settings.output_directory,'X.pt'))

        np.savez(
            opj(settings.output_directory,'arrays.npz'),
            y=lg.y.values,
            feats=lg.features(),
            featuretypes=lg.featuretypes(),
            ylabels=np.array(lg.y.index),
            n=torch.tensor(len(lg.patients)),
            ofinterest_full=ofinterest,
        )

        msg('Scoring spoofed hierarchies...')
        waitbar=tqdm(total=2*settings.n_spoofs)
        waitbar.display()
        for pref,tf in zip(['fake','wild'],[True,False]) : 
            for x in range(settings.n_spoofs) : 
                torch.cuda.empty_cache()
                thisspdir=opj(settings.output_directory,pref+'_'+str(x))
                os.makedirs(thisspdir,exist_ok=True)
                sh=spoof(lg.hierarchy,preserve_rowsums=True)
                lgspoof=LARGeSSE_G(omics=lg.omics,signatures=lg.signatures,lengths=lg.lengths,timings=lg.timings)
                lgspoof._assign_hierarchy(sh,system_limit_upper=settings.hsystem_limit_upper,system_limit_lower=settings.hsystem_limit_lower)
                sp_IH=lgspoof.build_IH(inplace=True,weight=False)
                sp_J=lgspoof.build_J(inplace=True,correlation_p=settings.correlation_p)
                X=lgspoof.build_X(normalize=True,inplace=True)
                
                spoof_kws=compile_kwargs(lgspoof,settings,which='spoof')
                spoof_kws.update(dict(savefile=opj(thisspdir,'logger.pt')))
                torch.save(X,opj(thisspdir,'spX.pt'))
                #spooffeaturetypes=lgspoof.featuretypes()
                #spoofsigmask=( spooffeaturetypes != 'gene')  & ( spooffeaturetypes != 'system')
                
                #RESUME -- fix this, add a spoof setting, whatever, so that it doesn't overwrite main
                # then, do a final prep run before deploying in batch
                splog=run_model(
                                lgspoof,
                                strengths=bir.strengths,
                                **spoof_kws,
                                )
                
                waitbar.update()
        waitbar.close()


if __name__ == '__main__' : 
    if len(sys.argv) > 1 : 
        runjson=sys.argv[1]
        with open(runjson) as f : 
            settings=Settings(**json.load(f))

        main_script_process(settings) ; 
    else : 
        pass

#   if patient_indicesis not None : 
#       outside_pat_indices=lg._t_pat_indices[ ~torch.isin(lg._t_pat_indices,patient_indices) ]
#       itx,iky=lg.build_xy_from_subset(gene_indices=None,patient_indices=patient_indices,recalc_J=recalc_J,renormalize_X=recalc_J)
#       ni=torch.tensor(len(patient_indices),device=DEVICE,dtype=torch.int)
#       if not score_by_train : 
#           otx,oky=lg.build_xy_from_subset(gene_indices=None,patient_indices=outside_pat_indices)
#           no=torch.tensor(len(outside_pat_indices),device=DEVICE,dtype=torch.int)
#   else : 
#       itx=tX
#       otx=tX
#       iky=torch.tensor(lg.y.values,device=DEVICE,dtype=torch.float)
#       ni=torch.tensor(lg.omics.shape[0],device=DEVICE,dtype=torch.int)
#       if not score_by_train : 
#           oky=torch.tensor(lg.y.values,device=DEVICE,dtype=torch.float)
#           no=torch.tensor(lg.omics.shape[0],device=DEVICE,dtype=torch.int)
