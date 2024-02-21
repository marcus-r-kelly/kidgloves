import numpy as np
from scipy.special import expit
import pandas as pd
import pickle
import os
from collections import namedtuple
import largesse_g
from largesse_g import OptimizerResult,BurnInResult,Settings
from types import SimpleNamespace
from importlib import reload
import kidgloves as kg
import torch
from concurrent import futures
from functools import reduce
from tqdm.auto import tqdm
opj=os.path.join

def deepunpickle(fn) : 
    pdata=list()
    with open(fn,'rb') as f : 
        while True : 
            try : 
                pdata.append(pickle.load(f))
            except EOFError as e : break
            except Exception as e : 
                print(type(e),e)
                break ;
    return pdata

def qp(filename) : 
    with open(filename,'rb') as f : 
        return pickle.load(f)
    
def doubleup(array): 
    return np.stack([array,array],axis=1).reshape(-1,1).ravel()

rakws=[ 
        'main',
        'matched_null',
        'burnin',
        'boot',
        'settings',
        'X',
        'boot_X',
        'y',
        'feats',
        'featuretypes',
        'ylabels',
        'n',
        'ofinterest',
        'path',
        'hier',
        ]

def _unpack_arrays(npzpath) :
    npz=np.load(npzpath,allow_pickle=True)
    return dict(X=npz['X'],
                y=npz['y'],
                feats=npz['feats'],
                featuretypes=npz['featuretypes'],
                ylabels=npz['ylabels'],
                n=npz['n'],
                ofinterest=npz.get('ofinterest_full'))

def taic(w,i,X,y,n) : 
    eps=torch.finfo(w.dtype).eps
    ak=( w > eps).sum()
    phat=torch.clip(
            torch.special.expit(
                torch.matmul(X,w)+i
            ),
            eps,
            1-eps,
    )

    if not torch.is_tensor(n) : 
        n=torch.tensor(n)

    tbinom=torch.distributions.Binomial(
        total_count=n,
        probs=phat)

    ll=tbinom.log_prob(y).sum()

    aicn=X.shape[0]

    criterion=2*ak-2*ll+2*ak*(ak+1)/max(aicn-1-ak,aicn/ak)

    return largesse_g.AICResult(
            criterion.item(),
            ak.item(),
            ll.item(),
            n.item(),
            )

def modgen(ra) : 
    paths=[ 'main','null' ]+[ opj(ra.path,subf) for subf in os.listdir(ra.path) if os.path.isdir(opj(ra.path,subf)) ] 

    for p in paths : 
        match p : 
            case 'xval' :
                pass
            case 'main' : 
                yield ('main',ra.main,ra.X,ra.y,ra.n)
            case 'null' : 
                yield ('null',ra.matched_null,largesse_g.mask_sparse_columns(ra.X,torch.tensor(ra.featuretypes != 'system',dtype=torch.bool)),ra.y,ra.n)
            case _ : 
                yield (p,torch.load(opj(ra.path,p,'logger.pt')),torch.load(opj(ra.path,p,'spX.pt'),map_location=torch.device('cpu')),ra.y,ra.n)

def score_optres_and_x(p,optres,X,y,n) : 

    if not torch.is_tensor(y)  : 
        if hasattr(y,'values') : 
            #quacks like a pandas.Series
            y=torch.tensor(y.values)
        else:
            y=torch.tensor(y)
    bw,bi=best_of_or(optres)
    return taic(torch.tensor(bw),torch.tensor(bi),X,y,n)

def lowest_of_path(path) : 
    return [ d for d in path.split(os.sep) if len(d) > 0 ][-1]

def score_ensemble(ra) :

    flags=list()
    aicreses=list()
    gutssizes=list()
    n_sig_genes=list()
    n_sig_systems=list()
    n_sig_signatures=list()

    n_genes_in_matrix=(ra.featuretypes == 'gene').sum()
    n_signatures_in_matrix=(( ra.featuretypes != 'system' ) & (ra.featuretypes != 'gene') ).sum()


    for guts in modgen(ra) : 
        flags.append(lowest_of_path(guts[0]).split('_')[0])
        nsystems_this_h=guts[2].shape[1]
        gutssizes.append(nsystems_this_h)

        bw,bi=best_of_or(guts[1])
        bwoi=np.argwhere(bw > 0).ravel()

        nsgenes_this=(bwoi < n_genes_in_matrix ).sum()
        nssignatures_this= (bwoi >= (guts[2].shape[1]-n_signatures_in_matrix) ).sum()

        n_sig_genes.append( nsgenes_this )
        n_sig_signatures.append(nssignatures_this)
        n_sig_systems.append(len(bwoi) - nsgenes_this - nssignatures_this)

        aicreses.append(score_optres_and_x(*guts))

    fr=pd.DataFrame(aicreses)
    fr=fr.assign(model_type=flags)
    fr=fr.assign(parent_model=lowest_of_path(ra.path),gutssize=gutssizes)
    fr=fr.assign(n_systems = fr.gutssize  - fr.query('model_type == "null"').iloc[0].gutssize,
                 n_sig_genes=n_sig_genes,
                 n_sig_systems=n_sig_systems,
                 n_sig_signatures=n_sig_signatures,
                 )
    return fr

def best_of_or(optres) : 
    bwi=optres.loss.argmin()
    wt=optres.weight[bwi]
    i=optres.intercept[bwi]
    return wt,i






class RunAnalysis(object) : 
    
    def __init__(self,**kwargs) : 
        super(RunAnalysis,self).__init__()

        for rk in rakws : 
            if kwargs.get(rk) is not None : 
                self.__setattr__(rk,kwargs[rk])
            else : 
                print('Argument',rk,'not found.')

        if not hasattr(self,'ofinterest') : 
            self._fix_ofinterest()

        with open(self.settings.hierarchy_path,'rb') as f :
            self.hier=pickle.load(f)

    def _fix_ofinterest(self) : 
        oifromburnin    =   self.burnin.ofinterest
        bw,bi           =   self.best_main()
        oifrommain      =   np.argwhere( bw > 0 )
        oifrommn        =   np.argwhere(self.matched_null.weight[self.matched_null.loss.argmin()] > 0)

        self.ofinterest=reduce(np.union1d,[oifromburnin,oifrommain,oifrommn])
        return self.ofinterest
        

    def from_dir(path) : 

        mr=torch.load(kg.opj(path,'main.pt'))
        mn=torch.load(kg.opj(path,'matched_null.pt'))
        bir=torch.load(kg.opj(path,'burnin.pt'))
        boot=torch.load(kg.opj(path,'boot.pt'))
        settings=torch.load(kg.opj(path,'settings.pt'))
        npz=np.load(kg.opj(path,'arrays.npz'),allow_pickle=True)
        X=torch.load(kg.opj(path,'X.pt'))
        with open(settings.hierarchy_path,'rb') as f : 
            hier=pickle.load(f)

        wx=largesse_g.mask_sparse_columns(
                X,
                torch.isin(
                    torch.arange(X.shape[1]),
                    torch.tensor(bir.ofinterest)
                )
            )

        arrays=dict(
                y=torch.tensor(npz['y'],device=largesse_g.CPU),
                feats=npz['feats'],
                featuretypes=npz['featuretypes'],
                ylabels=npz['ylabels'],
                n=npz['n'],
                ofinterest=npz['ofinterest_full'],
                )

        return RunAnalysis(main=mr,
                    matched_null=mn,
                    burnin=bir,
                    boot=boot,
                    settings=settings,
                    X=X,
                    boot_X=wx,
                    path=path,
                    hier=hier,
                    **arrays
                    )

    def predict(self,weights,intercept) : 
        return torch.special.expit(self.predict_odds(weights,intercept))

    def predict_odds(self,weights,intercept) : 
        if len(self.weights) < self.X.shape[1] : 
            return torch.matmul(self.wx,weights)+intercept
        else: 
            return torch.matmul(self.X,weights)+intercept


    def best_main(self) : 
        return best_of_or(self.main)  

    def get_feature_stats(self) : 

        bw,bi=self.best_main()
        bwargs=np.argwhere( bw > 0).ravel()

        featindices=np.arange(self.X.shape[1])

        fsfr=pd.DataFrame(index=self.feats)
        fsfr=fsfr.assign(
                featuretype=self.featuretypes,
                feature_indices=featindices,
                of_interest=np.isin(featindices,self.ofinterest),
                in_burnin=np.isin(featindices,self.burnin.ofinterest),
                in_final_model=np.isin(featindices,bwargs),
                in_mn_model=np.isin(featindices,np.argwhere(self.matched_null.weight[self.matched_null.loss.argmin()] > 0)).ravel(),
                nmembers=(self.X.to_dense() > 0).sum(axis=0).numpy(),
                )

        stability=((self.boot >= bw[self.ofinterest]*0.5 ) & (self.boot <= bw[self.ofinterest]*2)).sum(axis=0)/self.boot.shape[0]

        sp_fsfr=pd.DataFrame(index=fsfr.index[self.ofinterest])
        sp_fsfr=sp_fsfr.assign(
                    value=bw[self.ofinterest],
                    frac_nz=(self.boot > 0).sum(axis=0)/self.boot.shape[0],
                    average=self.boot.mean(axis=0),
                    median=np.median(self.boot,axis=0),
                    stability=stability)

        fsfr=fsfr.join(sp_fsfr,how='left').fillna(0)


        self.feature_stats=fsfr
        return self.feature_stats

    def generate_cytoscape_frames(self) : 

        bw,bi=self.best_main()

        cytox=largesse_g.mask_sparse_columns(self.X,torch.tensor(bw >0,dtype=torch.bool))
        tbw=torch.tensor(bw[ bw > 0 ]).to_sparse_coo()
        bwde=largesse_g.sptops.diag_embed(tbw)
        thrustrength=torch.matmul(cytox,bwde)

        largesse_g.kg._get_geneinfo()

        r2ev=dict(zip(range(self.y.shape[0]),self.ylabels))
        c2feat=dict(zip(range(cytox.shape[1]),
                        [ largesse_g.kg._e2s.get(f,f) for f in self.feats[ bw > 0 ] ]))

        npi=thrustrength.coalesce().indices().numpy()
        npv=thrustrength.coalesce().values().numpy()


        edgeframe=pd.DataFrame(data=npi.transpose(),columns=['target_index','source_index']).astype(int)
        edgeframe=edgeframe.assign(
                        target_name=edgeframe.target_index.apply(r2ev.get),
                        source_name=edgeframe.source_index.apply(lambda x : c2feat.get(x,x)),
                        strength=npv,
                        edge_type='model_relation',
        )


        hier_edges=pd.DataFrame(data=get_hierarchy_topology(self.hier),
                                columns=['source_name','target_name'])
        hier_edges['edge_type']='hierarchy_relation'
        hier_edges['strength']=0.5

        edgeframe=pd.concat([edgeframe,hier_edges],axis=0).reset_index(drop=True)


        nodeframe_events=pd.DataFrame(data=self.y.numpy().reshape(-1,1),columns=['frequency'])
        nodeframe_events['node_name']=self.ylabels
        nodeframe_events['featuretype']=nodeframe_events.node_name.apply(lambda x: x.split('_')[-1])

        if not hasattr(self,'feature_stats') or self.feature_stats is None: 
            self.get_feature_stats()
        fs=self.feature_stats.reset_index(drop=False)
        fs=fs.rename(columns={'index' : 'node_name'})

        nodeframe=pd.concat([ nodeframe_events,fs ],axis=0).reset_index(drop=True)

        return edgeframe,nodeframe

    def assemble_xval_summary(self,omics) : 

        xvaldir=kg.opj(self.settings.output_directory,'xval')
        splits=sorted([ fn for fn  in os.listdir(xvaldir) if fn.startswith('splits_') ])
        X=self.X

        nullmask=(self.featuretypes != 'system')
        Xn=largesse_g.mask_sparse_columns(X,torch.tensor(nullmask))
        xval_summary_data=list()

        for sp in splits: 
            spno=sp.split('.')[0].split('_')[1]
            tr,te=torch.load(os.path.join(xvaldir,sp))
            orfull=torch.load(os.path.join(xvaldir,'xval_full_'+spno+'.pt'))
            ornull=torch.load(os.path.join(xvaldir,'xval_null_'+spno+'.pt'))
            ytr=omics.reindex(tr).sum().reindex(self.ylabels).fillna(0)
            yte=omics.reindex(te).sum().reindex(self.ylabels).fillna(0)
            
            stats_full_tr=score_optres_and_x(xvaldir,orfull,X,ytr,len(tr))
            stats_full_te=score_optres_and_x(xvaldir,orfull,X,yte,len(te))
            stats_null_tr=score_optres_and_x(xvaldir,ornull,Xn,ytr,len(tr))
            stats_null_te=score_optres_and_x(xvaldir,ornull,Xn,yte,len(te))
            
            results=[ stats_full_tr, stats_full_te, stats_null_tr, stats_null_te, ]
            modelclasses=['full','full','null','null']
            patientgroups=['train','test','train','test']
            
            subf=pd.DataFrame(results)
            subf['modelclasses']=modelclasses
            subf['patient_groups']=patientgroups
            subf['fold']=spno
            
            xval_summary_data.append(subf)

        xvdf=pd.concat(xval_summary_data).reset_index(drop=True)
        xvdfp=xvdf.pivot(index=['fold','patient_groups'],columns='modelclasses',values='criterion')
        xvdf=xvdf.merge((xvdfp.full-xvdfp.null).rename('dAIC').reset_index(),on=['fold','patient_groups'],how='left')    

        self.xvdf=xvdf
        return xvdf


def is_possible_child(hier,parent,child) :
    return (len(hier[child] - hier[parent]) == 0 )

def get_hierarchy_topology(hier) : 
    from tqdm.auto import tqdm

    ssk=sorted(hier.keys(),key=lambda x : len(hier[x]),reverse=True)
    child2parents=dict()
    parents2children=dict()

    for x in tqdm(range(len(ssk)),total=len(ssk)) : 
        for y in range(x+1,len(ssk)) : 
            if is_possible_child(hier,ssk[x],ssk[y]) : 
                # this smaller system is a child of the larger
                if any([ is_possible_child(hier,c,ssk[y]) for c in parents2children.get(ssk[x],set()) ]) : 
                    continue
                else : 
                    parents2children.update({ ssk[x] : parents2children.get(ssk[x],set()) | { ssk[y], } })
                    child2parents.update({ ssk[y] : child2parents.get(ssk[y],set()) | { ssk[x], } })

    edgedatas=list()
    for p in parents2children :
        for c in parents2children[p] : 
            edgedatas.append((p,c))

    return edgedatas


cell_block_width=0.3
cell_block_height=0.5
cell_margin_factor=0.1
cell_padding_width=0.05
cell_padding_height=0.05
full_patch_height=cell_block_height-cell_padding_height*cell_block_height
full_patch_width=cell_block_width-cell_padding_height*cell_block_width
small_patch_height=0.2
small_patch_upshift=(full_patch_height-small_patch_height)/2

cell_bg_color='#dddddd'

beautify_model_name={
        'reactome_prelim' : 'reactome' , 
        'go_prelim' : 'GO CC' ,
        'kamber' : 'W&K',
        'kuenzi_census' : 'Kuenzi Census',
        'kuenzi_consensus' : 'Kuenzi Consensus',
        'pcawg' : 'PCAWG',
        'webster' : 'WEBSTER' ,
        'H_IAS_clixo_hidef_Nov17.edges' : 'NESTv0',
        'corum' : 'CORUM',}


def invert_hue(ctf) : 
    import colorsys
    h,s,v=colorsys.rgb_to_hsv(*ctf)
    newh=h+0.5 % 1.0
    return colorsys.hsv_to_rgb(newh,s,v)

def ct2h(ctup) : 
    return '#'+''.join([ hex(int(ct*255))[2:] for ct in ctup[:3] ]) 

def h2cti(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16) for x in range(1,4) ])

def h2ctf(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16)/255 for x in range(1,4) ])

class palette() : 
    def __init__(self,**kwargs) : 
        for k in kwargs : 
            self.__setattr__(k,kwargs[k])

spots=palette(      
    vermillion="#E91E25",
    navy="#2E4A91",
    yellow="#E4D120",
    lime="#57A247",
    orange="#D25427",
    pink="#D55B92",
    teal="#2A7DAE",
    forest="#176935",
    purple="#882885",
    brown="#966436",
)

routes=palette(
    filminspace="#5A95B6",
    resource1="#CB2027",
    materialidad="#288F4E",
    mustard="#B89C32",
    circle="#472E80",
    comisariado="#96B85F",
    investigacion="#544A40",
    creacion="#95C4B8",
    poetika="#AF8B52",
)


fuscolor=spots.orange
dncolor=spots.navy
mutcolor=spots.lime
upcolor=spots.vermillion
white='#ffffff'
offwhite='#ffeedd'

from matplotlib.colors import LinearSegmentedColormap
tealcm=LinearSegmentedColormap.from_list('sub_indel',[(1.0,1.0,1.0),h2ctf(spots.teal)])
tealdcm=LinearSegmentedColormap.from_list('sub_indel_d',[invert_hue(h2ctf(spots.teal)),(1.0,1.0,1.0),h2ctf(spots.teal)])

purpcm=LinearSegmentedColormap.from_list('cna',[(1.0,1.0,1.0),h2ctf(spots.purple)])
purpdcm=LinearSegmentedColormap.from_list('cna_d',[invert_hue(h2ctf(spots.purple)),(1.0,1.0,1.0),h2ctf(spots.purple)])

udcm=LinearSegmentedColormap.from_list('arm',[h2ctf(spots.navy),(1.0,1.0,1.0),h2ctf(spots.vermillion)])

upcm=LinearSegmentedColormap.from_list('up_over_cohort',[(1.0,1.0,1.0),h2ctf(upcolor)])
dncm=LinearSegmentedColormap.from_list('dn_over_cohort',[(1.0,1.0,1.0),h2ctf(dncolor)])
mutcm=LinearSegmentedColormap.from_list('mut_over_cohort',[(1.0,1.0,1.0),h2ctf(mutcolor)])
fuscm=LinearSegmentedColormap.from_list('fus_over_cohort',[(1.0,1.0,1.0),h2ctf(fuscolor)])

def pivot_event_classes(ra) : 

    if kg._e2s is None : 
        kg.get_geneinfo()

    e2s=kg._e2s

    omicscounts=pd.DataFrame(data=[ra.ylabels,ra.y],index=['event_name','event_counts']).transpose()
    omicscounts['eid']=omicscounts.event_name.apply(lambda x : x.split('_')[0])
    omicscounts['event_class']=omicscounts.event_name.apply(lambda x : x.split('_')[1])
    omicscounts['symbol']=omicscounts.eid.apply(e2s.get)

    ocpiv=omicscounts.pivot_table(index='symbol',columns='event_class',values='event_counts').fillna(0)
    occounts=ocpiv.sum(axis=1)
    occounts=occounts[occounts.gt(1)]
    occounts=occounts.sort_values(ascending=True)
    ocpiv=ocpiv.reindex(occounts.index)

    return ocpiv

def onco_hbar(ocpiv,figsize=(0.5,3)) : 
    from matplotlib import pyplot as plt
    f=plt.figure(figsize=figsize)

    asmal=f.add_subplot(111)
    whys=np.arange(ocpiv.shape[0])
    asmal.barh(y=whys,width=ocpiv['fus'].values,color=fuscolor,label='Fusions',ec=white)
    asmal.barh(y=whys,width=ocpiv['dn'].values,left=ocpiv['fus'].values,color=dncolor,label='Copy loss',ec=white)
    asmal.barh(y=whys,width=ocpiv['up'].values,left=ocpiv['fus'].values+ocpiv['dn'].values,color=upcolor,label='Copy gain',ec=white)
    asmal.barh(y=whys,width=ocpiv['mut'].values,left=ocpiv['fus'].values+ocpiv['dn'].values+ocpiv['up'].values,color=mutcolor,label='Mutation',ec=white)
    asmal.set_yticks(whys); 
    asmal.set_yticklabels(ocpiv.index,fontdict=dict(family='arial',style='italic',size=6)) ;
    asmal.spines['right'].set_visible(True)
    asmal.legend(loc='lower left',bbox_to_anchor=[0.5,0.05,0.1,0.01],fontsize=6,labelspacing=0.02,handlelength=0.5,borderpad=0.3,framealpha=1)

    asmalyt=asmal.get_yticks()
    intertickdistance=asmalyt[1]-asmalyt[0]
    asmal.set_ylim([asmalyt.min()-0.5*intertickdistance,asmalyt.max()+0.5*intertickdistance])

    asmal.spines['right'].set_visible(False)

    return f


def onco_hbar_split(ocpiv,figsize=(0.5,3),transition=15) : 
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    f=plt.figure(figsize=figsize)

    gs=gridspec.GridSpec(nrows=1,ncols=2,figure=f,wspace=0)

    abigg=f.add_subplot(gs[0,1])
    whys=np.arange(ocpiv.shape[0])
    abigg.barh(y=whys,width=ocpiv['fus'].values,color=fuscolor,label='Fusions',ec=white)
    abigg.barh(y=whys,width=ocpiv['dn'].values,left=ocpiv['fus'].values,color=dncolor,label='Copy loss',ec=white)
    abigg.barh(y=whys,width=ocpiv['up'].values,left=ocpiv['fus'].values+ocpiv['dn'].values,color=upcolor,label='Copy gain',ec=white)
    abigg.barh(y=whys,width=ocpiv['mut'].values,left=ocpiv['fus'].values+ocpiv['dn'].values+ocpiv['up'].values,color=mutcolor,label='Mutation',ec=white)
    abigg.set_yticks([]); 
    abigg.set_xlim([transition,150])
    abigg.set_xticks([transition,150]) ;
    asmal=f.add_subplot(gs[0,0])

    whys=np.arange(ocpiv.shape[0])
    asmal.barh(y=whys,width=ocpiv['fus'].values,color=fuscolor,label='Fusions',ec=white)
    asmal.barh(y=whys,width=ocpiv['dn'].values,left=ocpiv['fus'].values,color=dncolor,label='Copy loss',ec=white)
    asmal.barh(y=whys,width=ocpiv['up'].values,left=ocpiv['fus'].values+ocpiv['dn'].values,color=upcolor,label='Copy gain',ec=white)
    asmal.barh(y=whys,width=ocpiv['mut'].values,left=ocpiv['fus'].values+ocpiv['dn'].values+ocpiv['up'].values,color=mutcolor,label='Mutation',ec=white)
    asmal.set_yticks(whys); 
    asmal.set_yticklabels(ocpiv.index,fontdict=dict(family='arial',style='italic',size=6)) ;
    asmal.set_xlim([0,transition])
    asmal.spines['right'].set_visible(True)
    asmal.legend(loc='lower left',bbox_to_anchor=[1.6,0.05,0.1,0.01],fontsize=6,labelspacing=0.02,handlelength=0.5,borderpad=0.3,framealpha=1)
    if transition > 10 : 
        asmal.set_xticks([0,(transition//5-2)*5])
    else : 
        asmal.set_xticks([0])

    asmalyt=asmal.get_yticks()
    intertickdistance=asmalyt[1]-asmalyt[0]
    asmal.set_ylim([asmalyt.min()-0.5*intertickdistance,asmalyt.max()+0.5*intertickdistance])
    abigg.set_ylim(asmal.get_ylim())

    asmal.spines['right'].set_visible(False)

    return f

# you get the argument for the below function by running postrunutils.predsum on all the slices and concatenating them
#def AIC_waterfall_plotter(aicthreshdf) : 

    


#a.set_xscale('symlog')




def stripsuf(s) : 
    return s.split('_')[0]

