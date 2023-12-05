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
    default=50,
    help='Number of cohort bootstraps to conduct')

parser.add_argument('--debug',
    action='store_true',
    help='Turn on debugging messages.')


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

rulespath=ns.rules
hpath=ns.hierarchy
hname='.'.join(hpath.split(os.path.sep)[-1].split('.')[:-1])
outpath=ns.outpath
DEBUG=ns.debug
NREPEATS=ns.n_repeats


#ogenes={ g for v in hierarchy.values() for g in v } # commented out 230920

#~~~~~~~~Read signatures and patient omics~~~~~~~~~~~~~~~~~
msg(f"Reading in signatures from folder {rulespath}...",end='')
#lt=qunpickle(kg.opj(rules,'logittransformer.pickle'))
from sklearn.preprocessing import MaxAbsScaler

#TODO
# this needs to be made more generalizable, esp for other cohorts
# needs to be reworked into cohort_dump type earlier scripts
armdata=pd.read_csv('/cellar/users/mrkelly/Data/largesse_paper/notebooks/positive_bigJ_comparison/arms_mbnmf.csv',index_col=0)

msig=pd.read_csv(kg.opj(rulespath,'mutation_signatures.csv'),index_col=0)
msig=msig[ msig.columns[~msig.columns.str.startswith('arm')]]
from sklearn.preprocessing import MaxAbsScaler
msigscale=pd.DataFrame(
    data=MaxAbsScaler().fit_transform(
        pd.concat([msig,armdata],axis=1)),
                       index=msig.index,
                       columns=list(msig.columns)+list(armdata.columns))


msg('Done.')

msg(f"Reading in omics from LUAD COHORT...",end='')
lt=qunpickle(kg.opj(rulespath,'logittransformer.pickle'))
omics=lt.training_data


#~~~~~~~~Sync up indices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
msg("Syncing up data...",end='')
common_patients=np.intersect1d(omics.index,msigscale.index)
msigscale=msigscale.reindex(common_patients)
omics=omics.reindex(common_patients)
ogenes={ c.split('_')[0] for c in omics.columns } # uncommented 230920
aslo=np.array(sorted(list(ogenes)))
msg('Done.')

#~~~~~~~~Read in hierarchy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
msg(f"Reading in hierarchy from {hpath}...",end='')
hierarchy=qunpickle(hpath)
system_size_filter=2000
hierarchy={ h : hierarchy[h] for h in hierarchy.keys() if len(hierarchy[h]) < 2000  and len(hierarchy[h]&ogenes) > 1 }
hn=hpath.split(os.sep)[-1].split('.')[0]
msg('Done.')

msg("masking to hierarchy...",end='')
nmo_h=kg.mask_nest_systems_from_omics(hierarchy,omics)
nma_h=kg.arrayify_nest_mask(nmo_h,omics.columns)
nma_h=nma_h.numpy()
nkeys=sorted(nmo_h.keys())
msg('Done.')

import multiprocessing as mp
import warnings
from tqdm.auto import tqdm

msg("Calculcating event:signature relationships...",end='')
import warnings
def mycw_pearson(x) : 
    with warnings.catch_warnings() :
        warnings.simplefilter('ignore')
        return msigscale.corrwith(omics[x],method='pearson')

import multiprocessing as mp
with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
    cw=pd.concat(
        [ x for x in p.imap(mycw_pearson,omics.columns) ],
    axis=1)
msg("Done.")

cw=cw.transpose()

msg("Assembling features...",end='')
bigI=np.array([ ( c.split('_')[0] == aslo ) for c in omics.columns])
bigX=np.concatenate([bigI,nma_h,cw.values],axis=1)
bigxcols=np.array(list(aslo) + nkeys + list(cw.columns))
msg("Done.")

#origy=np.log(omics.sum(axis=0)+1)#  230920
origy=(omics.sum(axis=0)+1)/omics.shape[0] #231005
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

from sklearn.linear_model import LassoLarsIC
outdata=list()

msg('Preparing pytorch...',end='')
import torch
if torch.cuda.is_available() : 
    msg('with a GPU...',end='')
    DEVICE=torch.device('cuda:0') ;
else : 
    msg('with CPUS...',end='')
    DEVICE=torch.device('cpu') ;

yactual=torch.tensor(np.log(omics.sum(axis=0)+1),device=DEVICE)

msg('coefficient guesses ...',end='')
with warnings.catch_warnings() :
    warnings.simplefilter('ignore')
    premodel=LassoLarsIC(noise_variance=0.1,fit_intercept=False,positive=True)
    premodel.fit(bigX,yactual.cpu().numpy())
    centers=torch.tensor(premodel.coef_,device=DEVICE,dtype=torch.float64)

#bigT=torch.concat((yactual.reshape(-1,1),torch.tensor(bigX,device=DEVICE)),dim=1)
#cc=bigT.transpose(0,1).corrcoef()
#centers=cc[0,1:].reshape(-1,1)
#centers=torch.clip(centers.to(DEVICE),0,torch.inf).nan_to_num(0)

class LARGeSSE_LASSO(torch.nn.Module) : 
    def __init__(self, nsigs,device=torch.device('cpu'),init_method='zeros',centers=None):
        super(LARGeSSE_LASSO, self).__init__()
        
        self.relu = torch.nn.ReLU()
        if init_method == 'zeros' : 
            self.weights = torch.nn.Parameter(torch.empty(size=(nsigs,1),device=device,dtype=torch.float64))
            torch.nn.init.zeros_(self.weights)
        elif init_method == 'correlates' : 
            self.weights = torch.nn.Parameter(torch.relu_(centers + torch.normal(0,0.2,centers.shape,device=device,dtype=torch.float64)))
        else : 
            self.weights = torch.nn.Parameter(torch.empty(size=(nsigs,1),device=device,dtype=torch.float64))
            torch.nn.init.uniform_(self.weights,a=0,b=1)
        
    def forward(self,J,return_weights=False) :
        corrected_weight=self.relu(self.weights) ;
        out=torch.matmul(J,corrected_weight)
        if not return_weights : 
            return out
        else : 
            return out,corrected_weight
        
class AIC_Loss(torch.nn.Module): 
    
    def __init__(self,nv,device=torch.device('cpu')) : 
        super(AIC_Loss,self).__init__()
        self.mseloss=torch.nn.MSELoss()
        self.device=device
        self.nv=torch.tensor(nv,dtype=torch.float64,device=self.device)
        
    def forward(self,outputs,targets,Jweights) : 
        """
        https://en.wikipedia.org/wiki/Regularized_least_squares#Lasso_regression
        """
        
        n=outputs.shape[0]
        sse=self.mseloss(outputs.ravel(),targets)*n
        llterm=-1*n*torch.log(2*np.pi*self.nv)-sse/self.nv
        kay=(Jweights != 0).sum()
        
        return 2*kay-llterm
    
gen_optimizer=lambda model: torch.optim.AdamW(model.parameters(),lr=0.1,weight_decay=1e-5)

# parameters of the optimization chain
NCHAINS=50
COOLDOWN=5
NEPOCHS=301
SAMPLERATE=11
SAMPLINGTIMES=[ e for e in range(NEPOCHS) if (e > COOLDOWN) and (e % SAMPLERATE ==0) ]
CHAINS_BEFORE_RESAMPLE=7
CHAINS_TO_SAMPLE=5
from torch.optim.lr_scheduler import CyclicLR,ReduceLROnPlateau
import time

NSAMPLINGS=len(SAMPLINGTIMES)
msg('Done.')

def dmsg(*args,**kwargs) : 
    if not DEBUG : return
    msg(*args,**kwargs)

def run_metachain(thisy) : 
    starttime=time.time()
    tdelta=0.0

    # logging tensors
    llog=torch.empty(size=(NCHAINS,NSAMPLINGS),requires_grad=False,device=DEVICE,dtype=torch.float64)
    wlog=torch.empty(size=(NCHAINS,bigX.shape[1]),requires_grad=False,device=DEVICE,dtype=torch.float64)
    ttbx=torch.tensor(bigX,dtype=torch.float64,device=DEVICE)
    colindices=torch.arange(0,wlog.shape[1],device=DEVICE).reshape(-1,1)
    mnweights=torch.ones((CHAINS_TO_SAMPLE,wlog.shape[1]),device=DEVICE).transpose(0,1)
    bestlosses=torch.empty((NCHAINS,),device=DEVICE,dtype=torch.float64)

    for c in range(NCHAINS) : 
        start=time.time()
        if c < CHAINS_BEFORE_RESAMPLE : 
            dmsg("Beginning chain {}, initating from correlation coefficients. Last chain runtime : {: >4.2}     ".format(c,tdelta))
            model=LARGeSSE_LASSO(bigX.shape[1],init_method='correlates',centers=centers,device=DEVICE)
        else : 
            dmsg("Beginning chain {}, initating from previous solutions. Last chain runtime : {: >4.2}           ".format(c,tdelta))
            revisitable=bestlosses[:c].argsort()[:CHAINS_TO_SAMPLE]
            revisitablelosses=bestlosses[revisitable]
            revisited=wlog[revisitable]
            resampled=torch.multinomial(mnweights,1,replacement=False)
            seedcoefs=revisited[resampled,colindices].ravel()
            model=LARGeSSE_LASSO(bigX.shape[1],init_method='correlates',centers=seedcoefs,device=DEVICE)

        sched=CyclicLR(gen_optimizer(model),base_lr=1e-1,max_lr=1,step_size_up=20,cycle_momentum=False,mode='triangular2')
        lossfitter=AIC_Loss(nv=0.1,device=DEVICE)
        bestloss=np.inf
        indexofbestloss=-1
        samplestaken=0

        for epoch in range(NEPOCHS) :

            sched.optimizer.zero_grad()
            outputs, corrected_weights = model(ttbx, return_weights=True)

            loss=lossfitter(
                outputs,
                thisy,
                corrected_weights
            )
            
            do_reporting=(epoch % SAMPLERATE == 0)
            
            if epoch in SAMPLINGTIMES : 
                if samplestaken > 0 : 
                    isbestloss=loss < llog[c,:samplestaken+1].min()
                    if isbestloss : 
                        do_reporting=True
                        bestloss=loss
                        indexofbestloss=epoch  
                        wlog[c,:]=corrected_weights.detach().ravel()
                else: 
                    indexofbestloss=epoch  
                    bestloss=loss
                    wlog[c,:]=corrected_weights.detach().ravel()

                llog[c,samplestaken]=loss.detach().item()
                    
                samplestaken += 1

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            sched.optimizer.step()
            sched.step(loss)
            
            if  do_reporting : 
                dmsg("Epoch {:0>4} of {:0>4}. Best loss {:0>4.2e} at epoch {:0>4}".format(epoch,NEPOCHS,bestloss,indexofbestloss),end='\r')
            
        if c >= CHAINS_BEFORE_RESAMPLE :
            if loss < revisitablelosses.min() : 
                dmsg("\nOn chain {:0>3}, new optimal loss {:0>4.2e} found.".format(c,bestloss))
            elif loss < revisitablelosses.max() : 
                dmsg("\nOn chain {:0>3}, new top-{: <3} loss {:0>4.2e} found.".format(c,CHAINS_TO_SAMPLE,bestloss))
                    
        tdelta=time.time()-start

    optima=llog.min(axis=1).values
    optimatimes=llog.argmin(axis=1)
    coords_of_best=(optima.argmin(),optimatimes[optima.argmin()])
    bestloss=llog[coords_of_best]
    bestcoefs=wlog[coords_of_best[0]]

    clipperguard=np.arange(0,2.01,0.05)
    clipperaic=bestloss
    besthaircut=bestcoefs
    for cg in clipperguard : 
        clipcoef=bestcoefs.clone().detach()
        clipcoef[ clipcoef < cg ]=0.0
        nremainingparams=(clipcoef != 0).sum()
        loss_this_haircut=lossfitter(torch.matmul(ttbx,clipcoef),thisy,clipcoef).item()
        if loss_this_haircut < clipperaic : 
            clipperaic=loss_this_haircut
            besthaircut=clipcoef

        dmsg('Below {: <4.2e}: {:0>5.4e} , {} params.'.format(cg,loss_this_haircut,nremainingparams))
    #optimatimes=npl.argmin(axis=1)

    return clipperaic,besthaircut.cpu().detach().numpy()


#RESUME
#~~~~~~~~Below is pasted from notebook~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        
    #sched=CyclicLR(gen_optimizer(model),base_lr=1e-3,max_lr=1,step_size_up=20,cycle_momentum=False)
    # was 0.1
    
        
    #end=time.time()
    #tdelta=end-start
    #print(tdelta,(nchains-c)*tdelta/3600)

SAMPLEFACTOR=0.95

lastroundstart=starttime
with open(opj(outpath,'all_models_pickled.pickle'),'wb') as f : 
    for x in ['orig']+['{:0>4}'.format(x) for x in range(1,int(NREPEATS)+1) ] : 
        if x == 'orig' : 
            thisy=yactual
        else: 
            npthisy=1/SAMPLEFACTOR*np.log(omics.sample(frac=SAMPLEFACTOR,replace=False).sum(axis=0)+1) #230920
            thisy=torch.tensor(npthisy,device=DEVICE,dtype=torch.float64)

        lmsg(x,lastroundstart) ;
        lastroundstart=time.time()

        with warnings.catch_warnings() :
            warnings.simplefilter('ignore')
            os.environ['PYTHONWARNINGS']='ignore'
            #mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1) #early
            #mod=LassoLarsIC(criterion='aic',positive=True,noise_variance=0.1,fit_intercept=False) #230910
            #mod=LassoLarsIC(criterion='aic',positive=False,noise_variance=0.1,fit_intercept=False) #230920
            aic,coef=run_metachain(thisy)
            
        nnz=( coef != 0).sum()
        nnz_systems=len(np.intersect1d(nkeys,np.array(bigxcols)[ coef != 0]))

        odd={ 'hierarchy' : hname ,
              'run_kind'  : x,
              'aic' : aic,
              'nnz' : nnz ,
              'nnz_systems' : nnz_systems,
               }
        outdata.append(odd)

        pickle.dump(coef,f)

odf=pd.DataFrame(outdata)
odf.to_csv(opj(outpath,'models_summary_data.csv'))
with open(opj(outpath,'bigX.pickle'),'wb') as f : 
    pickle.dump((bigxcols,omics.columns,bigX),f)
