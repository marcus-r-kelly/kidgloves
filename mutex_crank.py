import numpy as np
import pandas as pd
import sys
import kidgloves as kg

cohort=sys.argv[1]
task_id=sys.argv[2]
n_repeats=int(sys.argv[3])

cu=cohort.upper()
cl=cohort.lower()

logit_data=pd.read_csv(cu+'/'+'logit_data.csv',index_col=0)
import pickle
with open(cu+'/'+'logittransformer.pickle','rb') as f : 
    logittransformer=pickle.load(f) 

with open(cu+'/'+'nestmaskdict.pickle','rb') as f : 
    nestmaskdict=pickle.load(f)

import torch
import torch.sparse
import sptops

mvsize=100

nestmask=kg.arrayify_nest_mask(nestmaskdict,logit_data.lesion_class.values).to(torch.float32)
nestmask_sparse=nestmask.to_sparse()

truR=np.stack([ np.diag(logittransformer.training_data.values[x]) for x in range(logittransformer.training_data.shape[0])],axis=0) 
# observed gene-level events by patients (diag)
truL=np.matmul(truR,nestmask.numpy()) # observed system-gene event mappings
truQ=np.clip(np.matmul(truL.transpose((0,2,1)),truL),0,1) # observed system-level coincidences attributable to single events
truS=np.clip(np.dot(logittransformer.training_data.values,nestmask),0,1) # observed systems mutated by individual
truM=np.dot(truS.transpose(),truS) # observed system coincidences

tru_ss_obs=truM-truQ.sum(axis=0)

truW=truL.sum(axis=0)
truSE=np.dot(logittransformer.training_data.values.transpose(),truS)
#truSE has <genes> rows and <systems> columns
truX=(truSE-truW)

truGG=np.dot(logittransformer.training_data.values.transpose(),logittransformer.training_data.values)


with open(cu+'/'+'_'.join([task_id,'ss'])+'.txt','w') as ss : 
 with open(cu+'/'+'_'.join([task_id,'gs'])+'.txt','w') as gs : 
  with open(cu+'/'+'_'.join([task_id,'gg'])+'.txt','w') as gg : 

    for x in range(n_repeats) : 
        pat_gen=kg._ShC_generator(logittransformer.patients,mvsize)

        multiverselist=[ logittransformer(j) for j in pat_gen ]
        multiverse=sptops.sparse_tensor_from_index_array_iter(multiverselist,logittransformer.training_data.shape).to(torch.float32)

        megaR=sptops.diag_embed(multiverse,minorly=True).coalesce() # gene-gene mutation mapping by individual
        megaL=sptops.bmm(megaR,nestmask_sparse) # system-gene mutation mapping by individual
        megaQ=sptops.clip(sptops.bmm(megaL.transpose(2,3).coalesce(),megaL)) # system coincidences attributable to single mutations
        megaS=sptops.clip(sptops.bmm(multiverse,nestmask_sparse)) # systems mutated by individual
        megaM=sptops.bmm(megaS.transpose(2,1).coalesce(),megaS)  # multiverse-wide system coincidence

        megaMcounts=megaM.to_dense().numpy() #densification of system coincidence
        megaQcounts=megaQ.to_dense().numpy().sum(axis=1) #densification of system coincidence attributable to single events
        obsdmv=megaMcounts-megaQcounts #singleton-corrected coincidence counts


        megaW=megaL.to_dense().numpy().sum(axis=1) #system-event coincidences attributable to event within system
        megaSE=sptops.bmm(multiverse.transpose(1,2),megaS) # system-event coincidences
        megaX=(megaSE.to_dense()-megaW).numpy() # corrected system-event coincidences

        megaGG=sptops.bmm(multiverse.transpose(1,2),multiverse).to_dense().numpy()

        sscounts=(( obsdmv == tru_ss_obs ).astype(int) + 2*(obsdmv<tru_ss_obs).astype(int)).sum(axis=0)
        secounts=(( megaX == truX ).astype(int) + 2*(megaX<truX).astype(int)).sum(axis=0)
        ggcounts=(( megaGG == truGG ).astype(int) + 2*(megaGG<truGG).astype(int)).sum(axis=0)

        print(str(int(mvsize)*2)+','+','.join(np.cast['str'](np.cast['int'](sscounts.ravel()))),file=ss)
        print(str(int(mvsize)*2)+','+','.join(np.cast['str'](np.cast['int'](secounts.ravel()))),file=gs)
        print(str(int(mvsize)*2)+','+','.join(np.cast['str'](np.cast['int'](ggcounts.ravel()))),file=gg)
                


# each process needs to dump the items from its run as well;
# it can do that with 1 file per process

# for UCEC, one run through takes about 22 seconds
# so 100 runs through per process = 2,200 seconds
# 100M/10,000 simulated cohorts/process = 10,000 processes
# ^ necessary power for single-event
# of which we can do ~64 at a time --> 1 week for UCEC
# let's not worry about single event coincidences for now!

# power for system-event coincidences: 16M!
# 65M/10K 1600 processes
# of which we can do ~64 at a time --> 15h!
# this is doable.
# we can dump single-event coincidences anyway and restrict them to sufficiently common ones
# or events within systems, etc






