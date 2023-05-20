import numpy as np
import sys
import pandas as pd
import kidgloves as kg
from functools import reduce,partial
cohort=sys.argv[1]
cu=cohort.upper()
cl=cohort.lower()

import os
outpath=cu+'_epistasis'
if not os.path.exists(outpath) : 
    os.mkdir(outpath)

opj=os.path.join

datapref='/cellar/users/mrkelly/Data/'
casedatapath=opj(datapref,'tcga_processed','casedata')
aliquotpath=opj(casedatapath,'aliquot.tsv')
thiscohortpath=opj(datapref,'tcga_processed','cohort_'+cl)
cbioportal_path=datapref+'/canon/cbioportal/'+cl+'_tcga_pan_can_atlas_2018'


SIGNATURE_INNARDS=opj('Assignment_Solution','Activities')
NEST_HIERARCHY_PATH=datapref+'future_of_nest/RaTS_NeST/ratsnest_node_unrenamed_230425.csv'
#~~~~~~~~Read in and format omics signatures~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if cu == 'COADREAD' : 
    projects={ 'TCGA-COAD','TCGA_READ'}
    thisnestcancer='COAD'
else: 
    projects={ 'TCGA-'+cu ,}
    thisnestcancer=cu

aliquot=pd.read_csv(aliquotpath,sep='\t').query('project_id in @projects')
a2tsb=dict(zip(aliquot.aliquot_id,aliquot.case_submitter_id))

msig_subframes=list()
for msigdir in ('maf_out_sbs','maf_out_id','maf_out_dn') : 
    msig_subframe=pd.read_csv( opj( thiscohortpath,msigdir,SIGNATURE_INNARDS,'Assignment_Solution_Activities.txt'),
                            sep='\t')
    msig_subframe=msig_subframe.assign(Tumor_Sample_Barcode=msig_subframe.Samples.apply(lambda s : '-'.join(s.split('-')[:3])))
    msig_subframe=msig_subframe.drop(columns=['Samples']).groupby('Tumor_Sample_Barcode').mean()
    msig_subframes.append(msig_subframe)

sig_cnasubframe=pd.read_csv( opj( thiscohortpath,'maf_out_cn',SIGNATURE_INNARDS,'Assignment_Solution_Activities.txt'),
                            sep='\t')
sig_cnasubframe=sig_cnasubframe.assign(Tumor_Sample_Barcode=sig_cnasubframe.Samples.apply(a2tsb.get))
sig_cnasubframe=sig_cnasubframe.drop(columns=['Samples']).groupby('Tumor_Sample_Barcode').mean()
msig_subframes.append(sig_cnasubframe)

msigframe=np.log10(
    reduce(partial(pd.merge,left_index=True,right_index=True,how='outer'),msig_subframes).drop(cl)+1
)

df_arm=pd.read_csv(opj(cbioportal_path,'data_armlevel_cna.txt'),sep='\t').set_index('NAME').drop(columns=['ENTITY_STABLE_ID','DESCRIPTION']).transpose()
df_arm.index.name='Tumor_Sample_Barcode'
df_arm.index=[ x[:-3] for x in df_arm.index ]
df_arm=df_arm[ ~df_arm.isnull().all(axis=1) ].fillna('Gain').replace({ 'Unchanged' : 0, 'Loss' : -1  , 'Gain' : 1}).astype(int)
from sklearn.decomposition import PCA
df_arm_pc=pd.DataFrame(data=PCA(n_components=5).fit_transform(df_arm),index=df_arm.index,columns=['arm_pc_'+str(x) for x in range(5)])

msigframe=msigframe.join(df_arm_pc,how='inner').fillna(0.0)

msigframe=msigframe[ msigframe.columns[(msigframe!=0).sum().gt(0)] ]


#~~~~~~~~Read in the relevant parts of nest~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nestdf=kg.load_nest(NEST_HIERARCHY_PATH)
#nestdf=nestdf[ nestdf['Size'].le(50) & nestdf['No. significantly mutated cancer types'].ge(1) ]
nestdf=nestdf[nestdf['Size'].le(50) & 
    (nestdf['Significantly mutated cancer types'].str.contains(thisnestcancer)  | nestdf['No. significantly mutated cancer types'].ge(2)) ]

   #nest_cancer_types={ ct for ctl in nestdf['Significantly mutated cancer types'].values for ct in ctl.split(' ')} -{'',} 
   #if cl == "coadread"  : 
   #    nestdf=nestdf[ nestdf['Size'].le(50) & nestdf['Significantly mutated cancer types'].str.contains("COAD") ]
   #elif cl not in nest_cancer_types : 
   #    nestdf=nestdf[ nestdf['Size'].le(50) ]
   #else : 
   #    nestdf=nestdf[ nestdf['Size'].le(50) & nestdf['Significantly mutated cancer types'].str.contains(cu) ]

nest=kg.extract_nest_systems(nestdf)
nest_genes={ g for s in nest.values() for g in s }
nest_genes=nest_genes - {'',}

#~~~~~~~~Read in and transform cancer data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CBIOPORTAL_PATH=datapref+'/canon/cbioportal/'+cl+'_tcga_pan_can_atlas_2018'
preomics=kg.autoload_events(CBIOPORTAL_PATH,gene_set=list(nest_genes))
preomics.index=[ '-'.join(x.split('-')[:-1]) for x in preomics.index]
omics=preomics.reindex(msigframe.index).fillna(0)
omics=omics[ omics.columns[omics.sum().ne(0)]]

#~~~~~~~~... to create the training data... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.preprocessing import MaxAbsScaler
ft=pd.DataFrame(
    data=MaxAbsScaler().fit_transform(msigframe),
    index=msigframe.index,
    columns=msigframe.columns)
    

#~~~~~~~~... and use it to train the logit regressions~~~~~~~~~~~~~~~~~~~~~~~~~~
lt=kg.LogitTransformer(training_data=omics,bigC=0.1)
#~~~~~~~~VVV this defines the coefficients for each patient~~~~~~~~~~~~~~~~~~~~~
# the object can generate new patients

logit_data=lt.fit(ft)
logit_data.to_csv(opj(outpath,'logit_data.csv'))

import pickle
lt.save(opj(outpath,'logittransformer.pickle'))


#~~~~~~~~Finally, we need to grab the relevant nest systems~~~~~~~~~~~~~~~~~~~~~
nestmaskdict=kg.mask_nest_systems(nest,logit_data)
with open(opj(outpath,'nestmaskdict.pickle'),'wb') as f : 
    pickle.dump(nestmaskdict,f)



