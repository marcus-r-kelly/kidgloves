import numpy as np
import sys
import pandas as pd
import kidgloves as kg
from functools import reduce,partial
import yaml
import os
opj=os.path.join

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--cohort',action='store',help='TCGA cohort code, case-insensitive. Note that cbioportal combines colon and rectal adenocarcinoma; these cohorts should be passed as "COADREAD" to read both patient sets from data not feched from cbioportal')
parser.add_argument('--cohort_suffix',action='store',help='suffix to append to output folder; that is, outputfolder will be <cohort_lower>_<cohort_suffix>')
parser.add_argument('--config_file',action='store',default=opj(os.getenv('HOME'),'.config','kgconfig.yaml'),help='location of desired kgconfig file')
parser.add_argument('--n_pca_components',action='store',default=5,help='Number of PCs to use to reduce arm-level CNA data')

ns=parser.parse_args()

cohort=ns.cohort
csuf=ns.cohort_suffix
# was "_epistasis"
cu=cohort.upper()
cl=cohort.lower()


outpath=cu+csuf
if not os.path.exists(outpath) : 
    os.mkdir(outpath)

with open(ns.config_file,'r') as y :
    kgc=yaml.safe_load(y)

assert os.path.isdir(kgc.get('cbioportal_folder_prefix'))
assert os.path.isdir(kgc.get('signature_folder_prefix'))

casedatapath=opj(kgc['signature_folder_prefix'],'casedata')
aliquotpath=opj(casedatapath,'aliquot.tsv')
thiscohortpath=opj(kgc['signature_folder_prefix'],'cohort_'+cl)

cbioportal_path=opj(kgc['cbioportal_folder_prefix'],cl+'_tcga_pan_can_atlas_2018')


SIGNATURE_INNARDS=opj('Assignment_Solution','Activities')
#TODO: this needs to get  moved into its own script that saves the signature/arm frame
# then, that frame needs to be read and pointed to in this script
#~~~~~~~~Read in and format omics signatures~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Loading signatures...',end='')
if cu == 'COADREAD' : 
    projects={ 'TCGA-COAD','TCGA-READ'}
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
df_arm.index=[ '-'.join(x.split('-')[:3]) for x in df_arm.index ]
# used to axe last 3 characters

df_arm=df_arm[ ~df_arm.isnull().all(axis=1) ].fillna('Gain').replace({ 'Unchanged' : 0, 'Loss' : -1  , 'Gain' : 1}).astype(int)
from sklearn.decomposition import PCA
df_arm_pc=pd.DataFrame(data=PCA(n_components=int(ns.n_pca_components)).fit_transform(df_arm),index=df_arm.index,columns=['arm_pc_'+str(x) for x in range(5)])

msigframe=msigframe.join(df_arm_pc,how='inner').fillna(0.0)

msigframe=msigframe[ msigframe.columns[(msigframe!=0).sum().gt(0)] ]

print('Done.')

msigframe.to_csv(opj(outpath,'mutation_signatures.csv'))
