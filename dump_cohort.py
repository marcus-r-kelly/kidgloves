import sys
import pandas as pd
import kidgloves as kg
cohort=sys.argv[1]
cu=cohort.upper()
cl=cohort.lower()

import os
if not os.path.exists(cu) : 
    os.mkdir(cu)

#~~~~~~~~Read in the relevant parts of nest~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nestdf=kg.load_nest('~/Data/future_of_nest/mutation_epistasis/v0/nest_node.csv')
#nestdf=nestdf[ nestdf['Significantly mutated cancer types (aggregate)'].str.contains(cu) & nestdf['Size'].le(50) ]
nestdf=nestdf[ nestdf['Significantly mutated cancer types'].str.contains(cu) & nestdf['Size'].le(50) ]
nest=kg.extract_nest_systems(nestdf)
nest_genes={ g for s in nest.values() for g in s }

#~~~~~~~~Read in and transform cancer data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
muts=kg.read_mutation_file('~/Data/canon/cbioportal/'+cohort+'_tcga_pan_can_atlas_2018/data_mutations.txt')
cnas=kg.read_cna_file('~/Data/canon/cbioportal/'+cohort+'_tcga_pan_can_atlas_2018/data_log2_cna.txt')
rnas=kg.read_rna_file('~/Data/canon/cbioportal/'+cohort+'_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt')
omics=kg.sync_omics(muts,cnas,rnas,gene_set=set(nest_genes),logic='force')

hsgi=pd.read_csv('/cellar/users/mrkelly/Data/canon/ncbi_reference/Homo_sapiens.gene_info',sep='\t')
import re
hsgi['arm']=hsgi.map_location.apply(lambda x : ''.join(re.split(r'([pq])',x)[:2]))
hsgi=hsgi.query('arm != "-" and type_of_gene == "protein-coding"')

armlevel=omics['cnas'].transpose().reset_index().merge(
    hsgi[['Symbol','arm']],
    left_on='gene_symbol',
    right_on='Symbol',
    ).drop(
    columns=['Symbol','gene_symbol']).groupby('arm').mean()

ct=kg.CohortTransformer()
ct.fit(omics['muts'],omics['cnas'])
#~~~VV~~~These are the mutation burdens and CNA PCs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_patients=ct.transform(omics['muts'],omics['cnas'])


#~~~~~~~~Now we define the lesions of functional interest~~~~~~~~~~~~~~~~~~~~~~~
mpiv=kg.pivot_mutation_events(omics['muts'])
mpiv=mpiv.rename(columns=lambda x : x+'_mut')
ups=kg.define_lesionclass(
        [
            omics['rna'],
            omics['cnas'],
        ],
        [
            lambda x : x > 1.6 ,
            lambda x : x > 1.6 ,
        ],'up',min_events_to_keep=5)

dns=kg.define_lesionclass(
        [
            omics['rna'],
            omics['cnas'],
        ],
        [
            lambda x : x < -1.6 ,
            lambda x : x < -1 ,
        ],'dn',min_events_to_keep=5)

#~~~~~~~~... to create the training data... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
td=pd.concat([mpiv,ups,dns],axis=1).astype(int)

#~~~~~~~~... and use it to train the logit regressions~~~~~~~~~~~~~~~~~~~~~~~~~~
lt=kg.LogitTransformer(training_data=td)
#~~~~~~~~VVV this defines the coefficients for each patient~~~~~~~~~~~~~~~~~~~~~
# the object can generate new patients

logit_data=lt.fit(t_patients)
logit_data.to_csv(cu+'/'+'logit_data.csv')

import pickle
with open(cu+'/'+'logittransformer.pickle','wb') as f : 
    pickle.dump(lt,f)


#~~~~~~~~Finally, we need to grab the relevant nest systems~~~~~~~~~~~~~~~~~~~~~
nestmaskdict=kg.mask_nest_systems(nest,logit_data)
with open(cu+'/'+'nestmaskdict.pickle','wb') as f : 
    pickle.dump(nestmaskdict,f)



