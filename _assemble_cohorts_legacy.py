import pandas as pd
import os
import sys
from subprocess import run
from SigProfilerAssignment import Analyzer as Analyze
import numpy as np

p_cbio='/cellar/users/mrkelly/Data/canon/cbioportal'
p_raw='/cellar/users/mrkelly/Data/canon/tcga_segment'
p_hg19='/cellar/users/mrkelly/Data/canon/ncbi_reference/hg_19_blastdb_too_official/GRCh37_chrless.fna'

   #cohorts=[ 'blca', 'brca', 'cesc', 'coadread', 'esca', 'gbm',
   #'hnsc', 'kirc', 'kirp', 'laml', 'lgg', 'lihc', 'luad', 'lusc',
   #'ov', 'paad', 'prad', 'sarc', 'skcm', 'stad',
   #'thca', 'thym', 'ucec']
c=sys.argv[1]
#c='thym'

if not 'temporary' in os.listdir('.') : os.mkdir('temporary')
cwd=os.getcwd()

aliquot=pd.read_csv('casedata/aliquot.tsv',sep='\t')

aliquot_files={ os.path.join(triplet[0],fp) for triplet in os.walk(p_raw) for fp in triplet[-1] if fp.endswith('.allelic_specific.seg.txt') }

def get_aliquot_id_from_file(fp) : 
    with open(fp,'r') as thefile:
        thefile.readline()
        return (thefile.readline().strip().split('\t')[0],fp)
import multiprocessing as mp

with mp.Pool(processes=len(os.sched_getaffinity(0))) as p :  
    dl=dict([ x for x in p.imap_unordered(get_aliquot_id_from_file,aliquot_files) ])

ldlk=list(dl.keys())

thispath=os.path.join(cwd,'cohort_'+c)

def do_mutations() :
    print('Working on mutations for',c)
    #~~~~vcfs from maf files~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (not 'cohort_'+c in os.listdir('.')): os.mkdir('cohort_'+c)

    processed_muts_file_name='temporary/'+c+'.txt'

    muts_raw=pd.read_csv(os.path.join(p_cbio,c+'_tcga_pan_can_atlas_2018','data_mutations.txt'),sep='\t')
    if 'Annotation_Status' not in muts_raw.columns : 
        muts_raw.query('Verification_Status == "Verified"').to_csv(processed_muts_file_name,index=False,sep='\t')
    else : 
        muts_raw.query('Annotation_Status == "SUCCESS"').to_csv(processed_muts_file_name,index=False,sep='\t')

    this_maf_path=os.path.join(thispath,'mafs')
    if not os.path.exists(this_maf_path) :  os.mkdir(this_maf_path)

    run(['perl','/cellar/users/mrkelly/Data/canon/vcf2maf/mskcc-vcf2maf-754d68a/maf2vcf.pl',
        '--input-maf',processed_muts_file_name,'--ref-fasta',p_hg19,'--output-dir',
        os.path.join(thispath,'mafs'),'--per-tn-vcfs'])

    print('Mutation organizing done.')

    return

def do_segments() : 
    #~~~~segments from gdc data portal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Working on segments for',c)
    if c == 'coadread' : 
        projects={ 'TCGA-COAD','TCGA-READ'}
        valid_aliquot_rows=aliquot.query('project_id in @projects')
    else : 
        project='TCGA-'+c.upper()
        valid_aliquot_rows=aliquot.query('project_id == @project')
    valid_aliquots=valid_aliquot_rows.aliquot_id.unique()


    segs=pd.concat([ pd.read_csv(dl[va],sep='\t') for va in np.intersect1d(valid_aliquots,ldlk) ]).rename(
    columns={ 'GDC_Aliquot' : 'sample', 'Start' : 'startpos', 'End' : 'endpos', 'Major_Copy_Number' : 'nMajor' , 'Minor_Copy_Number' : 'nMinor' })

    #segs=seg_frame.query('GDC_Aliquot in @valid_aliquots')

    thissegpath=os.path.join(thispath,'segments')
    if not os.path.exists(thissegpath) : os.mkdir(thissegpath)

    segs.to_csv(os.path.join(thissegpath,'cn_segment.seg'),sep='\t',index=False)

    print('Segment organizing done.')

    return

pr1=mp.Process(target=do_mutations)
pr2=mp.Process(target=do_segments)

pr1.start()
pr2.start()

pr1.join()
pr2.join()

vcfpath=os.path.join(thispath,'mafs')
segpath=os.path.join(thispath,'segments','cn_segment.seg')
sbsoutpath=os.path.join(thispath,'maf_out_sbs')
dnoutpath=os.path.join(thispath,'maf_out_dn')
idoutpath=os.path.join(thispath,'maf_out_id')
cnoutpath=os.path.join(thispath,'maf_out_cn')

Analyze.cosmic_fit(samples=vcfpath,output=sbsoutpath,input_type='vcf',verbose=False,context_type="96")
Analyze.cosmic_fit(samples=vcfpath,output=dnoutpath,input_type='vcf',verbose=False,context_type="DINUC",collapse_to_SBS96=False)
Analyze.cosmic_fit(samples=vcfpath,output=idoutpath,input_type='vcf',verbose=False,context_type="ID",collapse_to_SBS96=False)
Analyze.cosmic_fit(samples=segpath,output=cnoutpath,input_type='seg:ASCAT',collapse_to_SBS96=False)
