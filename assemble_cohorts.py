import pandas as pd
import os
import sys
from subprocess import run
from SigProfilerAssignment import Analyzer as Analyze
import numpy as np
import multiprocessing as mp

#p_cbio='/cellar/users/mrkelly/Data/canon/cbioportal'
#p_raw='/cellar/users/mrkelly/Data/canon/tcga_segment'
p_hg19='/cellar/users/mrkelly/Data/canon/ncbi_reference/hg_19_blastdb_too_official/GRCh37_chrless.fna'
p_hg38='/cellar/users/mrkelly/Data/canon/ncbi_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna'
gdc_path='/cellar/users/mrkelly/Data/canon/gdc_direct/'

cwd=os.getcwd()
   #cohorts=[ 'blca', 'brca', 'cesc', 'coadread', 'esca', 'gbm',
   #'hnsc', 'kirc', 'kirp', 'laml', 'lgg', 'lihc', 'luad', 'lusc',
   #'ov', 'paad', 'prad', 'sarc', 'skcm', 'stad',
   #'thca', 'thym', 'ucec']

def get_aliquot_id_from_file(fp) : 
    with open(fp,'r') as thefile:
        thefile.readline()
        return (thefile.readline().strip().split('\t')[0],fp)

   #def do_mutations_from_cbioportal() :
   #    print('Working on mutations for',c)
   #    #~~~~vcfs from maf files~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #    if (not 'cohort_'+c in os.listdir('.')): os.mkdir('cohort_'+c)

   #    processed_muts_file_name='temporary/'+c+'.txt'

   #    muts_raw=pd.read_csv(os.path.join(p_cbio,c+'_tcga_pan_can_atlas_2018','data_mutations.txt'),sep='\t')
   #    if 'Annotation_Status' not in muts_raw.columns : 
   #        muts_raw.query('Verification_Status == "Verified"').to_csv(processed_muts_file_name,index=False,sep='\t')
   #    else : 
   #        muts_raw.query('Annotation_Status == "SUCCESS"').to_csv(processed_muts_file_name,index=False,sep='\t')

   #    this_maf_path=os.path.join(thispath,'mafs')
   #    if not os.path.exists(this_maf_path) :  os.mkdir(this_maf_path)

   #    run(['perl','/cellar/users/mrkelly/Data/canon/vcf2maf/mskcc-vcf2maf-754d68a/maf2vcf.pl',
   #        '--input-maf',processed_muts_file_name,'--ref-fasta',p_hg19,'--output-dir',
   #        os.path.join(thispath,'mafs'),'--per-tn-vcfs'])

   #    print('Mutation organizing done.')

   #    return
def read_maf(maf_path) : 
    return pd.read_csv(maf_path,comment='#',header=0,sep='\t')

def do_mutations_from_gdc(c,gdc_path) : 

    print('Working on mutations for',c)
    os.makedirs('cohort_'+c,exist_ok=True)
    os.makedirs('temporary',exist_ok=True)
    thispath=os.path.join(cwd,'cohort_'+c)

    processed_muts_file_name='temporary/'+c+'.txt'

    maffilepaths=list()
    for triplet in os.walk(gdc_path) :
        for terminalfile in triplet[-1] : 
            if terminalfile.endswith('.maf') : 
                maffilepaths.append(os.path.join(triplet[0],terminalfile))

    muts_raw=pd.concat([ read_maf(mfp) for mfp in maffilepaths ]).reset_index(drop=True)

    if 'Annotation_Status' not in muts_raw.columns : 
        #muts_raw.query('Verification_Status == "Verified"').to_csv(processed_muts_file_name,index=False,sep='\t')
        muts_good=muts_raw.assign(Verification_Status="Verified")
    else : 
        muts_good=muts_raw.query('Annotation_Status == "SUCCESS"')
    muts_good.to_csv(processed_muts_file_name,index=False,sep='\t')
    muts_good.to_csv(os.path.join(thispath,'mutations.maf'),index=False,sep='\t')

    run(['perl','/cellar/users/mrkelly/Data/canon/vcf2maf/mskcc-vcf2maf-754d68a/maf2vcf.pl',
        '--input-maf',processed_muts_file_name,'--ref-fasta',p_hg38,'--output-dir',
        os.path.join(thispath,'mafs'),'--per-tn-vcfs'])
    print('Mutation organizing done.')

    return

def do_segments(c,gdc_path) : 
    #~~~~segments from gdc data portal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aliquot=pd.read_csv(gdc_path+'/aliquot.tsv',sep='\t')
    aliquot_files={ os.path.join(triplet[0],fp) for triplet in os.walk(gdc_path) for fp in triplet[-1] if fp.endswith('.seg.txt') }
    #aliquot_files={ os.path.join(triplet[0],fp) for triplet in os.walk(gdc_path) for fp in triplet[-1] if fp.endswith('.allelic_specific.seg.txt') }
    with mp.Pool(processes=len(os.sched_getaffinity(0))) as p :  
        dl=dict([ x for x in p.imap_unordered(get_aliquot_id_from_file,aliquot_files) ])
    ldlk=list(dl.keys())

    print('Working on segments for',c)
    thispath=os.path.join(cwd,'cohort_'+c)
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

if __name__ == '__main__' and len(sys.argv) > 0 : 
    c=sys.argv[1]
    try : 
        gdc_path=sys.argv[2]
    except : 
        pass
#c='thym'

    pr1=mp.Process(target=do_mutations_from_gdc,args=(c,gdc_path))
    pr2=mp.Process(target=do_segments,args=(c,gdc_path))

    pr1.start()
    pr2.start()

    pr1.join()
    pr2.join()

    thispath=os.path.join(cwd,'cohort_'+c)
    vcfpath=os.path.join(thispath,'mafs')
    segpath=os.path.join(thispath,'segments','cn_segment.seg')
    sbsoutpath=os.path.join(thispath,'maf_out_sbs')
    dnoutpath=os.path.join(thispath,'maf_out_dn')
    idoutpath=os.path.join(thispath,'maf_out_id')
    cnoutpath=os.path.join(thispath,'maf_out_cn')

    Analyze.cosmic_fit(samples=vcfpath,output=sbsoutpath,input_type='vcf',verbose=False,context_type="96",genome_build='GRCh38')
    Analyze.cosmic_fit(samples=vcfpath,output=dnoutpath,input_type='vcf',verbose=False,context_type="DINUC",collapse_to_SBS96=False,genome_build='GRCh38')
    Analyze.cosmic_fit(samples=vcfpath,output=idoutpath,input_type='vcf',verbose=False,context_type="ID",collapse_to_SBS96=False,genome_build='GRCh38')
    Analyze.cosmic_fit(samples=segpath,output=cnoutpath,input_type='seg:ASCAT',collapse_to_SBS96=False,genome_build='GRCh38')
