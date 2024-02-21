import sys
import pandas as pd
import numpy as np
import os
from SigProfilerAssignment import Analyzer as Analyze
from sklearn.decomposition import NMF
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial,reduce
from itertools import product


# What workflows is this module/script trying to cover?
# Generation of "omics" tables for use by the largesses
# Generation of "signature" tables for use by the largesses
# - generation of arm-level PCs
# - aggregation of activities
_gi=None
_s2e=None
_e2s=None
_ens2e=None
_e2ens=None
BADGENES=None
geneinfopath='/cellar/users/mrkelly/Data/canon/ncbi_reference/Homo_sapiens.gene_info'

def msg(*args,**kwargs) : 
    print(*args,**kwargs)
    sys.stdout.flush()

def _get_geneinfo() : 
    """
    generate event pairs for logit field analysis, excluding those where 
    a single gene is associated with both events.
    """
    global _gi 
    global _s2e
    global _e2s
    global _ens2e
    global _e2ens
    global BADGENES

    _gi=pd.read_csv(geneinfopath,sep='\t')[::-1] 
    _gi['Ensembl']=get_ensembl_xref(_gi.dbXrefs)
    _gi['GeneID']=_gi.GeneID.astype(str)
    # [::-1] this means that for items iterating through, "older"/more canonical entries will be last and supersede shakier ones


    _e2s=dict()
    _s2e=dict()
    _ens2e=dict()
    _e2ens=dict()
    for r in _gi.itertuples() :
        _e2s.update({ r.GeneID : r.Symbol })
        _e2ens.update({ r.GeneID : r.Ensembl})
        _ens2e.update({ r.Ensembl : r.GeneID})
        _s2e.update({ r.Symbol : r.GeneID })

    bad_gene_categories={'other','pseudo','biological-region','unknown'}
    BADGENES=set(list(_gi.query("type_of_gene in @bad_gene_categories or (type_of_gene == 'ncRNA' and Ensembl == 'None')").GeneID.unique()))



@np.vectorize
def get_ensembl_xref(dbxrefs) : 
    for xref in dbxrefs.split('|') : 
        subfields=xref.split(':')
        if subfields[0] == 'Ensembl' : 
            return subfields[1]  
    else :
        return None
@np.vectorize
def trim_ensembl_version(ensembl) :
    return ensembl.split('.')[0]

@np.vectorize
def trim_gz(filename) : 
    if filename.endswith('.gz'): 
        return '.'.join(filename.split('.')[:-1])
    return filename
    
def fix_tsb(tsb) : 
    return '-'.join(tsb.split('-')[:3])

fix_tsb=np.vectorize(fix_tsb,otypes=[str,])

def get_pipeline(s) : 
    if not s.endswith('vcf') : 
        return ''
    else :
        ssp=s.split('.')
        for x,_ssp in enumerate(ssp) : 
            if _ssp == 'somatic_annotation' : 
                return ssp[x-1]
            
get_pipeline=np.vectorize(get_pipeline,otypes=[str,])

def read_cna(path) : 
#if True : 

    if _gi is None : 
        _get_geneinfo()

    df=pd.read_csv(path,sep='\t')
    df['Ensembl']=trim_ensembl_version(df.gene_id)
    df=df.merge(_gi[['Ensembl','GeneID']],on='Ensembl',how='left')
    df=df.dropna(subset='GeneID')
    df=df.query('GeneID not in @BADGENES').copy()
    df['GeneID']=df.GeneID.astype(int).astype(str)
    ser=df.set_index('GeneID').copy_number
    ser.name=path
    return ser.dropna()

import time
def process_cna_group(cnapaths,sampsheet) : 
    start=time.time()
    #msg('Reading...')
    with mp.Pool(processes=int(3*len(os.sched_getaffinity(0))//4)) as p : 
        cserieses=[ x for x in p.imap(read_cna,cnapaths) ]
    #cserieses=[ read_cna(cnap) for cnap in cnapaths ]
    #msg('{: >10.2f}'.format(time.time()-start))
    #msg('Forming protoc...')
    protoc=pd.DataFrame(cserieses).reset_index()
    #msg('{: >10.2}f'.format(time.time()-start))
    #msg('Handling protoc keys...')
    protoc=protoc.rename(columns={ 'index' : 'File Name' })
    protoc['File Name']=np.vectorize(lambda p : p.split('/')[-1])(protoc['File Name'])
    protoc=protoc.sort_values('File Name')
    #msg('{: >10.2f}'.format(time.time()-start))
    #msg('Merging into meta...')
    subss=sampsheet[['File Name','simple_case_id']].sort_values('File Name')
    urmerc=protoc.merge(subss,on='File Name',how='left').drop(columns=['File Name'])
    merc=urmerc.groupby('simple_case_id').mean()
    #merc=merc.set_index('simple_case_id',drop=True)
    #msg('{: >10.2f}'.format(time.time()-start))
    #merc=merc.fillna(merc.median(axis=2))
    merc=merc.fillna(2.0)
    #msg('Breaking into up/dn...')
    up=pd.DataFrame(index=merc.index,columns=merc.columns,data=np.where(merc.values >=4 ,True,False))
    dn=pd.DataFrame(index=merc.index,columns=merc.columns,data=np.where(merc.values <=1 ,True,False))
    #msg('{: >10.2f}'.format(time.time()-start))
    #msg('Merging up/dn...')
    cna=pd.merge(up,dn,suffixes=['_up','_dn'],left_index=True,right_index=True)
    #msg('{: >10.2f}'.format(time.time()-start))
    return cna

_consequential_mutations={'NMD_transcript_variant',
 'missense_variant',
 'splice_acceptor_variant',
 'splice_donor_variant',
 'splice_region_variant',
 'stop_gained',
 'non_coding_transcript_exon_variant',"5_prime_UTR_variant","3_prime_UTR_variant"}

def is_consequential(mutstr) : 
    muttypes=set(mutstr.split(';'))
    if _consequential_mutations & muttypes : 
        return True
    return False

is_consequential=np.vectorize(is_consequential,otypes=[bool,])

def read_maf(t) : 
#if True : 
    path,tsb=t
    try: 
        maf=pd.read_csv(path,comment='#',sep='\t',low_memory=False)
        if maf.shape[0] == 0 : 
            return pd.DataFrame(data=[['tick',tsb],],columns=['Entrez_Gene_Id','Tumor_Sample_Barcode'])
        
        maf['Entrez_Gene_Id']=maf.Entrez_Gene_Id.astype(str)
        maf['Tumor_Sample_Barcode']=fix_tsb(maf.Tumor_Sample_Barcode)
        maf['is_consequential']=is_consequential(maf.Consequence)
        maf=maf[ maf.is_consequential & maf.Entrez_Gene_Id.ne('0') & ~maf.Entrez_Gene_Id.isin(BADGENES) ]
        mv=maf[['Entrez_Gene_Id','Tumor_Sample_Barcode']].copy()
        mv.iloc[-1]=pd.Series({'Entrez_Gene_Id' : 'tick', 'Tumor_Sample_Barcode' : tsb })
        #mv.iloc[-1]=pd.Series({'Entrez_Gene_Id' : 'tick', 'Tumor_Sample_Barcode' : maf.loc[maf.index[0],'Tumor_Sample_Barcode'] })
    except : 
        msg(path) 
        raise
    
    
    return mv
                                       
def gen_mut_jobs(mafpaths,tsbs) :
    for m,tsb in zip(mafpaths,tsbs): 
        yield (m,tsb)
                                       
def process_maf_group(mafpaths,sampsheet): 
    slc=sampsheet[ sampsheet.path.isin(mafpaths) ]
    jobs=gen_mut_jobs(slc.path.values,slc.simple_case_id.values)
               
    with mp.Pool(processes=int(len(os.sched_getaffinity(0))//4)) as p : 
        mserieses=[ x for x in tqdm(p.imap(read_maf,jobs),total=slc.shape[0]) ]
    #mserieses=[ read_maf(j) for j in jobs ]
               
    protom=pd.concat(mserieses)
    vc=protom.value_counts()
    vf=( vc/vc.xs('tick',level=0) >= 0.5 )
    
    mpiv=vf.sort_index().drop('tick').reset_index().pivot(index='Tumor_Sample_Barcode',columns='Entrez_Gene_Id',values=0)
    mpiv=mpiv.reindex(slc.simple_case_id.unique())
    mpiv.values[ mpiv.isnull() ]=False
    return mpiv

def parse_arriba(path) : 
    fus=pd.read_csv(path,sep='\t')
    outserdata=list()

    for x,r in fus.iterrows() : 
        if r.confidence == 'high' or r.confidence == 'medium' : 
            g1=_s2e.get(r['#gene1'])
            g2=_s2e.get(r['gene2'])
            
            outserdata.append({ 'File Name' : os.path.split(path)[-1]  , 
                                'Entrez_Gene_Id' : g1})          
            outserdata.append({ 'File Name' : os.path.split(path)[-1]  , 
                                'Entrez_Gene_Id' : g2})
    outserdata.append({ 'File Name' :os.path.split(path)[-1], 'Entrez_Gene_Id' : 'tick' })
    return pd.DataFrame(outserdata)

def extract_entrez_starfus(starfusgene) : 
    ens=starfusgene.split('^')[-1].split('.')[0]
    e=_ens2e.get(ens)
    return e

extract_entrez_starfus=np.vectorize(extract_entrez_starfus,otypes=[str,])

def parse_star_fusion(path) : 
    outserdata=list()
    fus=pd.read_csv(path,sep='\t')
    fus['entrez_left']=extract_entrez_starfus(fus.LeftGene)
    fus['entrez_right']=extract_entrez_starfus(fus.RightGene)
    fus=fus[ ~fus.entrez_left.isnull() & ~fus.entrez_right.isnull()  & fus.LargeAnchorSupport.eq('YES_LDAS') ]
    for x,r in fus.iterrows() : 
        outserdata.append({ 'File Name' : os.path.split(path)[-1], 'Entrez_Gene_Id' : r.entrez_left})
        outserdata.append({ 'File Name' : os.path.split(path)[-1], 'Entrez_Gene_Id' : r.entrez_right })
        
    outserdata.append({ 'File Name' :os.path.split(path)[-1], 'Entrez_Gene_Id' : 'tick' })
    return pd.DataFrame(outserdata)

def read_fusion(path) : 
    if 'arriba' in path.lower() : return parse_arriba(path)
    elif 'star_fusion' in path.lower() : return parse_star_fusion(path)  
    else: 
        raise ValueError('Do not know how to parse path {}'.format(path))

def process_fusion_group(fuspaths,sampsheet) : 

    if _gi is None :
        _get_geneinfo()

    #with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
        #fusserieses=[ x for x in tqdm(p.imap(read_fusion,fuspaths),total=len(fuspaths)) ]
    fusserieses=[ read_fusion(fp) for fp in fuspaths ]

    fv=pd.concat(fusserieses)
    fv=fv.merge(sampsheet[['File Name','simple_case_id']],on='File Name',how='left').drop(columns=['File Name']).dropna()
    fv=fv.rename(columns={'simple_case_id' : 'Tumor_Sample_Barcode'})
    outindex=fv.query('Entrez_Gene_Id == "tick"').Tumor_Sample_Barcode.unique()
    fvc=fv.value_counts() 
    fvf=(fvc/fvc.xs('tick',level=0) > 0.5)
    fpiv=fvf.drop('tick',axis=0,level=0).reset_index().pivot(index='Tumor_Sample_Barcode',columns='Entrez_Gene_Id',values=0)
    fpiv=fpiv.reindex(outindex).sort_index().drop(columns='None',errors='ignore')
    fpiv.values[fpiv.isnull()]=False
    return fpiv

@np.vectorize
def simplify_sample_type(sampletype) : 
    return {
    'Solid Tissue Normal, Primary Tumor'  : "pair"  , 
    'Primary Tumor' : 'tumor' ,
    'Blood Derived Normal, Primary Tumor' : 'pair'  , 
    'Primary Tumor, Solid Tissue Normal'  : 'pair' , 
    'Solid Tissue Normal'                 : 'normal' , 
    'Primary Tumor, Blood Derived Normal'  : 'pair'  , 
    'Blood Derived Normal, FFPE Scrolls'   : 'pair'  , 
    'Recurrent Tumor, Blood Derived Normal' : 'pair' , 
    'FFPE Scrolls, Blood Derived Normal'    : 'pair'  , 
    'Blood Derived Normal, Recurrent Tumor' : 'pair'  , 
    'Recurrent Tumor'                      : 'tumor' , 
    }.get(sampletype)


def generate_mutational_signatures(vcf_paths,patient_ids,jobroot) : 

    vcffiledir=os.path.join(jobroot,'vcf_inputs/')
    out_sbs_dir=os.path.join(jobroot,'out_sbs/')
    out_dn_dir=os.path.join(jobroot,'out_dn/')
    out_id_dir=os.path.join(jobroot,'out_id/')

    outpaths=[ out_sbs_dir, out_dn_dir, out_id_dir ]
    context_types=['96','DINUC','ID']

    os.makedirs(vcffiledir,exist_ok=True)

    for x,vcfp in tqdm(enumerate(vcf_paths),total=len(vcf_paths)) : 
        last=os.path.split(vcfp)[-1]
        futurepath=os.path.join(vcffiledir,patient_ids[x]+'.vcf')
        if os.path.exists(futurepath) and os.path.islink(futurepath) : 
            continue
        elif os.path.exists(futurepath) : 
            raise FileError(f"{futurepath} exists already and is not a symlink")
        else : 
            os.symlink(vcfp,futurepath)

    for outpath,ct in zip(outpaths,context_types) : 
        Analyze.cosmic_fit(
                samples=vcffiledir,
                output=outpath,
                input_type='vcf',
                verbose=False,
                context_type=ct,
                collapse_to_SBS96=( ct == '96'),
                genome_build='GRCh38',
                exclude_signature_subgroups=['Artifact_signatures',],
                make_plots=False,
                )

def organize_segments(seg_paths,patient_ids,jobroot) : 
    segfiledir=os.path.join(jobroot,'seg_inputs/')
    segfile=os.path.join(segfiledir,'segments.seg')

    os.makedirs(segfiledir,exist_ok=True)

    frames=list()
    for x,sfp in enumerate(seg_paths) : 
        sfplast=os.path.split(sfp)[-1]
        frame=pd.read_csv(sfp,sep='\t').rename(
                columns={
                    'GDC_Aliquot' : 'sample',
                    'Start' : 'startpos',
                    'End' : 'endpos',
                    'Major_Copy_Number' : 'nMajor' ,
                    'Minor_Copy_Number' : 'nMinor' 
                    }
                )
        frame['sample']=patient_ids[x]
        frames.append(frame)

    pd.concat(frames).to_csv(segfile,sep='\t')

@np.vectorize
def expandgfftags(tagstring) : 
    tags=tagstring.split(';')
    td=dict()
    for tag in tags: 
        ts=tag.split('=')
        td.update({ ts[0] : ts[1] })
    return td

@np.vectorize
def acc_to_chr(x) : 
    xsp=x.split('.')[0].split('0')
    chrno=xsp[-1]
    if chrno == '' : 
        chrno = ''.join([xsp[-2],'0'])
    if chrno == '23' :
        return 'chrX'
    if chrno == '24' : 
        return 'chrY'
    return 'chr'+chrno
        


def generate_cn_signatures(segfilepath,jobroot) : 

    out_cn_dir=os.path.join(jobroot,'out_cn/')
    Analyze.cosmic_fit(
            samples=segfilepath,
            output=out_cn_dir,
            input_type='seg:ASCAT',
            verbose=False,
            collapse_to_SBS96=False,
            genome_build='GRCh38',
            make_plots=False,
            )

def load_gene_regions() : 
    gff=pd.read_csv('/cellar/users/mrkelly/Data/canon/ncbi_reference/GRCh38_latest_genomic.gff',sep='\t',comment='#',names=['acc','src','kind','start','stop','foo1','strand','foo2','tags'])
    gff=gff[ gff.tags.str.contains('genome=chromosome') | gff.kind.eq('centromere') ]
    gffp=gff.pivot_table(index='acc',columns='kind',values=['stop','start'])
    gffp[('other','chr')]=acc_to_chr(gffp.index)
    return gffp

def get_armlevel_cnas(segfilepath,gffp) : 
    armregions=list()
    for x,r in gffp.iterrows() : 
        serp=pd.Series({ 'chr' : r[('other','chr')] ,
                        'arm' : 'p' , 
                        'start' : r[('start','region')],
                        'stop'  : r[('start','centromere')],
                      })
        
        armregions.append(serp)
        
        serq=pd.Series({ 'chr' : r[('other','chr')] ,
                        'arm' : 'q' , 
                        'start' : r[('stop','centromere')],
                        'stop'  : r[('stop','region')],
                      })
        
        armregions.append(serq)
    dfarm=pd.DataFrame(armregions)
    dfarm['chrarm']=dfarm.chr+dfarm.arm
    dfarm=dfarm.set_index('chrarm')

    allseg=pd.read_csv('seg_inputs/segments.seg',sep='\t',index_col=0)

    ntasks=dfarm.shape[0]*allseg['sample'].nunique()
    alcnadatas=list()
    chrarm='chr2p'
    for chrarm in tqdm(dfarm.index,total=dfarm.shape[0]) :
        thechr=chrarm[:-1]
        slc=allseg.query('Chromosome == @thechr')
        @np.vectorize
        def getoverlap(segmin,segmax) :
            armmin=dfarm.loc[chrarm].start
            armmax=dfarm.loc[chrarm].stop
            return max(0,min(armmax,segmax)-max(armmin,segmin))
            
            
        overlaps=getoverlap(
                slc.startpos,
                slc.endpos,
                )
        
        odf=pd.DataFrame(data=[slc['sample'].values,overlaps],index=['sample','overlap']).transpose()
        sampwiseoverlapsum=odf.groupby('sample').overlap.sum()
        sampwiseoverlapsum.name='sample_wise_overlap_sum'
        sampwiseoverlapsum=sampwiseoverlapsum.reset_index(drop=False)
        
        slc=slc.merge(sampwiseoverlapsum,on='sample',how='left')
        slc['overlap']=overlaps
        slc['wcn']=overlaps*slc.Copy_Number/slc.sample_wise_overlap_sum
        
        subfr=slc.groupby('sample').wcn.sum()
        subfr=subfr.reset_index(drop=False)
        subfr['chrarm']=chrarm
        
        alcnadatas.append(subfr)
                          
            
    dfalcna=pd.concat(alcnadatas)

    armpiv=dfalcna.pivot_table(index='sample',columns='chrarm',values='wcn').fillna(0)
    mynmf=NMF(n_components=5)
    arm5=pd.DataFrame(data=mynmf.fit_transform(armpiv),index=armpiv.index,columns=['arm_pc_'+str(x) for x in range(5)])

    return arm5


def parse_sample_sheet(samplesheetpath,gdc_root) : 
    sampsheet=pd.read_csv(samplesheetpath,sep='\t')
    sampsheet['simple_case_id']=np.vectorize(lambda s : s.split(',')[0])(sampsheet['Case ID'])
    sampsheet['File Name']=trim_gz(sampsheet['File Name'])
    sampsheet['simple_sample_type']=simplify_sample_type(sampsheet['Sample Type'])
    sampsheet['extension']=np.vectorize(lambda s : s.split('.')[-1])(sampsheet['File Name'])

    fn2path=dict()
    for triplet in os.walk(gdc_root) : 
        for subtriplet in triplet[-1] : 
            fn2path.update({ subtriplet : os.path.join(triplet[0],subtriplet) })

    sampsheet['path']=np.vectorize(fn2path.get)(sampsheet['File Name'])
    sampsheet['pipeline']=get_pipeline(sampsheet['File Name'])

    return sampsheet

def  _wr_process_cna_group(cnapaths,sampsheet,outQ) : 
    outQ.put(('cna',process_cna_group(cnapaths,sampsheet)))

def  _wr_process_maf_group(mutpaths,sampsheet,outQ) : 
    outQ.put(('mut',process_maf_group(mutpaths,sampsheet)))

def  _wr_process_fusion_group(fuspaths,sampsheet,outQ) : 
    outQ.put(('fus' , process_fusion_group(fuspaths,sampsheet) ))

def compile_omics(root_folder,sampsheet) : 

    if _gi is None : 
        _get_geneinfo()

    outQ=mp.Queue()
    cnapaths=sampsheet[ sampsheet['Data Type'].eq('Gene Level Copy Number')  & sampsheet['simple_sample_type'].eq('pair') ].path
    proc_cna=mp.Process(target=_wr_process_cna_group,args=(cnapaths,sampsheet,outQ))
    #msg('Processing cnas...')
    #cnas=process_cna_group(cnapaths,sampsheet)

    #msg('Processing mutations...')
    mutpaths=sampsheet[ sampsheet['Data Type'].eq('Masked Somatic Mutation') & sampsheet['simple_sample_type'].eq('pair') ].path.values
    proc_mut=mp.Process(target=_wr_process_maf_group,args=(mutpaths,sampsheet,outQ))

    #msg('Processing fusions...')
    fuspaths_tumor=sampsheet[ sampsheet['Data Type'].eq('Transcript Fusion') & sampsheet.simple_sample_type.eq('tumor') ].path
    proc_fus=mp.Process(target=_wr_process_fusion_group,args=(fuspaths_tumor,sampsheet,outQ))
    #fuspaths_norm=sampsheet[ sampsheet['Data Type'].eq('Transcript Fusion') & sampsheet.simple_sample_type.eq('normal') ].path
    #fus_norm=read_fusion_group(fuspaths_norm,sampsheet)

    processdict={ 
            'cna' : proc_cna ,
            'mut' : proc_mut ,
            'fus' : proc_fus ,
            }

    for k in processdict : 
        msg('Processing',k,'...')
        processdict[k].start()

    retdict=dict()
    while len(retdict) < 3 : 
        try : 
            res=outQ.get(block=False,timeout=1)
        except mp.queues.Empty : 
            res=None
        if res is not None : 
            retdict.update({ res[0] : res[1] })
            msg('Finished',res[0],'.')
            processdict[res[0]].join()

    mpiv=retdict['mut']
    cna=retdict['cna']
    fus=retdict['fus']

    msg('Merging...')
    omics=reduce(
            partial(
                pd.DataFrame.join,
                how='inner'
                ),
            [mpiv.rename(columns=lambda s : s+'_mut'),
                cna,
                fus.rename(columns=lambda s : s+'_fus')]
            )

    ot=omics.transpose()
    ot=ot.reset_index()
    ot['eid']=ot['index'].apply(lambda s : s.split('_')[0])
    ot['etype']=ot['index'].apply(lambda s : s.split('_')[0])
    ot=ot.drop(columns='index')
    genewisesums=ot.drop(columns='etype').groupby('eid').sum().sum(axis=1)
    boringgenes=genewisesums[ genewisesums.eq(0) ].index

    droppables=np.array([ g+'_'+e for g,e in product(boringgenes,['mut','up','dn','fus']) ])

    omics=omics.drop(columns=np.intersect1d(omics.columns,droppables))

    return omics

import multiprocessing as mp

def wf_compile_signature_activities(ns) : 

    ss=parse_sample_sheet(ns.sample_sheet_path,ns.gdc_root)
    slc_vcf=ss.query('simple_sample_type == "pair" and extension == "vcf" and pipeline == "MuTect2"')
    slc_seg=ss[ ss['Data Type'].str.lower().str.contains('allele') ].copy()
    generate_mutational_signatures(slc_vcf.path.values,slc_vcf.simple_case_id.values,ns.cohort_root) 
    organize_segments(slc_seg.path.values,slc_seg.simple_case_id.values,ns.cohort_root)
    generate_cn_signatures(
            os.path.join(ns.cohort_root,'seg_inputs','segments.seg'),ns.cohort_root
            )

    SIGNATURE_INNARDS=os.path.join('Assignment_Solution','Activities')
    msig_subframes=list()
    for msigdir in ('out_sbs','out_id','out_dn','out_cn') : 
        msig_subframe=pd.read_csv( os.path.join( ns.cohort_root,msigdir,SIGNATURE_INNARDS,'Assignment_Solution_Activities.txt'),
                                sep='\t')
        msig_subframe=msig_subframe.assign(Tumor_Sample_Barcode=msig_subframe.Samples.apply(lambda s : '-'.join(s.split('-')[:3])))
        msig_subframe=msig_subframe.drop(columns=['Samples']).groupby('Tumor_Sample_Barcode').mean()
        msig_subframes.append(msig_subframe)

    from functools import partial,reduce
    msigframe=reduce(partial(pd.merge,left_index=True,right_index=True,how='inner'),msig_subframes)
    msigframe=np.log10(msigframe[msigframe.columns[msigframe.sum(axis=0).gt(0)]]+1)

    if hasattr(ns,'as_script') and ns.as_script : 
        msigframe.join(arm,how='inner').to_csv(os.path.join(ns.cohort_root,'mutation_signatures.csv'))
    else : 
        return msigframe
