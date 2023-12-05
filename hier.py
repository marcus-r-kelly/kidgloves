import kidgloves as kg
from converters  import s2eid,eid2s_current
import numpy as np
import pandas as pd

def load_nest_hierarchy(edges_path) : 

    g2ns=dict()
    nh=dict()
    ogenes=set()

    with open(edges_path) as f: 
        for line in f : 
            ls=line.strip().split('\t')
            protochild=ls[1]
            if ls[-1] == 'gene'  :
                child=s2eid.get(protochild)
                if child is None : continue
                # add gene to list of genes of interest
                ogenes.add(child)
            else : 
                child=protochild
            
            # index system to gene
            g2ns.update({ child : g2ns.get(child,set()) | {ls[0],}})
            # and parent to all its children
            nh.update({ ls[0] : nh.get(ls[0],set()) | {child,}})
        
    nhkeys=set(nh.keys())
    changes_made=True
    x=0
    while changes_made : 
        x+=1
        #print(x) ;
        changes_made=False
        for k in nhkeys: 
            for cs in (nh[k] & nhkeys) :
                nh.update({ k : (nh[k] | nh[cs])-{cs,} })
                changes_made=True 

    return nh

def ont2dict(ontfilename,debug=False,lowlouvain=False,merge=False) : 
    g2ns=dict() ;
    nh=dict() ;
    ogenes=set()
    with open(ontfilename) as f: 
        for line in f : 
            ls=line.strip().split('\t')
            protochild=ls[1]
            if ls[-1] == 'gene' or ls[-2] == 'gene' :
                child=s2eid.get(protochild)
                if child is None : continue
                # add gene to list of genes of interest
                ogenes.add(child)
            else : 
                child=protochild

            # index system to gene
            g2ns.update({ child : g2ns.get(child,set()) | {ls[0],}})
            # and parent to all its children
            nh.update({ ls[0] : nh.get(ls[0],set()) | {child,}})

    nhkeys=set(nh.keys())
    changes_made=True
    x=0
    while changes_made : 
        x+=1
        #print(x) ;
        changes_made=False
        for k in nhkeys: 
            for cs in (nh[k] & nhkeys) :
                nh.update({ k : (nh[k] | nh[cs])-{cs,} })
                changes_made=True
                
    if debug : print(len(nh))
    if debug : print(sum([ len(nh[k]) for k in nh.keys() ])/len(nh))
    
    #print(len(nh))
    
    while len(nhkeys) > 0 : 
        nhk=nhkeys.pop()
        isconvertible=False
        try : int(nhk) ; isconvertible=True
        except ValueError : isconvertible=False
        if isconvertible : 
            values=nh.pop(nhk)
            nh.update({'Sys'+nhk : values })
            
    #print(len(nh))
    
    if lowlouvain : 
        for nhk in list(nh.keys()) : 
            if 'Louv' in nhk and len(nh[nhk]) < 20 : 
                nh.pop(nhk)
            elif len(nh[nhk])< 4 : 
                nh.pop(nhk) 

    return nh

import sys
def msg(*args,**kwargs) : 
    print(*args,**kwargs) ; sys.stdout.flush()
                
                
def jmerge(nh,minsimilarity,debug=False) : 
    # find jaccard similarities
    snames=list(nh.keys())
    jgrid=np.zeros((len(snames),len(snames)))
    pcgrid=np.zeros((len(snames),len(snames)))
    sn0s=list(nh.keys())
    i=0

    if debug : msg('Instantiated jgrid.')
    while len(sn0s) > 0 :
        #if debug : msg(f"{len(sn0s)} systems remain.")
        sn0=sn0s.pop(0)
        sn1s=list(sn0s)
        jgrid[i,i]=1.0
        j=i+1
        while len(sn1s)>0 :
            sn1=sn1s.pop()
            jacc=len(nh[sn0]&nh[sn1])/len(nh[sn0]|nh[sn1])
            jgrid[i,j]=jacc
            jgrid[j,i]=jacc

            if len(nh[sn0]-nh[sn1]) == 0 or len(nh[sn1]-nh[sn0]) == 0  : 
                pcgrid[i,j]=1
                pcgrid[j,i]=1

            j+= 1
        i+=1

    if debug : msg('Identified high-similarity non-parent relationships.')


    goodrel_indices=np.argwhere((jgrid>=minsimilarity)&(~(pcgrid!=0))) 

    import networkx as nx
    G=nx.from_edgelist(goodrel_indices)
    ccs=list(nx.connected_components(G))
    if debug : msg('Found connected components')

    from functools import reduce

    if debug : msg('Rebuilding dictionary...',end='')
    newh=dict()
    keys_to_exclude=set()
    for x,cc in enumerate(ccs) : 

        newh.update({ 'jmerge'+str(x) : reduce(set.union,map(lambda c : nh.get(snames[c]),cc))})
        keys_to_exclude |= set(cc) 

    if debug : msg(f"Done.\n Reduced from {len(nh)} to {len(newh)} systems.")

    newh.update({ nhk : nh[nhk] for nhk in set(nh.keys()) - keys_to_exclude})

    return newh

