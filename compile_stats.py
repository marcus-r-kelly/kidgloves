import numpy as np
import pandas as pd
import sys

target=sys.argv[1]
import multiprocessing as mp
import os

midQ=mp.Queue()
outQ=mp.Queue()

def reader(filename) : 
    weightct=0
    with open(target+'/'+filename,'r') as f :
        initline=f.readline().strip().split(',') ;
        weightct += int(initline[0])
        counts =np.array(initline[1:]).astype(float)
        for x,line in enumerate(f):
            linel=line.strip().split(',') ;
            weightct += int(linel[0])
            counts += np.array(linel[1:]).astype(float)
    midQ.put((filename,weightct,counts))
    return
    
class Adder(mp.Process) : 
    def __init__(self,inQ,outQ) : 
        mp.Process.__init__(self) ;
        self.inQ=inQ
        self.outQ=outQ
    def run(self) : 
        counter=0
        alldata=dict()
        totalweight=dict()
        
        while True: 
            print('Awaiting input...',end='\r')
            tritup=self.inQ.get()
            if tritup[0] is None :
                break
            print('{: >80}'.format('Processing file '+tritup[0]+'('+str(counter)+').'),end='\r')
            filename=tritup[0]
            weight=tritup[1]
            data=tritup[2]

            kind=filename.split('_')[-1].split('.')[0]
            if not kind in alldata : 
                alldata.update({ kind : data })
                totalweight.update({ kind : weight })
                counter += 1
                continue

            counter += 1
            alldata[kind] += data
            totalweight.update({ kind : totalweight.get(kind,0) + weight })
            
        for k in alldata.keys() :
            alldata[k]=alldata[k]/(totalweight[k])
            
        self.outQ.put(alldata)
        print('Shutting down',self.name)
        return

adder=Adder(midQ,outQ)
adder.start()

jobs=[ fp for fp in os.listdir(target) if fp.endswith('.txt')]
            
from tqdm.auto import tqdm
import os

with mp.Pool(processes=16) as p :
    #foo=[ x for x in p.imap_unordered(reader,jobs) ]
    foo=[ x for x in p.imap_unordered(reader,jobs) ]
midQ.put((None,))
print('Done reading, closing processes.')
refined=outQ.get()
adder.join() # processes won't join until all associated queues are empty

import pickle
logit_data=pd.read_csv(target+'/logit_data.csv',index_col=0)
with open(target+'/nestmaskdict.pickle','rb') as f : 
    nestmaskdict=pickle.load(f)
    
with open(target+'/logittransformer.pickle','rb') as f : 
    lt=pickle.load(f)
    
systemkeys=list(sorted(nestmaskdict.keys()))
nsystems=len(systemkeys)
events=logit_data.lesion_class.values
nevents=logit_data.shape[0]

gg=refined['gg'].reshape((nevents,nevents))
sg=refined['gs'].reshape((nevents,nsystems))
ss=refined['ss'].reshape((nsystems,nsystems))

dfgg=pd.DataFrame(data=gg,index=events,columns=events)
dfgg.index.name='eventA'
dfgg.columns.name='eventB'

dfsg=pd.DataFrame(data=sg,index=events,columns=systemkeys)
dfsg.index.name='event'
dfsg.columns.name='system'

dfss=pd.DataFrame(data=ss,index=systemkeys,columns=systemkeys)
dfss.index.name='systemA'
dfss.columns.name='systemB'

dfggm=dfgg.reset_index().melt(id_vars='eventA',var_name='eventB',value_name='p').query(' eventA <= eventB')
dfsgm=dfsg.reset_index().melt(id_vars='event',var_name='system',value_name='p')
dfssm=dfss.reset_index().melt(id_vars='systemA',var_name='systemB',value_name='p').query('systemA <= systemB')

dfggm.to_csv(target+'/gg.csv')
dfsgm.to_csv(target+'/sg.csv')
dfssm.to_csv(target+'/ss.csv')

