import torch
import torch.sparse
import numpy as np

def np_to_sparse_tensor(ndarray) : 
    """
    Converts np array directly to sparse tensor (returned)
    """
    idxs=ndarray.nonzero();
    vals=ndarray[idxs] ;
    idxs=np.stack(idxs,axis=0)
    #print(idxs.shape)
    return torch.sparse_coo_tensor(indices=idxs,values=vals,size=ndarray.shape).coalesce()

def identity(shape) : 
    return torch.sparse_coo_tensor(indices=np.c_[np.arange(shape),np.arange(shape)].transpose(),values=np.ones((shape,)),size=(shape,shape))


def flatten_majorly(thetensor) :
    """
    Flattens sparse tensor thetensor maintaining the first (=0th) axis and flattening the remainder
    """

    use_indices=thetensor.indices().numpy()
    use_shape=thetensor.shape

    outsize=(use_shape[0],np.prod(use_shape[1:]))
    ri=np.ravel_multi_index(use_indices[1:],dims=use_shape[1:])
    maj_i=use_indices[0]

    outindices=np.stack([maj_i,ri],axis=0)

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outsize).coalesce()

def flatten_minorly(thetensor) : 
    """
    Flattens sparse tensor thetensor maintaining the last (-1th) axis and flattening the remainder
    """

    use_indices=thetensor.indices().numpy()
    use_shape=thetensor.shape

    outsize=(np.prod(use_shape[:-1]),use_shape[-1])
    ri=np.ravel_multi_index(use_indices[:-1],dims=use_shape[:-1])
    maj_i=use_indices[-1]

    outindices=np.stack([ri,maj_i],axis=0)

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outsize).coalesce()

def flatten_secondarily(thetensor):
    """
    Flattens sparse tensor thetensor, using the first axis as tile number 
    along the first axis of the output.
    """

    return flatten_majorly(thetensor.transpose(0,1).coalesce())


def sproing(theflattensor,minor_dims,minorly=False) :
    """
    Unflattens a flat tensor so that its second dimension (=1st) is
    distributed among the dimensions listed in minor_dims.
    ^^ If "minorly", otherwise so that first dimension is distributed among minor dims.
    """

    if minorly : 
        inindices=theflattensor.indices().numpy()
        major=inindices[0]
        rest=inindices[1]

        #print(theflattensor.shape[0]*np.prod(minor_dims),np.prod(theflattensor.shape))
        assert theflattensor.shape[0]*np.prod(minor_dims)==np.prod(theflattensor.shape)

        sproinged_rest=np.unravel_index(rest,shape=minor_dims)

        outindices=np.stack([major,*sproinged_rest],axis=0)
        outsize=(theflattensor.shape[0],*minor_dims)

    else : 
        inindices=theflattensor.indices().numpy()
        major=inindices[1]
        rest=inindices[0]

        #print(theflattensor.shape[0]*np.prod(minor_dims),np.prod(theflattensor.shape))
        assert theflattensor.shape[1]*np.prod(minor_dims)==np.prod(theflattensor.shape)

        sproinged_rest=np.unravel_index(rest,shape=minor_dims)

        outindices=np.stack([*sproinged_rest,major],axis=0)
        outsize=(*minor_dims,theflattensor.shape[1])
    
    return torch.sparse_coo_tensor(indices=outindices,
        values=theflattensor.values(),
        size=outsize).coalesce()

def rotate(thetensor) : 
    in_indices=thetensor.indices().numpy()
    use_indices=np.stack([*in_indices[1:],in_indices[0]],axis=0)
    return torch.sparse_coo_tensor(
            indices=use_indices,
            values=thetensor.values(),
            size=(*thetensor.shape[1:],thetensor.shape[0])).coalesce()

def antirotate(thetensor) :

    in_indices=thetensor.indices().numpy()
    use_indices=np.stack([in_indices[-1],*in_indices[:-1]],axis=0)
    return torch.sparse_coo_tensor(
            indices=use_indices,
            values=thetensor.values(),
            size=(thetensor.shape[-1],*thetensor.shape[:-1])).coalesce()
    

def sw(thetensor,a1,a2) :

    in_indices=thetensor.indices().numpy()
    use_indices=in_indices.copy()
    use_indices[[a2,a1]]=use_indices[[a1,a2]]
    use_shape=np.array(thetensor.shape)
    use_shape[[a2,a1]]=use_shape[[a1,a2]]

    return torch.sparse_coo_tensor(indices=use_indices,values=thetensor.values(),size=tuple(use_shape)).coalesce()
    
def diag_embed(thetensor,minorly=False) :

    if not minorly :
        use_indices=thetensor.indices().numpy()
        outindices=np.stack([use_indices[0],*use_indices],axis=0)
        outshape=(thetensor.shape[0],*tuple(thetensor.shape))
    else : 
        use_indices=thetensor.indices().numpy()
        outindices=np.stack([*use_indices,use_indices[-1]],axis=0)
        outshape=(*tuple(thetensor.shape),thetensor.shape[-1],)

    #print(outindices.shape)
    #print(outshape)
    

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outshape).coalesce()

def clip(thetensor) : 

    thevals=np.clip(thetensor.values(),0,1)

    return torch.sparse_coo_tensor(indices=thetensor.indices().numpy().copy(),
                                   values=thevals,size=tuple(thetensor.shape)).coalesce()

def as_directsum(theflattensor,n_blocks,minorly=False) :

    if minorly : 
        # viz. preserving the minor dimension as is
        assert theflattensor.shape[0] % n_blocks ==0 

        use_indices=theflattensor.indices().numpy().copy()
        #n_blocks=theflattensor.shape[-1]//blocksize
        blocksize=theflattensor.shape[-1]//n_blocks
        out_shape=(theflattensor.shape[0]*n_blocks,theflattensor.shape[1])

        use_indices[0]=theflattensor.shape[0]*(use_indices[1]//blocksize)+use_indices[0]
    else : 
        key_modulus=theflattensor.shape[0] % n_blocks
        if not ( key_modulus ==0 ) : 
            raise ValueError('flat tensor shape incopatible with n blocks; ',theflattensor.shape,n_blocks,key_modulus)



        use_indices=theflattensor.indices().numpy().copy()
        blocksize=theflattensor.shape[0]//n_blocks
        out_shape=(theflattensor.shape[0],theflattensor.shape[1]*n_blocks)

        use_indices[-1]=theflattensor.shape[-1]*(use_indices[0]//blocksize)+use_indices[1]

    return torch.sparse_coo_tensor(
                indices=use_indices,
                values=theflattensor.values().numpy().copy(),
                size=out_shape).coalesce() ;

#def revert_directsum(theds,rowblocksize,columnblocksize) : 

def sparse_tensor_from_index_array_iter(sparseiter,dense_shape_of_slice,verbose=False) : 
    thearr=None
    for i,si in enumerate(sparseiter) : 
        if verbose : print('{:0>8}'.format(i),end='\r')
        if thearr is None : 
            thearr=np.r_[np.ones((1,si.shape[1]))*i,si]
        else :
            thearr=np.c_[thearr,
                         np.r_[np.ones((1,si.shape[1]))*i,si]
                        ]

        #print()
        
    #thearr=thearr.astype(np.int32)
    outtensor=torch.sparse_coo_tensor(indices=thearr[:3,:],values=thearr[-1,:],size=(i+1,*dense_shape_of_slice))
    return outtensor.coalesce()

def flat_sparse_tensor_tile(theflattensor,times_to_repeat) : 
    use_indices=theflattensor.indices().numpy().copy()
    use_values=theflattensor.values()

    #out_indices=np.concatenate([ np.stack([use_indices[0,:]+j*theflattensor.shape[0],use_indices[1,:]],axis=0)
                           #for j in range(times_to_repeat) ],axis=1)
    use_indices_tiled=np.tile(use_indices,(1,times_to_repeat))
    boosts_top=np.tile(np.arange(0,times_to_repeat),(use_indices.shape[1],1)).transpose().ravel()
    boosts_top=boosts_top.reshape(1,len(boosts_top))*theflattensor.shape[0]
    #print(boosts_top.shape)
    boosts=np.concatenate([ boosts_top, np.zeros((use_indices.shape[0]-1,boosts_top.shape[1])) ],axis=0)
    # use arange and np.ones (multiplication thereof) to createa  matrix that is then added to out_indices
    #print(boosts.shape)
    out_indices=use_indices_tiled+boosts
    #print(out_indices.shape)


    out_values=np.ones((out_indices.shape[1],))

    outtensor=torch.sparse_coo_tensor(indices=out_indices,values=out_values,size=(theflattensor.shape[0]*times_to_repeat,theflattensor.shape[1]))

    return outtensor.coalesce()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   flat_ot_nestlike torch.Size([10000, 23])
#   thet torch.Size([200, 100, 50])
#   ds torch.Size([20000, 10000])
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def bmm(mat1,mat2) :
    """
    An admittedly limited copy of torch's bmm modified to use two sparse matrices.
    Satisfies the following use cases : 

    mat1=(k*m*n), mat2=(n*p) ==> (k*m*p)
    mat1=(k*m*n)  mat2=(k*n*p) ==> (k*m*p)
    mat1=(j*k*m*n) mat2=(n*p) ==> (j*k*n*p)
    mat1=(j*k*m*n) mat2=(k*n*p) ==> (j*k*n*p)
    mat1=(j*k*m*n) mat2=(j*k*n*p) ==> (j*k*n*p)
    """
    if not mat1.shape[-1] == mat2.shape[-2] : 
        raise ValueError('Incompatible dimensions between mat1 (',mat1.shape,') and mat2 (',mat2.shape,')')

    lsm1=len(mat1.shape)
    lsm2=len(mat2.shape)

    if not any([ (lsm1,lsm2) == t for t in {(3,2),(3,3),(4,2),(4,3),(4,4)} ]) : 
        raise ValueError('mat1 of shape',mat1.shape,' and mat2 of shape',mat2.shape,'not supported by sptops.bmm')
        
    # common operations for cases 1-2 and 3-5
    if lsm1 == 3 : 
        f1=flatten_minorly(mat1.coalesce())
        op1=as_directsum(f1,n_blocks=mat1.shape[0],minorly=False)

    else : 
        f1=flatten_minorly(mat1.coalesce())
        op1=as_directsum(f1,n_blocks=np.prod(mat1.shape[:-2]))

    if lsm2 == 2  : 
        times_to_multiply=np.prod(mat1.shape[:-2]) # viz. k or j*k times
        op2=flat_sparse_tensor_tile(mat2,times_to_multiply).to(dtype=mat2.dtype).coalesce()
    elif lsm2 ==3 and lsm1==4 : 
        print('warning! not sure I\'m going to test this use case.')
        times_to_multiply=mat1.shape[0] # viz. j
        op2=flat_sparse_tensor_tile(flatten_secondarily(mat2),times_to_multiply)
    elif lsm2 == 3 and lsm1==3 :
        op2=flatten_minorly(mat2).coalesce()
    elif lsm2 == 4 :
        op2=flatten_minorly(flatten_minorly(mat2))


    try : 
        assert op1.shape[-1] == op2.shape[0]
        theprod=torch.matmul(op1,op2).coalesce()
    except Exception as e : 
        print(lsm1,lsm2,mat1.shape,mat2.shape,op1.shape,op2.shape)
        raise e 

    if (lsm1,lsm2) in {(3,2),(3,3)} : 
        # since we are flattening the first tensor "secondarily"
        #print(theprod.shape)
        out=sproing(theprod,tuple(mat1.shape[:-1]))
    if (lsm1,lsm2) in {(4,2),(4,3),(4,4)} : 
        out=sproing(theprod,tuple(mat1.shape[:-1]))

    return out


    
        
    
