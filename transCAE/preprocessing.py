import scanpy as sc
import pandas as pd
import numpy as np
import os
from anndata import AnnData
from scipy.sparse import issparse
from natsort import natsorted
from anndata import read_mtx
from anndata.utils import make_index_unique


def read_10X(data_path, var_names='gene_symbols'):
    adata = read_mtx(data_path + '/matrix.mtx').T
    genes = pd.read_csv(data_path + '/genes.tsv', header=None, sep='\t')
    adata.var['gene_ids'] = genes[0].values
    adata.var['gene_symbols'] = genes[1].values
    assert var_names == 'gene_symbols' or var_names == 'gene_ids', \
        'var_names must be "gene_symbols" or "gene_ids"'
    if var_names == 'gene_symbols':
        var_names = genes[1]
    else:
        var_names = genes[0]
    if not var_names.is_unique:
        var_names = make_index_unique(pd.Index(var_names)).tolist()
        print('var_names are not unique, "make_index_unique" has applied')
    adata.var_names = var_names
    cells = pd.read_csv(data_path + '/barcodes.tsv', header=None, sep='\t')
    adata.obs['barcode'] = cells[0].values
    adata.obs_names = cells[0]
    return adata

def change_to_continuous(q):
    y_pred=np.asarray(np.argmax(q,axis=1),dtype=int)
    unique_labels=np.unique(q.argmax(axis=1))
    #turn to continuous clusters label
    test_c={}
    for ind, i in enumerate(unique_labels):
        test_c[i]=ind
    y_pred=np.asarray([test_c[i] for i in y_pred],dtype=int)
    ##turn to categories
    labels=y_pred.astype('U')
    labels=pd.Categorical(values=labels,categories=natsorted(np.unique(y_pred).astype('U')))
    return labels

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
        
def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
 
def normalize_log1p_scale(adata,units="UMI",n_top_genes=1000):
    if units=="UMI" or units== "CPM":
        sc.pp.normalize_per_cell(adata,counts_per_cell_after=10e4)
    sc.pp.filter_genes_dispersion(adata,n_top_genes=n_top_genes)
    sc.pp.log1p(adata)
    sc.pp.scale(adata,zero_center=True,max_value=6)
                    
#creat DCEC object
def get_xinput(adata):
    if not isinstance(adata,AnnData):
        raise ValueError("adata must be an AnnData object")
    if issparse(adata.X):
        x=adata.X.toarray()
    else:
        x=adata.X
    return x



def first2prob(adata):
    first2ratio=[name for name in adata.uns.key() if str(name).startswith("prob_matrix")]
    for key_ in first2ratio:
        q_pred=adata.uns[key_]
        q_pred_sort=np.sort(q_pred,axis=1)
        y=q_pred_sort[:,-1]/q_pred_sort[:,-2]
        adata["first2ratio_"+str(key_).split("matrix")[1]]=y

def expand_grid(dictionary):
    from itertools import product
    return pd.DataFrame([row for row in product(*dictionary.values())],columns=dictionary.keys())
