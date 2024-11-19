from __future__ import division
from time import time
import tensorflow as tf
from . DCEC import DCEC
from . preprocessing import *
import os
from tensorflow.keras.optimizers import SGD
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
from natsort import natsorted


class  transfer_learning_clf(object):
    '''
    The transfer learning clustering and classification model.
    This class has following api: fit(), predict(), Umap(), tSNE()
    '''

    def __init__(self):
        super(transfer_learning_clf, self).__init__()

    def fit(self,
            reference_data,  # adata
            query_data,  # adata
            normalize=True,
            take_log=True,
            scale=True,
            batch_size=256,
            maxiter=1000,
            pretrain_epochs=50,
            epochs_fit=5,
            tol=[0.001],
            alpha=[1.0],
            resolution=[0.2, 0.4, 0.8, 1.2, 1.6],
            filter1 = 16,
            filter2 = 32,
            hidden_dim = 30,
            n_neighbors=20,
            softmax=False,
            init="glorot_uniform",
            save_atr="_trans_True"
            ):
        '''
        Fit the transfer learning model using provided data.
        This function includes preprocessing steps.
        Input: reference_data(anndata format), query_data(anndata format).
        Source and target data can be in any form (UMI or TPM or FPKM)
        Retrun: No return
        '''
        self.batch_size = batch_size
        self.maxiter = maxiter
        self.pretrain_epochs = pretrain_epochs
        self.epochs_fit = epochs_fit
        self.tol = tol
        self.alpha = alpha
        self.reference_data = reference_data
        self.query_data = query_data
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.softmax = softmax
        self.init = init
        self.save_atr = save_atr
        self.filter1 = filter1
        self.filter2 = filter2
        self.hidden_dim = hidden_dim

        reference_data.var_names_make_unique(join="-")
        reference_data.obs_names_make_unique(join="-")
        query_data.var_names_make_unique(join="-")
        query_data.obs_names_make_unique(join="-")

        reference_data.var_names = [i.upper() for i in reference_data.var_names]


        # 1.pre filter cells
        prefilter_cells(reference_data, min_genes=100)
        # 2 pre_filter genes
        prefilter_genes(reference_data, min_cells=10)  # avoiding all gene is zeros
        # 3 prefilter_specialgene: MT and ERCC
        prefilter_specialgenes(reference_data)
        # 4 normalization,var.genes,log1p,scale
        if normalize:
            sc.pp.normalize_per_cell(reference_data)
        # 5 scale
        if take_log:
            sc.pp.log1p(reference_data)
        if scale:
            sc.pp.scale(reference_data, zero_center=True, max_value=6)


        # 1.pre filter cells
        prefilter_cells(query_data, min_genes=100)
        # 2 pre_filter genes
        prefilter_genes(query_data, min_cells=10)  # avoiding all gene is zeros
        # 3 prefilter_specialgene: MT and ERCC
        prefilter_specialgenes(query_data)
        # 4 normalization,var.genes,log1p,scale
        if normalize:
            sc.pp.normalize_per_cell(query_data)

        # select top genes
        tg = 2000
        # select common genes

        common_genes = np.intersect1d(reference_data.var_names, query_data.var_names)
        reference_data = reference_data[:, common_genes].copy()
        query_data = query_data[:, common_genes].copy()
        sc.pp.filter_genes_dispersion(query_data, n_top_genes=3000)
        query_data = query_data[:,query_data.var['dispersions'].sort_values(ascending=False).index[0:tg]]
        if take_log:
            sc.pp.log1p(query_data)
        if scale:
            sc.pp.scale(query_data, zero_center=True, max_value=6)
        #select common variable genes
        variable_genes = np.intersect1d(reference_data.var_names, query_data.var_names)
        adata_test= query_data[:, variable_genes]
        # Update target data using the highly variable genes
        adata_train = reference_data[:, variable_genes]
        if issparse(adata_train.X):
            x_train = adata_train.X.toarray()
        else:
            x_train = adata_train.X
        y_train = pd.Series(adata_train.obs["celltype"], dtype="category")
        y_train = y_train.cat.rename_categories(range(len(y_train.cat.categories)))
        print("The number of training celltypes is: ", len(set(y_train)))
        if issparse(adata_test.X):
            x_test = adata_test.X.toarray()
        else:
            x_test = adata_test.X

        # Training Data DCEC
        print("Training the reference network")
        print(":".join(["The shape of xtrain is", str(x_train.shape[0]), str(x_train.shape[1])]))
        print(":".join(["The shape of xtest is", str(x_test.shape[0]), str(x_test.shape[1])]))
        assert x_train.shape[1] == x_test.shape[1]

        dcec=DCEC(y=y_train,x=x_train,alpha=alpha,init=self.init,
                pretrain_epochs=self.pretrain_epochs,softmax=softmax,
                  filter1 = self.filter1,filter2 = self.filter2,hidden_dim = self.hidden_dim)
        dcec.compile(optimizer=SGD(lr=0.01, momentum=0.9))
        Embeded_z, q_pred = dcec.fit_supervise(x=x_train, y=y_train, epochs=2e3,
                                              batch_size=self.batch_size)  # fine tunning

        # ---------------------------------------------------------------------------------------------------
        weights = [layer.get_weights() for layer in dcec.model.layers]
        features = dcec.encoder.predict(tf.reshape(x_test, [-1, 1, x_test.shape[1], 1]))
        q = dcec.model.predict(tf.reshape(x_test, [-1, 1, x_test.shape[1], 1]), verbose=0)

        # np.savetxt("testq.txt",q)
        print("Training model finished! Start to fit query network!")
        val_y_pre = dcec.model.predict(tf.reshape(x_train, [-1, 1, x_test.shape[1], 1]), verbose=0)
        val_y_pre = [np.argmax(i) for i in val_y_pre]
        dcec2 = DCEC(x=x_test, alpha=alpha, init=self.init, pretrain_epochs=self.pretrain_epochs,
                   softmax=softmax, transfer_feature=features, model_weights=weights,
                   y_trans=q.argmax(axis=1),filter1 = self.filter1,filter2 = self.filter2,hidden_dim = self.hidden_dim)
        dcec2.compile(optimizer=SGD(0.01, 0.9))
        trajectory_z, trajectory_l, Embeded_z, q_pred = dcec2.fit_trajectory(x=x_test, tol=tol,
                                                                            epochs_fit=self.epochs_fit,
                                                                            batch_size=self.batch_size)  # Fine tunning
        print("How many trajectories ", len(trajectory_z))
        for i in range(len(trajectory_z)):
            adata_test.obsm["trajectory_Embeded_z_" + str(i)] = trajectory_z[i]
            adata_test.obs["trajectory_" + str(i)] = trajectory_l[i]

        # labels=change_to_continuous(q_pred)
        y_pred = np.asarray(np.argmax(q_pred, axis=1), dtype=int)
        labels = y_pred.astype('U')
        labels = pd.Categorical(values=labels, categories=natsorted(np.unique(y_pred).astype('U')))

        adata_test.obsm["X_Embeded_z" + str(self.save_atr)] = Embeded_z
        adata_test.obs["dcec" + str(self.save_atr)] = labels
        adata_test.obs["maxprob" + str(self.save_atr)] = q_pred.max(1)
        adata_test.obsm["prob_matrix" + str(self.save_atr)] = q_pred
        adata_test.obsm["X_pcaZ" + str(self.save_atr)] = sc.tl.pca(Embeded_z)

        self.adata_train = adata_train
        self.adata_test = adata_test
        self.dcec2 = dcec2
        self.labels = labels

    def predict(self, save_dir="./results", write=False):
        '''
        Will return clustering prediction(DataFrame),
        clustering probability (DataFrame) and
        celltype assignment confidence score(dictionary).
        If write is True(default), results will also be written provided save_dir
        '''
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Cluster prediction
        pred = {'cell_id': self.adata_test.obs.index.tolist(),
                'cluster': self.adata_test.obs["dcec_trans_True"].tolist()}
        pred = pd.DataFrame(data=pred)
        # Confidence score
        celltype_pred = {}
        source_label = pd.Series(self.adata_train.obs["celltype"], dtype="category")
        source_label = source_label.cat.categories.tolist()
        num_ori_ct = self.adata_test.obsm["prob_matrix" + str(self.save_atr)].shape[1]
        target_label = [str(i) for i in range(num_ori_ct)]
        for i in range(num_ori_ct):
            end_cell = self.adata_test.obs.index[self.adata_test.obs["dcec_trans_True"] == target_label[i]]
            start_cell = self.adata_test.obs.index[self.adata_test.obs["trajectory_0"] == target_label[i]]
            overlap = len(set(end_cell).intersection(set(start_cell)))
            celltype_pred[target_label[i]] = [source_label[i], round(overlap / (len(end_cell) + 0.0001), 3)]

        # Clustering probability
        prob = pd.DataFrame(self.adata_test.obsm["prob_matrix" + str(self.save_atr)])
        prob.index = self.adata_test.obs.index.tolist()
        prob.columns = ["cluster" + str(i) for i in range(len(set(prob.columns)))]
        if write:
            pred.to_csv(save_dir + "/clustering_results.csv")
            prob.to_csv(save_dir + "/clustering_prob.csv")
            f = open(save_dir + "/celltype_assignment.txt", "w")
            for k, v in celltype_pred.items():
                f.write("Cluster " + str(k) + " is " + str(v[1] * 100) + "%" + " to be " + v[0] + " cell\n")

            f.close()
            print("Results are written to ", save_dir)

        return pred, prob, celltype_pred

    def Umap(self):
        '''
        Do Umap.
        Return: the umap projection(DataFrame)
        '''
        print("Doing U-map!")
        sc.pp.neighbors(self.adata_test, n_neighbors=10, use_rep="X_Embeded_z" + str(self.save_atr))
        sc.tl.umap(self.adata_test)
        return self.adata_test.obsm['X_umap']

    def tSNE(self):
        '''
        Do tSNE.
        Return: the umap projection(DataFrame)
        '''
        print("Doing t-SNE!")
        sc.tl.tsne(self.adata_test, use_rep="X_Embeded_z" + str(self.save_atr), learning_rate=150, n_jobs=10)
        return self.adata_test.obsm['X_tsne']