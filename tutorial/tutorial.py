import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from transCAE import transClust as ic
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc
import tensorflow as tf
from transCAE.preprocessing import *
seed = 42
tf.random.set_seed(seed)

adata_bh = sc.read('Bh.h5ad')
adata_grun = sc.read('grun.h5ad')

clf=ic.transfer_learning_clf()
clf.fit(adata_bh, adata_grun)

pred, prob, celltype_pred=clf.predict()

mapped_predictions = [celltype_pred[str(pr)][0] for pr in pred.cluster]
colors_use=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896','#bec1d4','#bb7784','#4a6fe3','#FFFF00''#111010']

num_celltype=len(clf.adata_grun.obs["celltype"].unique())


correct = 0  # 预测正确的样本数量

for pred, target in zip(mapped_predictions, adata_grun.obs['celltype']):
    if pred == target:
        correct += 1

accuracy = correct / len(mapped_predictions)
print("Accuracy: {:.2%}".format(accuracy))

ari = adjusted_rand_score(adata_grun.obs['celltype'], mapped_predictions)

nmi = normalized_mutual_info_score(adata_grun.obs['celltype'], mapped_predictions)
