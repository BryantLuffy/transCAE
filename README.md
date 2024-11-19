# transCAE: Enhancing cell type annotation in single-cell RNA-seq data with transfer learning and convolutional autoencoder
![image text](https://github.com/BryantLuffy/transCAE/blob/master/Overview%20plot.jpg)

## Description
Single-cell RNA-seq	(scRNA-seq) analysis holds significant potential for addressing various biological questions, with one key application being the annotation of query datasets with unknown cell types using well-annotated external reference datasets. However, the performance of existing supervised or semi-supervised methods largely depends on the quality of source data. Furthermore, these methods often struggle with the batch effects arising from different platforms when handling multiple reference or query datasets, making precise annotation challenging. 

We developed transCAE, a powerful transfer learning-based algorithm for single-cell annotation that integrates unsupervised dimensionality reduction with supervised cell type classification. This approach fully leverages information from both reference and query datasets to achieve precise cell classification within the query data. Through extensive evaluation, we demonstrate that transCAE significantly improves cell classification accuracy. Additionally, transCAE effectively mitigates batch effects, showing clear advantages over other state-of-the-art methods in experiments involving multiple reference or query datasets. This positions transCAE as the optimal annotation method for scRNA-seq datasets.


## Data Acquisition
| Dataset             | Organism | Tissue    | Platform       | Reference                                                                                                  |
|---------------------|----------|-----------|----------------|--------------------------------------------------------------------------------------------------------------------------|
| Baron et al.        | Human    | Pancreas  | InDrop         | Baron M, Veres A, Wolock SL, Faust AL, Gaujoux R, Vetere A, et al. A single-cell transcriptomic map of the human and mouse pancreas reveals inter-and intra-cell population structure. Cell systems. 2016;3:346-60. e4. |
| Lawlor et al.       | Human    | Pancreas  | Fluidigm C1    | Lawlor N, George J, Bolisetty M, Kursawe R, Sun L, Sivakamasundari V, et al. Single-cell transcriptomes identify human islet cell signatures and reveal cell-type–specific expression changes in type 2 diabetes. Genome research. 2017;27:208-22. |
| Grün et al.         | Human    | Pancreas  | CelSeq         | Grün D, Muraro MJ, Boisset J-C, Wiebrands K, Lyubimova A, Dharmadhikari G, et al. De novo prediction of stem cell identity using single-cell transcriptome data. Cell stem cell. 2016;19:266-77. |
| Segerstolpe et al.  | Human    | Pancreas  | Smart-seq2     | Segerstolpe Å, Palasantza A, Eliasson P, Andersson E-M, Andréasson A-C, Sun X, et al. Single-cell transcriptome profiling of human pancreatic islets in health and type 2 diabetes. Cell metabolism. 2016;24:593-607. |
| Muraro et al.       | Human    | Pancreas  | CelSeq2        | Muraro MJ, Dharmadhikari G, Grün D, Groen N, Dielen T, Jansen E, et al. A single-cell transcriptome atlas of the human pancreas. Cell systems. 2016;3:385-94. e3. |
| Wang et al.         | Human    | Pancreas  | Fluidigm C1    | Wang YJ, Schug J, Won K-J, Liu C, Naji A, Avrahami D, et al. Single-Cell Transcriptomics of the Human Endocrine Pancreas. Diabetes. 2016;65:3028-38.                                       |
| Han et al.          | Mouse    | Thymus    | Microwell-seq  | Han X, Wang R, Zhou Y, Fei L, Sun H, Lai S, et al. Mapping the mouse cell atlas by microwell-seq. Cell. 2018;172:1091-107. e17. |
| Tabula et al.       | Mouse    | Aorta     | Smart-seq2     | Schaum N, Karkanias J, Neff NF, May AP, Quake SR, Wyss-Coray T, et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris: The Tabula Muris Consortium. Nature. 2018;562:367. |
