import compoundsne as cs
import anndata as ad 

adata = ad.read_h5ad('sample_data/data.h5ad')

'''
set some basic parameters

cs.pp.setParameters(
adata (required) = adata object containing processed samples. Must contain .obsm['X_pca']
batch_obs (required) = label in .obs that corresponds to samples/batches
annotations (optional, default=None) = label in .obs that corresponds to cell annotations (ie 'celltype')
perplexity (optional, default=100) = t-SNE perplexity
force (optional, default=0.25) = strength of the force term for full alignment
)
'''
cs.pp.setParameters(adata, 'batch', annotations='cell_type')

'''
if annotations are not provided in setParameters, this function must
be run to generate annotations based on KMeans and MNN

cs.pp.generateAnnotations(
adata (required) = adata object
n_clusters (required) = number of clusters to find when generating annotations
reference_name (required) = name of sample in batch_obs (from setParameters) to use as the reference
)
'''
#cs.pp.generateAnnotations(adata, 5, 'BM1')

'''
integer-encode annotations
cs.pp.encodeAnnotations(adata)

'''
cs.pp.encodeAnnotations(adata)

'''
determine primary and secondary references based on the number
of unqiue annotations.

cs.tl.determineReferences(
adata (required) = adata object
reference_list (optional, default=None) = if provided, should be a list of strings corresponding to different sample names
										= if cs.pp.generateAnnotations() is used, the reference sample used there
										  should be used here (placed in a list)
)
'''
cs.tl.determineReferences(adata) #, reference_list=['BM1'])

'''
rotates X_pca (saved in X_pca_transformed) to align all samples to the primary reference
via a procrustes transformation

saves the first two components of X_pca_transformed into .obsm['Y_init']
creates placeholder arrays (np.zeros) for 'X_tsne_primary_alignment' and 'X_tsne_full_alignmnet'

cs.tl.primaryAlignment(adata)
'''
cs.tl.primaryAlignment(adata)

'''
embeds references (starting with the primary and progressing through secondaries)
and finds the annotation centers in embedding space, which are used to align
the remaining samples
'''
cs.tl.embedReferences(adata)

'''
go through the remaining samples and perform tsne with forces to achieve the full
alignment

cs.tl.embedSample(
adata (required) = adata object
sample (required) = name of sample to embed
alignment (required, either 'full' or 'primary') = if 'full': uses Yinit from cs.tl.primaryAlignment()
															  and additional force term to perform the full alignment
													if 'primary': initializes the embedding with Yinit, then runs a normal tsne
)


perform independent embeddings. save to .obsm['X_tsne_independent']

cs.tl.embedIndependent(
adata (required) = adata object
sample (required) = name of sample to embed
)
'''
for s in ['BM1', 'BM2']:
	cs.tl.embedIndependent(adata, s)
	cs.tl.embedSample(adata, s, 'primary')
	if s not in adata.uns['references']:
		cs.tl.embedSample(adata, s, 'full')

'''
get preservation and alignment metrics

cs.al.structurePreservation(
adata (required) = adata object
alignments (optional, default=['X_tsne_full_alignment']) = list of alignment methods to compare to the independent embeddings
k_knn (optional, default=10) = number of nearest neighbors for comparison
)

cs.al.alignment(
adata (required) = adata object
alignments (optional, default=['X_tsne_full_alignment']) = list of alignment methods to compare to the independent embeddings
)
'''

cs.al.structurePreservation(adata, alignments=['X_tsne_primary_alignment', 'X_tsne_full_alignment'])
cs.al.sampleAlignment(adata, alignments=['X_tsne_independent', 'X_tsne_primary_alignment', 'X_tsne_full_alignment'])

adata.write('embeddings.h5ad')

print('done')
