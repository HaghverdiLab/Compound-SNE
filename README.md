# Compound-SNE

Package code for Compound-SNE (https://www.biorxiv.org/content/10.1101/2024.02.29.582536v1). Download compoundsne into working directory.

# Usage
After processing data accordingly, run the following commands to embed with Compound-SNE. Note: data must be saved in an AnnData object containing .obsm['X_pca']. A full example of usage is in example.py.

Set parameters: saves into .uns['alignment_params'] a set of necessary parameters.

```
cs.pp.setParameters(
adata (required) = adata object containing processed samples. Must contain .obsm['X_pca']
batch_obs (required) = label in .obs that corresponds to samples/batches
annotations (optional, default=None) = label in .obs that corresponds to cell annotations (ie 'celltype')
perplexity (optional, default=100) = t-SNE perplexity
force (optional, default=0.25) = strength of the force term for full alignment
)
```

Encode annotations from strings to integers (necessary for alignment to run).

```
cs.pp.encodeAnnotations(adata)
```

Determine references: either input an option list of batches/patients/samples to use as references, starting with the first in the list, or let Compound-SNE determine the references, based on numbers of different cell-types.

```
cs.tl.determineReferences(
adata (required) = adata object
reference_list (optional, default=None) = if provided, should be a list of strings corresponding to different sample names. if cs.pp.generateAnnotations() is used, the reference sample used there should be used here (placed in a list)
)
```

Primary alignment: perform a Procrustes transformation on X_pca for each sample to the first reference. Saved as X_pca_transformed.

```
cs.tl.primaryAlignment(adata)
```

Embed references: starting with the first reference, run t-SNE, finding the cell-type centers in embedding space.

```
cs.tl.embedReferences(adata)
```

Embed a sample independently: perform t-SNE on a sample without any alignment.

```
cs.tl.embedIndependent(
adata (required) = adata object,
sample (required) = name of the sample to embed
)
```

Embed a sample with alignment: perform t-SNE with alignment to reference. A 'primary' alignment embeds with on X_pca_transformed as the embedding initialization. A 'full' alignment embeds with the addition attractive force between cell-type centers in embedding space.

```
cs.tl.embedSample(
adata (required) = adata object,
sample (required) = name of the sample to embed,
alignemnt (required) = 'primary' or 'full'
)
```

Calculate metrics: determine how well a set of embeddings are aligned with each other, and how well aligned embeddings compare to the original embeddings. Alignment metrics are saved as a DataFrame under .uns['Alignment'] and comparison via KNN to the original embeddings is saved under .uns['Structure preservation'].

```
cs.al.structurePreservation(adata, alignments=['X_tsne_primary_alignment', 'X_tsne_full_alignment'])
cs.al.sampleAlignment(adata, alignments=['X_tsne_independent', 'X_tsne_primary_alignment', 'X_tsne_full_alignment'])
```
