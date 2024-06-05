import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from .utils import *

def setParameters(adata, batch_obs, annotations=None, perplexity=100, force=0.25, n_jobs=2):
	params = {'batch_obs': batch_obs,
			  'perplexity': perplexity,
			  'force': force,
			  'annotations': annotations,
			  'n_jobs': n_jobs}

	adata.uns['alignment_params'] = params

	adata.obsm['X_tsne_independent'] = np.zeros((adata.X.shape[0], 2))


def generateAnnotations(adata, n_clusters, reference_name):
	batch_obs = adata.uns['alignment_params']['batch_obs']

	batches = list(set(adata.obs[batch_obs]))
	labels_r, centers = getClusters(adata[adata.obs[batch_obs]==reference_name].obsm['X_pca'], n_clusters)
	centerPoints = getClusterCenters(adata[adata.obs[batch_obs]==reference_name].obsm['X_pca'], labels_r, centers)

	labels = []
	for b in batches:
		if b == reference_name:
			labels.append(labels_r)
		else:
			mnn = getMNN(adata[adata.obs[batch_obs]==reference_name].obsm['X_pca'],
						 adata[adata.obs[batch_obs]==b].obsm['X_pca'], 
						 centerPoints)

			l = assignClusters(adata[adata.obs[batch_obs]==b].obsm['X_pca'], 
							   mnn)
			labels.append(l)

	annotations = np.zeros((adata.shape[0]))
	for (b, l) in zip(batches, labels):
		idx = 0
		max_idx = len(l)
		for i in range(annotations.shape[0]):
			if adata.obs[batch_obs][i] == b:
				annotations[i] = l[idx]
				idx += 1
			if idx > max_idx:
				exit('I think theres an issue with kmeans clustering')

	adata.obs['annotation'] = annotations
	adata.uns['alignment_params']['annotations'] = 'annotation'


def encodeAnnotations(adata):
	annotation = adata.uns['alignment_params']['annotations']
	if not annotation:
		exit('No annotations provided, run generateAnnotations()')

	le = LabelEncoder()
	le.fit(adata.obs[annotation])
	adata.obs['annotation_encoded'] = pd.Categorical(le.transform(adata.obs[annotation]))

