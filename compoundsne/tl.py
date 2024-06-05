import numpy as np 
from scipy.linalg import orthogonal_procrustes
from .utils import *

def determineReferences(adata, reference_list=None):
	'''
	ARGS
	----
	- labels -> class labels (ideally cell types) for each sample

	Determines the total number of classes and find the least number of samples in order to 
	have a list of reference samples that contain all classes

	RETURNS
	-------
	- indices of reference patients
	'''

	batch_obs = adata.uns['alignment_params']['batch_obs']
	batches = list(set(adata.obs[batch_obs].tolist()))

	if not reference_list:
		labels = []
		for b in batches:
			labels.append(adata[adata.obs[batch_obs]==b].obs['annotation_encoded'].tolist())

		# determin total number of classes
		totalClasses = len(list(set.union(*map(set, labels))))

		# get unique labels and number of unique labels for each sample
		classes = [list(set(l)) for l in labels]
		nClasses = [len(c) for c in classes]

		# determine which sample has the most unique lables
		# if multiple samples have the most, np.argmax defaults to the first in the list
		# the chosen sample is placed first in the reference list
		references = [np.argmax(nClasses)]

		# get the number of classes in the primary reference
		referenceClasses = nClasses[references[0]]

		# while the total number of classes in the reference list < total number of classes
		# add references in order to obtain all classes
		while referenceClasses < totalClasses:
			# get current classes and number of classes in references
			rc = [classes[r] for r in references]
			allRef = list(set.union(*map(set, rc)))
			newTotals = np.zeros((len(classes),))

			# for each sample not currently in the reference list
			for i in range(len(classes)):
				if i not in references:
					# determine how many classes would be in the reference list if
					# that sample was added to the reference list
					c = classes[i]
					combined = list(set.union(*map(set, [allRef, c])))
					newTotals[i] = len(combined)

			# choose whichever sample would add the most classes to the reference list
			newRef = np.argmax(newTotals)
			references.append(newRef)
			rc.append(classes[newRef])
			referenceClasses = len(list(set.union(*map(set, rc))))

		reference_names = [batches[i] for i in references]
		adata.uns['references'] = reference_names
	else:
		adata.uns['references'] = reference_list


def primaryAlignment(adata):
	batch_obs = adata.uns['alignment_params']['batch_obs']
	batches = list(set(adata.obs[batch_obs].tolist()))
	labels = []
	for b in batches:
		labels.append(adata[adata.obs[batch_obs]==b].obs['annotation_encoded'].tolist())

	adata.obsm['X_pca_transformed'] = adata.obsm['X_pca'].copy()
	for b in batches:
		adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==b] = scaleX(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==b])

	batch_r = adata.uns['references'][0]
	labels_r = adata.obs['annotation_encoded'][adata.obs[batch_obs]==batch_r].tolist()
	for b in batches:
		labels_b = adata.obs['annotation_encoded'][adata.obs[batch_obs]==b].tolist()
		sharedCellTypes = list(set.intersection(*map(set, list([labels_r, labels_b]))))
		
		c_r = alignment_findCenters(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==batch_r], labels_r, sharedCellTypes)
		c_b = alignment_findCenters(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==b], labels_b, sharedCellTypes)

		R, s = orthogonal_procrustes(c_r, c_b)
		adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==b] = np.dot(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==b], R.T)*s

	adata.obsm['Y_init'] = adata.obsm['X_pca_transformed'][:,:2]
	adata.obsm['X_tsne_primary_alignment'] = np.zeros((adata.shape[0],2))
	adata.obsm['X_tsne_full_alignment'] = np.zeros((adata.shape[0],2))


def embedReferences(adata):
	batch_obs = adata.uns['alignment_params']['batch_obs']
	perplexity = adata.uns['alignment_params']['perplexity']
	force = adata.uns['alignment_params']['force']
	n_jobs = adata.uns['alignment_params']['n_jobs']

	nCellTypes = int(np.max(adata.obs['annotation_encoded'].tolist())+1) # assuming proper encoding, cell types are [0, 1, 2, ..., n_types-1]
	centers = np.zeros((nCellTypes, 2))	
	for r in adata.uns['references']:
		Y = TSNE(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==r], 
				 Yinit=adata.obsm['Y_init'][adata.obs[batch_obs]==r], 
				 centers=centers, 
				 labels=adata.obs['annotation_encoded'][adata.obs[batch_obs]==r].tolist(), 
				 perplexity=perplexity, force=force,
				 n_jobs=n_jobs)

		adata.obsm['X_tsne_full_alignment'][adata.obs[batch_obs]==r] = np.array(Y)
		centers_prev = centers.copy()
		nCells = np.zeros((centers.shape[0],))
		for i in range(Y.shape[0]):
			l = int(adata.obs['annotation_encoded'][adata.obs[batch_obs]==r][i])
			if np.sum(centers_prev[l]) == 0:
				centers[l] += Y[i]
				nCells[l] += 1

		for i in range(centers.shape[0]):
			if nCells[i] > 0:
				centers[i] = centers[i]/nCells[i]

	adata.uns['embedding_centers'] = centers


def embedIndependent(adata, sample):
	batch_obs = adata.uns['alignment_params']['batch_obs']
	perplexity = adata.uns['alignment_params']['perplexity']
	n_jobs = adata.uns['alignment_params']['n_jobs']

	Y = TSNE(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==sample],
			 perplexity=perplexity, force=0.0, n_jobs=n_jobs)
	adata.obsm['X_tsne_independent'][adata.obs[batch_obs]==sample] = np.array(Y)
	


def embedSample(adata, sample, alignment):
	batch_obs = adata.uns['alignment_params']['batch_obs']
	centers = adata.uns['embedding_centers']
	perplexity = adata.uns['alignment_params']['perplexity']
	force = adata.uns['alignment_params']['force']
	n_jobs = adata.uns['alignment_params']['n_jobs']
	if alignment == 'primary':
		Y = TSNE(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==sample], 
				 Yinit=adata.obsm['Y_init'][adata.obs[batch_obs]==sample], 
				 centers=centers, 
				 labels=adata.obs['annotation_encoded'][adata.obs[batch_obs]==sample], 
				 perplexity=perplexity, force=0.0,
				 n_jobs=n_jobs)
		adata.obsm['X_tsne_primary_alignment'][adata.obs[batch_obs]==sample] = np.array(Y)
	elif alignment == 'full':
		Y = TSNE(adata.obsm['X_pca_transformed'][adata.obs[batch_obs]==sample], 
				 Yinit=adata.obsm['Y_init'][adata.obs[batch_obs]==sample], 
				 centers=centers, 
				 labels=adata.obs['annotation_encoded'][adata.obs[batch_obs]==sample], 
				 perplexity=perplexity, force=force)
		adata.obsm['X_tsne_full_alignment'][adata.obs[batch_obs]==sample] = np.array(Y)
	else:
		print()
		print("use only 'full' or 'primary'")
		print()
		exit()
