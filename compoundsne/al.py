import numpy as np 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from itertools import combinations
import pandas as pd

def calc_cpd(s, r):
	'''
	requires that s and r contain the same points
	'''
	sd = pdist(s[:1000])
	rd = pdist(r[:1000])
	coeff, _ = spearmanr(sd, rd)
	return coeff


def calc_knn(sample, k=10):
	nn = NearestNeighbors(n_neighbors=k+1).fit(sample)
	_, neighbors = nn.kneighbors(sample)
	return neighbors[:,1:]


def calc_knc(sample, labels, sharedLabels, k=5):
	centers = findCenters(sample, labels, sharedLabels)
	return calc_knn(centers, k)


def findCenters(sample, labels, sharedLabels):
	'''
	ARGS
	----
	- sample: embedding of a single sample
	- labels: labels for each cell in that sample
	- sharedLabels: list of labels that are shared between multiple samples

	Find the centers for each cell type in embedding space
	Retains only cell types that are shared between multiple samples

	Returns: cell-type x coordinates array of centers
	'''

	centers = np.zeros((int(np.max(labels)+1), sample.shape[1]))
	nCells = np.zeros((centers.shape[0],))
	for i in range(sample.shape[0]):
		l = int(labels[i])
		centers[l] += sample[i]
		nCells[l] += 1

	shared_centers = []
	for i in range(centers.shape[0]):
		if i in sharedLabels:
			shared_centers.append(centers[i]/nCells[i])

	return np.vstack(shared_centers)


def findSharedLabels(labels_list):
	return list(set.intersection(*map(set, labels_list)))


def compareNeighborhoods(n1, n2):
	'''
	ARGS
	----
	- n1: np.array of neighborhood 1. rows: points/cells
									  columns: idx of neighbors
	- n2: np.array of neighborhood 2

	Determines the fraction of each neighborhood shared between n1 and n2
	The neighbors for each point do NOT have to be in the same order

	Returns: single value for accuracy
	'''
	if n1.shape != n2.shape:
		print()
		print('Neighborhood shapes are not equal')
		print()
		exit('Goodbye')

	acc = []
	for i in range(n1.shape[0]):
		l1 = n1[i].tolist()
		l2 = n2[i].tolist()
		shared = list(set(l1) & set(l2))
		acc.append(len(shared)/len(l1))
	return np.mean(acc)


def structurePreservation(adata, alignments=['X_tsne_full_alignment'], k_knn=10):
	batch_obs = adata.uns['alignment_params']['batch_obs']

	sampleNames = list(set(adata.obs[batch_obs]))
	knn = []
	for align in alignments:
		knnS = [calc_knn(adata[adata.obs[batch_obs]==s].obsm[align], k=k_knn) for s in sampleNames]
		knnR = [calc_knn(adata[adata.obs[batch_obs]==s].obsm['X_tsne_independent'], k=k_knn) for s in sampleNames]
		knn.append([compareNeighborhoods(s, r) for (s,r) in zip(knnS, knnR)])

	stats = []
	alignName = []
	sampleName = []
	for (i, align) in zip(range(len(knn)), alignments):
		for (j, sample) in zip(range(len(knn[i])), sampleNames):
			stats.append(knn[i][j])
			alignName.append(align)
			sampleName.append(sample)

	df = pd.DataFrame()
	df['KNN'] = stats 
	df['Alignment'] = alignName
	df['Sample'] = sampleName

	adata.uns['Structure preservation'] = df


def sampleAlignment(adata, alignments=['X_tsne_full_alignment']):
	batch_obs = adata.uns['alignment_params']['batch_obs']
	sampleNames = list(set(adata.obs[batch_obs]))


	labels = [adata.obs['annotation_encoded'][adata.obs[batch_obs]==s].tolist() for s in sampleNames]
	sharedLabels = list(set.intersection(*map(set, labels)))

	distances = []
	for align in alignments:
		centers = [findCenters(adata[adata.obs[batch_obs]==s].obsm[align],
							   l, sharedLabels) for (s, l) in zip(sampleNames, labels)]
		dists = []
		for c in centers:
			c -= np.mean(c, 0)
			c /= np.linalg.norm(c)

		combos = combinations(np.arange(len(centers)), 2)
		for (i, j) in combos:
			dists.append(1/(1+np.sum(np.square(centers[i] - centers[j]))))
		distances.append(dists)
	distances = np.vstack(distances)

	df = pd.DataFrame(distances.T, columns=alignments)
	df = pd.melt(df, value_vars = alignments)
	df = df.rename(columns={'variable': 'Alignment method', 'value':'Distance'})

	adata.uns['Alignment'] = df
