import numpy as np 

from openTSNE import TSNEEmbedding, affinity, initialization

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from math import dist

import time


def scaleX(X):
	Xs = X - np.mean(X, 0)
	return Xs/np.linalg.norm(Xs)


def alignment_findCenters(X, labels, sharedCellTypes):
	'''
	ARGS
	----
	- X -> sample
	- labels -> integer-encoded cell-type labels
	- sharedCellTypes -> cell-types to keep

	Find the center of each cell-type, then keeps only those in sharedCellTypes

	returns parsed cell-type centers
	'''

	nCellTypes = int(np.max(labels)+1)
	centers = np.zeros((nCellTypes, X.shape[1]))
	nCells = np.zeros((nCellTypes,))
	for i in range(X.shape[0]):
		l = int(labels[i])
		centers[l] += X[i]
		nCells[l] += 1

	parsedCenters = []
	for i in range(centers.shape[0]):
		if i in sharedCellTypes:
			parsedCenters.append(centers[i]/nCells[i])

	parsedCenters = np.vstack(parsedCenters)

	return parsedCenters


def TSNE(X, Yinit=None, centers=None, labels=None, perplexity=100, K_star=2, max_iter=750, early_exag=4, init_moment=0.5, final_moment=0.8, switch_iter=250, force=0.25, n_jobs=-1):
	'''
	ARGS
	----
	- X -> original, high dimensional data
	- Yinit -> initial embedding. If None, initialization is performed
	- centers -> centers in embedding space of each label
	- labels -> cell type/other cluster info for each cell
	- perplexity -> TSNE perplexity. If float/int, use static perplexity
									 If list, use Multiscale
	- max_iter -> number of embedding iterations
	- early_exag -> exaggeration during early embedding
	- init_moment -> initial momentum
	- final_moment -> momentum after switching
	- switch_iter -> when to stop exaggeration and switch momentums
	- force -> force term for pulling cluster centers to a reference
	- n_jobs -> number of cpus to use

	X and labels should have the same number of rows

	Run an embedding

	RETURNS
	-------
	- Y -> completed TSNE embedding
	'''

	print('  size of X: ', X.shape)

	if perplexity > 0:
		P = affinity.PerplexityBasedNN(X, perplexity=perplexity)
	elif perplexity == 0:
		'''
		based on https://github.com/cdebodt/Multi-scale_t-SNE/blob/main/mstSNE.py#L875
		'''
		N = X.shape[0]
		L_min = 5
		L_max = int(round(np.log2(N/K_star)))
		L = L_max - L_min + 1
		perplexities = 2**np.linspace(L_min-1, L_max-1, L)*K_star
		P = affinity.Multiscale(X, perplexities)
	else:
		print()
		print('Perplexity must be >= 0')
		print()
		exit('Goodbye')

	if Yinit is not None:
		Y = TSNEEmbedding(Yinit, P)
	else:
		Yinit = initialization.pca(X)
		Y = TSNEEmbedding(Yinit, P)

	'''
	if using cluster center-based alignment, determine numbers of cells of each type
	'''
	if centers is not None:
		nCells = np.zeros((centers.shape[0],))
		for i in range(len(labels)):
			l = int(labels[i])
			nCells[l] += 1
		classMatrix = createClassMatrix(centers, labels)

	start = time.time()
	for itr in range(max_iter):
		if itr < switch_iter:
			momentum = init_moment
			exag = early_exag
		else:
			momentum = final_moment
			exag = 1.0

		Y = Y.optimize(n_iter=1, exaggeration=exag, momentum=momentum, n_jobs=n_jobs)#, learning_rate=learning_rate)

		if centers is not None:
			F = forces_calcForces(centers, Y, classMatrix)
			Y += force*F

		if (itr+1) % 50 == 0:
			stop = time.time()
			print('Iteration: ', itr+1, ' | time per iteration: ', (stop-start)/50)
			start = time.time()

	return Y


def createClassMatrix(centers, labels):
	classes = np.zeros((len(labels), centers.shape[0]))
	for i in range(len(labels)):
		l = int(labels[i])
		classes[i,l] = 1

	return classes


def forces_calcForces(c, Y, classMatrix):
	cY = np.matmul(classMatrix.T, Y)/np.sum(classMatrix, 0).T[:,None]
	cY = np.nan_to_num(cY)

	# create a mask for if sum(c[i]) == 0
	# assume that no center will have coordinates that sum to 0
	# if they do, it's because there is currently no information for that cell-type
	# and it should be excluded from the forces
	m = np.sum(c, 1) != 0

	dx = c - cY
	dx = dx*m[:,None]
	dx = dx/np.linalg.norm(dx, axis=1).T[:,None]
	dx = np.nan_to_num(dx)

	return np.matmul(classMatrix, dx)


def getClusters(Xr, k):
	kmeans = KMeans(n_clusters=k).fit(Xr)
	return kmeans.labels_, kmeans.cluster_centers_

def getClusterCenters(Xr, labels, centers):
	nClusters = centers.shape[0]
	centerPoints = np.zeros((centers.shape[0],))
	distances = np.zeros((centers.shape[0],)) + 1e6
	for i in range(Xr.shape[0]):
		cl = labels[i]
		d = dist(Xr[i], centers[cl])
		if d < distances[cl]:
			distances[cl] = d 
			centerPoints[cl] = i

	return centerPoints

def getMNN(Xr, Xs, centerPoints):
	centersR = Xr[centerPoints.astype(int)]

	nbrs_r = NearestNeighbors(n_neighbors=1).fit(centersR)
	_, idr = nbrs_r.kneighbors(Xs)

	nbrs_s = NearestNeighbors(n_neighbors=1).fit(Xs)
	_, ids = nbrs_s.kneighbors(centersR)

	nn_s_r = []
	nn_r_s = []

	for i in range(idr.shape[0]):
		snn = idr[i,:]
		points = []
		for j in range(snn.shape[0]):
			rnn = ids[snn[j]].tolist()
			if i in rnn:
				points.append(snn[j])
		nn_s_r.append(points)

	for i in range(ids.shape[0]):
		rnn = ids[i,:]
		points = []
		for j in range(rnn.shape[0]):
			snn = idr[rnn[j]].tolist()
			if i in snn:
				points.append(rnn[j])
		nn_r_s.append(points)

	return np.vstack(nn_r_s).flatten()

def assignClusters(Xs, mnn):
	centerPoints = Xs[mnn]
	kmeans = KMeans().fit(Xs)
	kmeans.cluster_centers_ = centerPoints
	labels = kmeans.predict(Xs)

	return labels

