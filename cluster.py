#!/usr/bin/env python3

import argparse
import numpy as np
import sklearn.cluster
import datetime
from my_util import *

class KMeansData(object):
	def __init__(self, labels_):
		self.labels_ = labels_
		self.memberships = {} # maps label idx to list of datapt indices
		for idx, lbl in enumerate(labels_):
			if lbl not in self.memberships:
				self.memberships[lbl] = []
			self.memberships[lbl].append(idx)

	# Look up cluster id for given data pt. Then return list of members of that cluster.
	def get_neighbour_idxs(self, pt_idx):
		cluster_id = self.labels_[pt_idx]
		return self.memberships[cluster_id]

######################################################################


# Select dimensions. Run K-means. Return KMeansData
def cluster(vecsfile, valsfile, dim, k):
	vecs = np.load(vecsfile)
	vals = np.load(valsfile)
	# Sort & selected dimensions
	i = np.argsort(vals)
	dims = i[:dim]
	vecs = vecs[:,dims]
	# K-means clustering
	tic = datetime.datetime.now()
	kmeans = sklearn.cluster.KMeans(k).fit(vecs)
	toc = datetime.datetime.now(); print('tictoc kmeans', toc-tic)
	# Save centroids
	np.save('centroids-%d-%d-%s' % (k, dim, vecsfile), kmeans.cluster_centers_)
	# Save labels
	np.save('labels-%d-%d-%s' % (k, dim, vecsfile), kmeans.labels_)
	# Return
	return KMeansData(kmeans.labels_)

def test_pt(true_lbl, mean_rating, mov_id, usr_id, uidmap, mov_km, usr_km):
	mov_idx = get_mov_idx(mov_id)
	usr_idx = get_usr_idx(uidmap, usr_id)
	# Does user have ratings for any movies in the movie cluster?
	neighbour_mov_idxs = mov_km.get_neighbour_idxs(mov_idx)
	if 0 == len(neighbour_mov_idxs):
		usr_mean = mean_rating
	else:
		ratings_from_usr = np.nonzero(ratings[usr_idx, neighbour_mov_idxs])[0] # zeros are not ratings; they are unknown values
		usr_mean = ratings_from_usr.mean()
	# Does movie have ratings from any users in the user cluster?
	neighbour_usr_idxs = usr_km.get_neighbour_idxs(usr_idx)
	if 0 == len(neighbour_usr_idxs):
		mov_mean = mean_rating
	else:
		ratings_for_mov = np.nonzero(ratings[neighbour_user_idxs, mov_idx])[0] # zeros are not ratings; they are unknown values
		mov_mean = ratings_for_mov.mean()
	# Check accuracy on user_mean, movie_mean, and the average of the two
	usr_hyp  = math.round(usr_mean)
	mov_hyp  = math.round(mov_mean)
	mean_hyp = math.round((mov_mean + usr_mean) / 2.)
	# Return
	if true_lbl is None:
		return usr_hyp, mov_hyp, mean_hyp
	else:
		return int(usr_hyp == true_lbl), int(mov_hyp == true_lbl), int(mean_hyp == true_lbl)

def test(title, dataset, mov_km, usr_km):
	uidmap = construct_user_id_map()
	ratings = [row[2] for row in dataset]
	mean_rating = np.mean(ratings)
	acc = np.zeros((len(dataset), 3), np.short)
	for i, row in enumerate(dataset):
		mov_id, usr_id, rating, date = row
		usr, mov, mean = test_pt(true_lbl, mean_rating, mov_id, usr_id, uidmap, mov_km, usr_km)
		acc[i,:] = usr, mov, mean
	acc_usr, acc_mov, acc_mean = acc.sum(axis=0)
	print('%s : usr %7d : mov %7d : mean %7d' % (title, acc_usr, acc_mov, acc_mean))

def infer(dataset, mov_km, usr_km):


if __name__ == '__main__':
	# Parse args
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--k-usrs', default=700, type=int)
	parser.add_argument('-m', '--k-movs', default=10000, type=int)
	parser.add_argument('--usr-eigenvectors-file', '--uvecs', default='vecs-user-graph-128.npy')
	parser.add_argument('--mov-eigenvectors-file', '--mvecs', default='vecs-movie-graph-128.npy')
	parser.add_argument('--usr-eigenvalues-file', '--uvals', default='vals-user-graph-128.npy')
	parser.add_argument('--mov-eigenvalues-file', '--mvals', default='vals-movie-graph-128.npy')
	parser.add_argument('--usr-dim', '--du', default=150, type=int) # fix these
	parser.add_argument('--mov-dim', '--dm', default=200, type=int) # fix these
	args = parser.parse_args()

	# Cluster
	usr_km = cluster(args.usr_eigenvectors_file, args.usr_eigenvalues_file, args.usr_dim, args.k_usrs)
	mov_km = cluster(args.mov_eigenvectors_file, args.mov_eigenvalues_file, args.mov_dim, args.k_movs)

	# Load trainset for inference
	tic = datetime.datetime.now()
	with open('trainset.pkl', 'rb') as fin:
		trainset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load trainset', toc-tic)
	# Load testset for inference
	tic = datetime.datetime.now()
	with open('testset.pkl', 'rb') as fin:
		testset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load testset', toc-tic)

	# Test on training data
	test('TEST k-usr %6d k-mov %6d d-usr %4d d-mov %4d', trainset, mov_km, usr_km)

	# Infer on testing data
	infer(testset, mov_km, usr_km)
