#!/usr/bin/env python3

import argparse
import numpy as np
import sklearn.cluster
import os, datetime
import pickle
from my_util import *
from my_load import *

class KMeansData(object):
	def __init__(self, labels_, is_test=False):
		self.labels_ = labels_
		self.is_test = is_test
		self.memberships = {} # maps label idx to list of datapt indices
		for idx, lbl in enumerate(labels_):
			if lbl not in self.memberships:
				self.memberships[lbl] = []
			self.memberships[lbl].append(idx)

	# Look up cluster id for given data pt. Then return list of members of that cluster.
	def get_neighbour_idxs(self, pt_idx):
		if pt_idx is None: return []
		cluster_id = self.labels_[pt_idx]
		members = self.memberships[cluster_id]
		if self.is_test:
			return members
		else:
			return [ m for m in members if m != pt_idx ]

######################################################################

def _labels_fpath(k, dim, vecsfile): return 'labels-%d-%d-%s' % (k, dim, vecsfile)

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
	np.save(_labels_fpath(k, dim, vecsfile), kmeans.labels_)
	# Return
	return KMeansData(kmeans.labels_)

# Ratings is a user-by-movie matrix
def test_pt(true_lbl, mean_rating, ratings, mov_id, usr_id, uidmap, mov_km, usr_km):
	nzs = lambda ndarray : ndarray[ndarray.nonzero()] # Get nonzero values
	mov_idx = get_mov_idx(mov_id)
	usr_idx = get_usr_idx(uidmap, usr_id)
	# Nested function to get...
	# `usr` and `mov` can be scalars or lists of indices
	def selection_mean(usr, mov):
		if (not usr) and (not mov): # One or both was not in training set and so has no cluster
			return mean_rating
		elif not usr: # Either user was not in training set or just has no ratings for the movies cluster
			selection = ratings[:, mov] # Select all ratings for movie (by all users)
		elif not mov: # Either movie was not in training set or just has no ratings from the users cluster
			selection = ratings[usr, :] # Select all ratings by user (for all movies)
		else:
			selection = ratings[usr, mov] # Either (1) all ratings by one user for a cluster of movies or (2) all ratings for one movie from a cluster of users
		nonzero_values = nzs(selection)
		return nonzero_values.mean() if len(selection) else mean_rating
	# Get indices of neighbours
	neighbour_usr_idxs = usr_km.get_neighbour_idxs(usr_idx)
	neighbour_mov_idxs = mov_km.get_neighbour_idxs(mov_idx)
	usr_mean = selection_mean( neighbour_usr_idxs, mov_idx )
	mov_mean = selection_mean( usr_idx, neighbour_mov_idxs )
	# Check accuracy on user_mean, movie_mean, and the average of the two
	usr_hyp  = np.round(usr_mean)
	mov_hyp  = np.round(mov_mean)
	mean_hyp = np.round((mov_mean + usr_mean) / 2.)
	meanmean_hyp = np.round((mov_mean + usr_mean + mean_rating) / 3.)
	# Return
	hyps = (usr_hyp, mov_hyp, mean_hyp, meanmean_hyp)
	if true_lbl is None: # Return inference values
		return hyps
	else:                # Return accuracy
		return (	hyps,
					[ int(h == true_lbl) for h in hyps ],
					[ (h - true_lbl)**2 for h in hyps ]
			)

def validate(title, dataset, mov_km, usr_km):
	uidmap = construct_user_id_map()
	all_ratings = [row[2] for row in dataset]
	mean_rating = np.mean(all_ratings)
	ratings_matrix, _ = user_by_movie_matrix(dataset, uidmap)
	hyps = np.zeros((len(dataset), 4), np.short)
	accs = np.zeros((len(dataset), 4), np.short)
	errs = np.zeros((len(dataset), 4), np.float)
	for i, row in enumerate(dataset):
		mov_id, usr_id, true_lbl, date = row
		hyps[i,:], accs[i,:], errs[i,:] = test_pt(true_lbl, mean_rating, ratings_matrix, mov_id, usr_id, uidmap, mov_km, usr_km)
	n = float(len(dataset))
	n_correct = accs.sum(axis=0)
	accuracy  = n_correct / n
	mses      = errs.mean(axis=0)
	favs      = (accuracy / mses)
	print(title)
	print('  COR : usr %7d : mov %7d : mean %7d : meanmean %7d' % tuple(n_correct))
	print('  ACC : usr %.5f : mov %.5f : mean %.5f : meanmean %.5f' % tuple(accuracy))
	print('  MSE : usr %.5f : mov %.5f : mean %.5f : meanmean %.5f' % tuple(mses))
	print('  FAV : usr %.5f : mov %.5f : mean %.5f : meanmean %.5f' % tuple(favs))

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
	is_test = False
	if os.path.exists(_labels_fpath(args.k_usrs, args.usr_dim, args.usr_eigenvectors_file)):
		usr_km = KMeansData(np.load(_labels_fpath(args.k_usrs, args.usr_dim, args.usr_eigenvectors_file)), is_test)
		print('loaded user kmeans')
	else:
		usr_km = cluster(args.usr_eigenvectors_file, args.usr_eigenvalues_file, args.usr_dim, args.k_usrs)
		print('built user kmeans')
	if os.path.exists(_labels_fpath(args.k_movs, args.mov_dim, args.mov_eigenvectors_file)):
		mov_km = KMeansData(np.load(_labels_fpath(args.k_movs, args.mov_dim, args.mov_eigenvectors_file)), is_test)
		print('loaded movie kmeans')
	else:
		mov_km = cluster(args.mov_eigenvectors_file, args.mov_eigenvalues_file, args.mov_dim, args.k_movs)
		print('loaded movie kmeans')

	# Load trainset for inference
	tic = datetime.datetime.now()
	with open('trainset.pkl', 'rb') as fin:
		trainset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load trainset', toc-tic)

	# Test on training data
	tic = datetime.datetime.now()
	validate('TEST k-usr %6d k-mov %6d d-usr %4d d-mov %4d' % (
		args.k_usrs, args.k_movs,
		args.usr_dim, args.mov_dim
		), trainset, mov_km, usr_km)
	toc = datetime.datetime.now(); print('tictoc validate', toc-tic)
