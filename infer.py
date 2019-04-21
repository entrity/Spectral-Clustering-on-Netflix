#!/usr/bin/env python3

import sys, os, datetime
import numpy as np
from my_util import *
from my_load import *
import cluster

def _load_ratings_data(uidmap):
	# Load trainset for ratings means
	tic = datetime.datetime.now()
	with open('trainset.pkl', 'rb') as fin:
		trainset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load trainset', toc-tic)
	# Compile ratings data from trainset
	all_ratings = [row[2] for row in trainset]
	mean_rating = np.mean(all_ratings)
	ratings_matrix, _ = user_by_movie_matrix(trainset, uidmap)
	return mean_rating, ratings_matrix

def _load_kmeans_data():
	mov_labels_file = sys.argv[1]
	usr_labels_file = sys.argv[2]
	mov_labels = np.load(mov_labels_file)
	usr_labels = np.load(usr_labels_file)
	mov_km = cluster.KMeansData(mov_labels, True)
	usr_km = cluster.KMeansData(usr_labels, True)
	return mov_km, usr_km

def infer(uidmap, testset, mean_rating, ratings_matrix, mov_km, usr_km):
	hyps = np.zeros((len(testset), 4), np.short)
	# accs = np.zeros((len(testset), 4), np.short)
	# errs = np.zeros((len(testset), 4), np.float)

	n_matches_in_matrix = 0
	for i, row in enumerate(testset):
		mov_id, usr_id, empty_val, date = row
		usr_idx = get_usr_idx(uidmap, usr_id)
		mov_idx = get_mov_idx(mov_id)
		matrix_val = ratings_matrix[usr_idx,mov_idx]
		if 0 == matrix_val:
			hyps[i,:] = cluster.test_pt(None, mean_rating, ratings_matrix, mov_id, usr_id, uidmap, mov_km, usr_km)
		else:
			hyps[i,:] = matrix_val
			n_matches_in_matrix += 1
		lineargs = [i] + [x for x in hyps[i,:]]
		print('\t'.join([str(x) for x in lineargs]))

	print(sys.argv[1])
	print(sys.argv[2])
	n = float(len(testset))
	print('n_matches_in_matrix : %d' % n_matches_in_matrix)


if __name__ == '__main__':
	uidmap = construct_user_id_map()
	# Load trainset data
	mean_rating, ratings_matrix = _load_ratings_data(uidmap)
	# Load k-means data
	mov_km, usr_km = _load_kmeans_data()
	# Load testset for inference
	tic = datetime.datetime.now()
	with open('testset.pkl', 'rb') as fin:
		testset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load testset', toc-tic)
	# Infer on testing data
	infer(uidmap, testset, mean_rating, ratings_matrix, mov_km, usr_km)
