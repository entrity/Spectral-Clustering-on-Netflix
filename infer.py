#!/usr/bin/env python3

import sys, os, datetime
import numpy as np
from my_util import *
from my_load import *
import cluster

def _load_ratings_data():
	# Load trainset for ratings means
	tic = datetime.datetime.now()
	with open('trainset.pkl', 'rb') as fin:
		trainset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load trainset', toc-tic)
	# Compile ratings data from trainset
	all_ratings = [row[2] for row in dataset]
	mean_rating = np.mean(ratings)
	ratings_matrix, _ = user_by_movie_matrix(dataset, uidmap)
	return mean_rating, ratings

def _load_kmeans_data():
	mov_labels_file = sys.argv[1]
	usr_labels_file = sys.argv[2]
	mov_labels = np.load(mov_labels_file)
	usr_labels = np.load(usr_labels_file)
	mov_km = cluster.KMeansData(mov_labels)
	usr_km = cluster.KMeansData(usr_labels)
	return mov_km, usr_km

def infer(testset, mean_rating, ratings_matrix, mov_km, usr_km):
	uidmap = construct_user_id_map()
	hyp = np.zeros((len(dataset), 3), np.short)
	for i, row in enumerate(dataset):
		mov_id, usr_id, empty_val, date = row
		usr, mov, mean = cluster.test_pt(None, mean_rating, ratings_matrix, mov_id, usr_id, uidmap, mov_km, usr_km)
		hyp[i,:] = usr, mov, mean
	print(sys.argv[1])
	print(sys.argv[2])
	acc_usr, acc_mov, acc_mean = acc.sum(axis=0)
	print('\t usr %7d : mov %7d : mean %7d' % (acc_usr, acc_mov, acc_mean))

if __name__ == '__main__':
	# Load trainset data
	mean_rating, ratings_matrix = _load_ratings_data()
	# Load k-means data
	mov_km, usr_km = _load_kmeans_data()
	# Load testset for inference
	tic = datetime.datetime.now()
	with open('testset.pkl', 'rb') as fin:
		testset = pickle.load(fin)
	toc = datetime.datetime.now(); print('tictoc load testset', toc-tic)
	# Infer on testing data
	infer(testset, mean_rating, ratings_matrix, mov_km, usr_km)
