#!/usr/bin/env python3
import pickle
import numpy as np
import sys, os
import argparse

from my_load import *
from my_util import *

def adjacency_matrices(movies, uidmap, ratings_csv):
  ratings, datings = user_by_movie_matrix(ratings_csv, uidmap)
  nonzeros = np.nonzero(ratings)
  xs = [x for x in nonzeros[0]]
  ys = [x for x in nonzeros[1]]
  nonzeros = zip(xs, ys)
  all_ratings = np.array([row[RATE_COL] for row in ratings_csv])
  mean_rating = all_ratings.mean()
  n_movies = len(movies)
  n_users  = len(uidmap.keys())

  # Adjacency matrices (un-thresholded)
  movie_by_movie = np.zeros((n_movies, n_movies), np.float) # could use mean for this but won't. negatives will cancel out positives.
  user_by_user   = np.zeros((n_users, n_users), np.float)

  print('building movie graph...')
  for user in ratings.transpose():
    nzidxs = user.nonzero()[0] # nonzero indices for user
    nzrats = user[nzidxs] # nonzero ratings for user
    for i, rating in enumerate(nzidxs):
      nzdeltas = 1.5 - abs(nzrats - nzrats[i])
      m = nzidxs[i]
      movie_by_movie[m,nzidxs] += nzdeltas
      movie_by_movie[nzidxs,m] += nzdeltas
  if movie_by_movie.min() < 0:
    movie_by_movie -= movie_by_movie.min()
  for i in range(movie_by_movie.shape[0]):  # Remove self-connections
    movie_by_movie[i,i] = 0
  print('built movie graph')

  print('building user graph...')
  for movie in ratings:
    nzidxs = movie.nonzero()[0]
    nzrats = movie[nzidxs]
    for i, rating in enumerate(nzidxs):
      nzdeltas = 1.5 - abs(nzrats - nzrats[i])
      u = nzidxs[i]
      user_by_user[u,nzidxs] += nzdeltas
      user_by_user[nzidxs,u] += nzdeltas
  if user_by_user.min() < 0:
    user_by_user -= user_by_user.min()
  for i in range(user_by_user.shape[0]): # Remove self-connections
    user_by_user[i,i] = 0
  print('built user graph')
  
  return movie_by_movie, user_by_user
  

if __name__ == '__main__':
  # Parse user args
  parser = argparse.ArgumentParser()
  parser.add_argument('-u', '--uid-map', default='user-id-map.pkl')
  parser.add_argument('--trainset', default='trainset.pkl')
  parser.add_argument('--valset', default='valset.pkl')
  parser.add_argument('--movie-graph', '--mg', default='movie-graph.npy')
  parser.add_argument('--user-graph', '--ug', default='user-graph.npy')
  args = parser.parse_args()

  # Load users
  if os.path.exists(args.uid_map):
    with open(args.uid_map, 'rb') as fin: uidmap = pickle.load(fin)
    print('loaded userid map')
  else:
    uidmap = construct_user_id_map(args.uid_map)
    print('saved userid map')
  
  # Load movies
  movies = load_movies('movie_titles.txt')
  print('loaded movie titles')
  
  # Load trainset and valset
  if os.path.exists(args.trainset) and os.path.exists(args.valset):
    with open(args.trainset, 'rb') as fin: trainset = pickle.load(fin)
    with open(args.valset, 'rb') as fin:   valset   = pickle.load(fin)
    print('loaded trainset, valset')
  else:
    train = load_csv('train.csv')
    print('loaded train.csv')
    trainset, valset = split_dataset(train, 0.15)
    print('split dataset')
    with open(args.trainset, 'wb') as fout: pickle.dump(trainset, fout)
    with open(args.valset, 'wb') as fout: pickle.dump(valset, fout)
    print('saved trainset, valset')
  
  # Create fully-connected graph
  if os.path.exists(args.movie_graph) and os.path.exists(args.user_graph):
    movie_graph = np.load(args.movie_graph)
    user_graph = np.load(args.user_graph)
  else:
    movie_graph, user_graph = adjacency_matrices(movies, uidmap, trainset)
    np.save(args.movie_graph, movie_graph)
    print('saved movie graph')
    np.save(args.user_graph, user_graph)
    print('saved user graph')

  # Sparsify graphs
  print('sparsifying movie graph...')
  movie_graph = sparsify(movie_graph, 5)
  print('sparsified movie graph')
  print('sparsifying user graph...')
  user_graph  = sparsify(user_graph, 5)
  print('sparsified user graph')
