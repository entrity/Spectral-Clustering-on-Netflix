import sys, os
import datetime
import numpy as np
import sklearn.neighbors

def parsedate(datestr): return datetime.datetime.strptime(datestr, '%Y-%m-%d')
def calc_date_distance( a, b ): return abs((a - b).days)
def calc_rate_distance( a, b ): return abs(int(a) - int(b))

def split_dataset(csvdata, percentage_to_hold_out):
  n = len(csvdata)
  k = round(percentage_to_hold_out * n) # number of held out examples
  idxs = np.arange(n)
  held_idxs = np.random.choice(idxs, k, replace=False)
  idxs_mask = np.ones(n, np.bool)
  idxs_mask[held_idxs] = False
  used_data = [csvdata[i] for i,v in enumerate(idxs_mask) if v == True]
  held_data = [csvdata[i] for i,v in enumerate(idxs_mask) if v == False]
  return used_data, held_data

def sparsify(graph, k):
	assert 0 == len(np.diag(graph).nonzero()[0])
	indicators = sklearn.neighbors.kneighbors_graph(graph, k).toarray()
	graph = indicators * graph # Sparse (possibly asymmetric) graph
	assert np.allclose(graph, graph.T, tol=0.01)
	return graph

def mysparsify(graph, k):
	tic = datetime.datetime.now()
	assert 0 == len(np.diag(graph).nonzero()[0])
	print('assert 1 done')
	for i, vec in enumerate(graph):
		idxs = np.argsort(vec) # ascending order
		nixs = idxs[:-k]
		graph[i,nixs] = 0
	print('applied knn')
	toc = datetime.datetime.now()
	print('tictoc', toc - tic)

if __name__ == '__main__':
	fin = sys.argv[1]
	k   = int(sys.argv[2])
	print('k = %d' % k)
	import matplotlib.pyplot as plt
	import scipy.linalg
	name, ext = os.path.splitext(fin)
	fout = '%s-sparse-%d-%s' % (name, k, ext)
	graph = np.load(fin)
	print(np.count_nonzero(graph))
	mysparsify(graph, k)
	print(np.count_nonzero(graph))
	fig = plt.figure()
	vals, vecs = scipy.linalg.eigh(graph, eigvals=(0, 1000))
	x = np.arange(1001)
	plt.plot(x, vals)
	print('saving...')
	fig.savefig('1000-eigvals-%d.png' % k)
	np.save(fout, graph)
	np.save('vals-%s-%d' % (name, k), vals)
	np.save('vecs-%s-%d' % (name, k), vecs)
	print('saved')
