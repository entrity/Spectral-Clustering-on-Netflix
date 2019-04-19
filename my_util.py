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
	assert 0 == len(np.diag(graph).nonzero()[0]) # ensure there are no self-connections
	print('assert 1 done')
	tic = datetime.datetime.now()
	for i, vec in enumerate(graph):
		idxs = np.argsort(vec) # ascending order
		nixs = idxs[:-k]
		graph[i,nixs] = 0
	print('applied knn')
	toc = datetime.datetime.now(); print('tictoc mysparsify', toc - tic)

# Pass in sparse graph
def norm_laplacian(graph):
	tic = datetime.datetime.now()
	A = (graph + graph.T) / 2 # Make it symmetric
	d = np.sum(A, axis=0)
	D = np.diag(d)
	Lu = D - A
	normD = np.diag(d**(-1/2));
	Ln = np.matmul(np.matmul(normD,Lu),normD)
	print('got Laplacian')
	toc = datetime.datetime.now(); print('tictoc norm_laplacian', toc - tic)
	return Ln

if __name__ == '__main__':
	fin = sys.argv[1]
	k   = int(sys.argv[2])
	print('k = %d' % k)
	import matplotlib.pyplot as plt
	import scipy.linalg
	name, ext = os.path.splitext(fin)
	# Load graph
	graph = np.load(fin)
	# Sparsify graph
	print('nonzeros', np.count_nonzero(graph))
	mysparsify(graph, k)
	print('nonzeros', np.count_nonzero(graph))
	# Get Laplacian
	Ln = norm_laplacian(graph)
	fout = 'Ln-%s-%d-%s' % (name, k, ext)
	np.save(fout, graph)
	print('saved Laplacian')
	# Get eigenvectors
	vals, vecs = scipy.linalg.eigh(Ln, eigvals=(0, 1000))
	np.save('vals-%s-%d' % (name, k), vals)
	np.save('vecs-%s-%d' % (name, k), vecs)
	print('saved eigenvectors, eigenvalues')
	# Plot eigenvalues
	fig = plt.figure()
	x = np.arange(1001)
	plt.plot(x, vals)
	fig.savefig('1000-eigvals-%s-%d.png' % (name, k))
