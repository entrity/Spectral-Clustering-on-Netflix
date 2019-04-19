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

if __name__ == '__main__':
	fin = sys.argv[1]
	name, ext = os.path.splitext(fin)
	fout = name + '-sparse' + ext
	graph = np.load(fin)
	# print(len(graph.nonzero()[0]))
	import IPython; IPython.embed(); # to do: delete
	graph = sparsify(graph, int(sys.argv[2]))
	# print(len(graph.nonzero()[0]))
	np.save(fout, graph)
