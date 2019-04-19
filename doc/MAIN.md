1. Construct movie graph
1. Construct user graph
1. For each graph
  1. Construct fully-connected graph, whose edge weights are the similarity of the two nodes.
    1. Let similarity exist as sum of alt-similarities, divided by N*0.9
      1. N is the number of alt-similarities
      1. An alt-similarity exists for two movies if a single user rated them similarly. Similarly means within 1 point.
  1. Remove self-connections.
  1. Sparsify via k-nearest-neighbours
  1. Compute normalized Laplacian
  1. Compute first k eigenvectors on Ln to obtain embedding
  1. Compute k-means clustering (then t-SNE for review)
1. Given a movie-user pair to test, predict the value as mean of avg from user cluster and avg from movie cluster.

Then redo this with the constraint matrix Q.
