```@meta
CurrentModule = Cluster
```
# API Reference

```@contents
Pages = ["api.md"]
Depth = 5
```

## Utility functions

All functions used by all algorithms.

```@autodocs; canonical=false
Modules = [Cluster]
Pages = ["src/utils.jl"]
```

## KMeans / KMeans++ Clustering Algorithm

- Initializes centroids using either random selection or KMeans++.
- Iteratively assigns points to the nearest centroid.
- Updates centroids based on the mean of assigned points.
- Stops when centroids converge or after a maximum number of iterations.

References:
- [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

```@autodocs; canonical=false
Modules = [Cluster]
Pages = ["src/KMeans.jl"]
```

## Bisecting KMeans Clustering Algorithm

- Starts with a single cluster containing all data points.
- Recursively splits clusters based on the highest SSE until `k` clusters are obtained.
- Uses standard KMeans for cluster splitting.

References:
- [Bisecting KMeans: An Improved Version of KMeans](https://en.wikipedia.org/wiki/K-means_clustering#Bisecting_K-means)

```@autodocs; canonical=false
Modules = [Cluster]
Pages = ["src/BKMeans.jl"]
```

## Distributional Clustering Method

References:
- [Krishna, A., Mak, S. and Joseph, R., 2019. Distributional clustering: A distribution-preserving clustering method. arXiv preprint arXiv:1911.05940](https://arxiv.org/abs/1911.05940)

```@autodocs; canonical=false
Modules = [Cluster]
Pages = ["src/DC.jl"]
```

## Full list of available functions
```@index
```

```@autodocs
Modules = [Cluster]
```