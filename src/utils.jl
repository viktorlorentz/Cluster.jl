"""
## Utility functions

This module contains utility functions that are used by the clustering algorithms.
"""

"""
$(TYPEDEF)

Initialize centroids for the chosen algorithm.

# Arguments
- `X::Array{Float64,2}`: The input data matrix where each row is a data point.
- `K::Int`: The number of clusters.
- `mode::Symbol`: The mode of initialization (`:kmeans` or `:kmeans++`).

## Fields
$(TYPEDFIELDS)

# Examples
```@repl
X = [
    1.0 1.0;
    1.5 2.0;
    3.0 4.0;
    5.0 6.0;
    8.0 9.0;
    10.0 11.0
]
K = 2
mode = :kmeans
centroids = init_centroids(X, K, mode)
```
"""
function init_centroids(X::Matrix{Float64}, K::Int64, mode::String)
    if !isa(K, Int) || K <= 0
        throw(ArgumentError("K must be a positive integer"))
    end
    if mode != "kmeans" && mode != "kmeanspp" && mode != "dc"
        throw(ArgumentError("mode must be either 'kmeans' or 'kmeanspp'"))
    end

    if mode == "kmeans"
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == "dc"
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == "kmeanspp"
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

        for k in 2:K
            D = compute_distance(X, centroids[1:k-1, :])
            Dm = minimum(D, dims=2)
            probabilities = vec(Dm) / sum(Dm)
            cummulative_probabilities = cumsum(probabilities)
            r_num = rand()
            next_centroid_ind = searchsortedfirst(cummulative_probabilities, r_num)
            centroids[k, :] = X[next_centroid_ind, :]
        end
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    return centroids
end
