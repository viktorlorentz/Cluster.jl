"""
    init_centroids(X::Matrix{Float64}, K::Int64, mode::String)

Initializes centroids for the clustering algorithm based on the specified mode.

### Input
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.
- `K::Int64`: Number of clusters.
- `mode::String`: Initialization mode, either "kmeans", "kmeanspp", or "dc".

### Output
- Returns a matrix of initialized centroid coordinates.

### Algorithm
1. If `mode` is "kmeans" or "dc":
    - Randomly select K data points from X as initial centroids.
2. If `mode` is "kmeanspp":
    - Initialize the first centroid randomly.
    - For each subsequent centroid:
        a. Compute the distance from each data point to the nearest centroid.
        b. Select the next centroid with probability proportional to the squared distance.

### Examples
```julia-repl
julia> X = rand(100, 2)
julia> centroids = init_centroids(X, 3, "kmeans")
3×2 Matrix{Float64}:
 0.386814  0.619566
 0.170768  0.0176449
 0.38688   0.398064
```
"""
function init_centroids(X::Matrix{Float64}, K::Int64, mode::String)
    if !isa(K, Int) || K <= 0
        throw(ArgumentError("K must be a positive integer"))
    end
    if mode != "kmeans" && mode != "kmeanspp" && mode != "dc"
        throw(ArgumentError("mode must be either 'kmeans', 'kmeanspp', or 'dc'"))
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