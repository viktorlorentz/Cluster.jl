module Cluster

using Random
using LinearAlgebra
using Statistics

export KMeans, init_centroids, fit!, compute_distance, assign_center, update_centroids, predict, BKMeans

# KMeans definition
"""
    mutable struct KMeans

A structure representing a KMeans clustering model.

# Fields
- `k::Int`: The number of clusters.
- `mode::Symbol`: The mode of initialization (`:kmeans` or `:kmeans++`).
- `max_try::Int`: The maximum number of iterations for the algorithm.
- `tol::Float64`: The tolerance for convergence.
- `centroids::Array{Float64,2}`: The centroids of the clusters.
- `labels::Array{Int,1}`: The labels assigned to each data point.

"""
mutable struct KMeans
    k::Int
    mode::String
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end

# Constructor
"""
    KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> KMeans

Creates a new KMeans clustering model.

# Keyword Arguments
- `k::Int`: The number of clusters (default: 3).
- `mode::Symbol`: The mode of initialization (`:kmeans` or `:kmeans++`, default: `:kmeans`).
- `max_try::Int`: The maximum number of iterations for the algorithm (default: 100).
- `tol::Float64`: The tolerance for convergence (default: 1e-4).

# Returns
A `KMeans` model with the specified parameters.

"""
function KMeans(; k::Int=3, mode::String="kmeans", max_try::Int=100, tol::Float64=1e-4)
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(max_try, Int) || max_try <= 0
        throw(ArgumentError("max_try must be a positive integer"))
    end
    if !isa(tol, Float64) || tol <= 0
        throw(ArgumentError("tol must be a positive number"))
    end
    if mode != "kmeans" && mode != "kmeanspp"
        throw(ArgumentError("mode must be either 'kmeans' or 'kmeanspp'"))
    end
    return KMeans(k, mode, max_try, tol, zeros(Float64, 0, 0), Int[])
end

"""
    init_centroids(X::Array{Float64,2}, K::Int, mode::Symbol) -> Array{Float64,2}

Initialize centroids for the chosen algorithm.

# Arguments
- `X::Array{Float64,2}`: The input data matrix where each row is a data point.
- `K::Int`: The number of clusters.
- `mode::Symbol`: The mode of initialization (`:kmeans` or `:kmeans++`).

# Returns
An array of centroids initialized based on the chosen mode.

# Examples
```julia-repl
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
function init_centroids(X, K, mode)
    if !isa(K, Int) || K <= 0
        throw(ArgumentError("K must be a positive integer"))
    end
    if mode != "kmeans" && mode != "kmeanspp"
        throw(ArgumentError("mode must be either 'kmeans' or 'kmeanspp'"))
    end

    if mode == "kmeans"
        # Generate random vector with the length of the data, then choose first K values
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == "kmeanspp"
        # kmeans++ initialization
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

        for k in 2:K
            # Distance from data points to centroids
            D = compute_distance(X, centroids[1:k-1, :])

            # Assign each data point to its nearest centroid
            Dm = minimum(D, dims=2)

            # Choose the next centroid
            probabilities = vec(Dm) / sum(Dm)
            cummulative_probabilities = cumsum(probabilities)

            # Perform weighted random selection
            r_num = rand()
            next_centroid_ind = searchsortedfirst(cummulative_probabilities, r_num)
            centroids[k, :] = X[next_centroid_ind, :]
        end
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    return centroids
end

"""
    fit!(model::KMeans, X)

Runs the KMeans algorithm for the given data and model.

# Arguments
- `model::KMeans`: The KMeans model to be trained.
- `X`: The input data matrix where each row is a data point.

# Examples
```julia-repl
model = KMeans(k=3, mode=:kmeans)
X = [
    1.0 1.0;
    1.5 2.0;
    3.0 4.0;
    5.0 6.0;
    8.0 9.0;
    10.0 11.0
]
fit!(model, X)
```
"""
function fit!(model::KMeans, X)
    if size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("X must be a non-empty matrix"))
    end

    model.centroids = init_centroids(X, model.k, model.mode)

    for i in 1:model.max_try
        D = compute_distance(X, model.centroids)
        model.labels = assign_center(D)
        new_centroids = update_centroids(X, model.labels, model)

        for j in 1:model.k
            if !(j in model.labels)
                new_centroids[j, :] = X[rand(1:size(X, 1)), :]
            end
        end

        if maximum(sqrt.(sum((model.centroids .- new_centroids) .^ 2, dims=2))) < model.tol
            break
        end
        model.centroids = new_centroids
    end
end

"""
    compute_distance(data, centroids) -> Array

Computes the distance from each data point to each centroid.

# Arguments
- `data`: The input data matrix where each row is a data point.
- `centroids`: The current centroids.

# Returns
A distance matrix `D` of size (number of data points, number of centroids), where `D[i, j]` is the distance from data point `i` to centroid `j`.

# Examples
```julia-repl
X = [
    1.0 1.0;
    1.5 2.0;
    3.0 4.0
]
centroids = [
    1.0 1.0;
    3.0 4.0
]
D = compute_distance(X, centroids)
```
"""
function compute_distance(X, centroids)

    # if size(X, 2) != size(centroids, 2)
    #     throw(DimensionMismatch("The number of columns in X ($(size(X, 2))) must match the number of columns in centroids ($(size(centroids, 2)))."))
    # end
    x = size(X)
    y = size(centroids)
    D = zeros(x[1], y[1])

    print(size(X))
    print(size(centroids))
    for i in 1:y[1]
        for j in 1:x[1]
            D[j, i] = sqrt(sum((X[j, :] .- centroids[i, :]) .^ 2))
        end
    end
    return D
end

"""
    assign_center(D) -> Array

Returns the index of the nearest centroid for each data point.

# Arguments
- `D`: The distance matrix.

# Returns
An array of indices indicating the nearest centroid for each data point.

# Examples
```julia-repl
D = [
    0.0 2.0;
    1.0 1.0;
    2.0 0.0
]
labels = assign_center(D)
```
"""
function assign_center(D)
    return [argmin(D[i, :]) for i in 1:size(D, 1)]
end

"""
    update_centroids(data, labelvector, model) -> Array

Calculates new centroids based on the given data and label vector.

# Arguments
- `data`: The input data matrix where each row is a data point.
- `labelvector`: The current labels of the data points.
- `model`: The KMeans model.

# Returns
An array of new centroids.

# Examples
```julia-repl
X = [
    1.0 1.0;
    1.5 2.0;
    3.0 4.0;
    5.0 6.0
]
labels = [1, 1, 2, 2]
model = KMeans(k=2)
new_centroids = update_centroids(X, labels, model)
```
"""
function update_centroids(X, label_vector, model)
    r, c = size(X)
    centroids = zeros(model.k, c)

    for label in 1:model.k
        # Create a mask for the current label
        mask = label_vector .== label
        # Average the values using the mask
        centroids[label, :] = mean(X[mask, :], dims=1)
    end

    return centroids
end

"""
    predict(model,data) -> Array

Returns the cluster labels for the given data points.

# Arguments
- `model::KMeans`: The trained KMeans model.
- `X::Array{Float64,2}`: The input data matrix where each row is a data point.

# Returns
An array of cluster labels for each data point.

# Examples
```julia-repl
data = [
    # Cluster 1
    1.0 1.0 1.5;
    1.5 2.0 1.6;
    1.3 1.8 1.4;
    # Cluster 2
    5.0 7.0 3.5;
    5.5 7.5 3.5;
    6.0 7.0 3.5;
    # Cluster 3
    8.0 1.0 6.5;
    8.5 1.5 6.5;
    8.3 1.2 7.5;
]
test_data = [1.1 1.1 1.2]
model = KMeans(k=3)
fit!(model, data)
labels = predict(model, test_data)
```
"""
function predict(model::KMeans, X)
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end

mutable struct BKMeans
    k::Int
    kmeans::KMeans
    labels::Array{Int,1}
    centroids::Matrix{Float64}
end

function BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode="kmeans"))
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(kmeans, KMeans)
        throw(ArgumentError("kmeans must be an instance of KMeans"))
    end
    return BKMeans(k, kmeans, Int[], Array{Float64}(undef, 0, 0))
end

function fit!(model::BKMeans, X)
    if size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("X must be a non-empty matrix"))
    end

    clusters = [X]

    while length(clusters) < model.k
        sse = [sum(compute_distance(clusters[i], mean(clusters[i], dims=1)) .^ 2) for i in 1:length(clusters)]
        i = argmax(sse)
        sub_model = deepcopy(model.kmeans)
        fit!(sub_model, clusters[i])
        new_clusters = [clusters[i][sub_model.labels .== 1, :], clusters[i][sub_model.labels .== 2, :]]

        deleteat!(clusters, i)
        for new_cluster in new_clusters
            if size(new_cluster, 1) > 0  # Ensure new cluster is non-empty
                push!(clusters, new_cluster)
            end
        end
    end
    model.labels = Int[]
    model.centroids = zeros(Float64, length(clusters), size(X, 2))
    for g1 in 1:length(clusters)
        model.centroids[g1 , :] = mean(clusters[g1], dims=1)[:]
        rows, _ = size(clusters[g1])
        for g2 in 1:rows
            push!(model.labels, g1)
        end
    end
end

function predict(model::BKMeans, X)
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end

end