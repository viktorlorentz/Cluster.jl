module Cluster

using Random
using LinearAlgebra
using Statistics

export KMeans, init_centroids, fit!, compute_distance, assign_center, update_centroids, predict, BKMeans, DC

# KMeans definition

"""
KMeans Clustering Algorithm

- Initializes centroids using either random selection or KMeans++.
- Iteratively assigns points to the nearest centroid.
- Updates centroids based on the mean of assigned points.
- Stops when centroids converge or after a maximum number of iterations.

References:
- [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
"""


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
function init_centroids(X::Matrix{Float64}, K, mode)
    if !isa(K, Int) || K <= 0
        throw(ArgumentError("K must be a positive integer"))
    end
    if mode != "kmeans" && mode != "kmeanspp" && mode != "dc"
        throw(ArgumentError("mode must be either 'kmeans' or 'kmeanspp'"))
    end

    if mode == "kmeans"
        # Generate random vector with the length of the data, then choose first K values
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode =="dc"
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
function fit!(model::KMeans, X::Matrix{Float64})
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
function compute_distance(X::Matrix{Float64}, centroids::Matrix{Float64})
    x = size(X)
    y = size(centroids)
    D = zeros(x[1], y[1])

    for (i, centroid) in enumerate(eachrow(centroids))
        for (j, x_row) in enumerate(eachrow(X))


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
function assign_center(D::Matrix{Float64})

    #return [argmin(D[i, :]) for i in 1:size(D, 1)]
    return [argmin(row) for row in eachrow(D)]
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
function update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model)
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
function predict(model::KMeans, X::Matrix{Float64})
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end



"""
Bisecting KMeans Clustering Algorithm

- Starts with a single cluster containing all data points.
- Recursively splits clusters based on the highest SSE until `k` clusters are obtained.
- Uses standard KMeans for cluster splitting.

References:
- [Bisecting KMeans: An Improved Version of KMeans](https://en.wikipedia.org/wiki/K-means_clustering#Bisecting_K-means)
"""
mutable struct BKMeans
    k::Int
    kmeans::KMeans
    labels::Array{Int,1}
    centroids::Matrix{Float64}
end

"""
    BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode="kmeans"))

Creates a new BKMeans clustering model.

Keyword Arguments:
- `k::Int`: The number of clusters (default: 3).
- `kmeans::KMeans`: An instance of the KMeans model used for cluster splitting (default: KMeans with 2 clusters).

Returns:
A `BKMeans` model with the specified parameters.
"""

function BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode="kmeans"))
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(kmeans, KMeans)
        throw(ArgumentError("kmeans must be an instance of KMeans"))
    end
    return BKMeans(k, kmeans, Int[], Array{Float64}(undef, 0, 0))
end

"""
    fit!(model::BKMeans, X)

Runs the Bisecting KMeans algorithm for the given data and model.

Arguments:
- `model::BKMeans`: The BKMeans model to be trained.
- `X`: The input data matrix where each row is a data point.
"""

function fit!(model::BKMeans, X::Matrix{Float64})
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

"""
    predict(model::BKMeans, X) -> Array

Returns the cluster labels for the given data points using the trained BKMeans model.

Arguments:
- `model::BKMeans`: The trained BKMeans model.
- `X::Array{Float64,2}`: The input data matrix where each row is a data point.

Returns:
An array of cluster labels for each data point.
"""
function predict(model::BKMeans, X)
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end



mutable struct DC
    k::Int
    mode
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end
# Constructor
"""
    DC(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> dc

    Creates a new DC clustering model.

    # Keyword Arguments
    - `k::Int`: The number of clusters (default: 3).
    - `mode::Symbol`: The mode of initialization (`:kmeans` or `:kmeans++`, default: `:kmeans`).
    - `max_try::Int`: The maximum number of iterations for the algorithm (default: 100).
    - `tol::Float64`: The tolerance for convergence (default: 1e-4).

    # Returns
    A `DC` model with the specified parameters.

"""
function DC(; k::Int=3, mode::String="dc", max_try::Int=100, tol::Float64=1e-4)
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(max_try, Int) || max_try <= 0
        throw(ArgumentError("max_try must be a positive integer"))
    end
    if !isa(tol, Float64) || tol <= 0
        throw(ArgumentError("tol must be a positive number"))
    end
    if mode != "kmeans" && mode != "kmeanspp" && mode != "dc"
        throw(ArgumentError("mode must be either 'kmeans' or 'kmeanspp'"))
    end
    return DC(k, mode, max_try, tol, zeros(Float64, 0, 0), Int[])
end

"""
    compute_objective_function(data, centroids,k) -> Array

    Computes the distance from each data point to each centroid.

    # Arguments
    - `data`: The input data matrix where each row is a data point.
    - `centroids`: The current centroids.
    - `k` the exponent of norm
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
function compute_objective_function(X::Matrix{Float64}, centroids::Matrix{Float64},k)
    x = size(X)
    y = size(centroids)
    D = zeros(x[1], y[1])
    delta = 0.0001

    ##if k = 2 the clustering criterion is the same as k means


    for (i, centroid) in enumerate(eachrow(centroids))
        for (j, x_row) in enumerate(eachrow(X))
            D[j, i] = sqrt(sum((x_row .- centroid) .^ k .+ delta))
        end
    end

    return D

end

"""
    update_centroids_dc(data, labelvector, model) -> Array

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
function update_centroids_dc(X::Matrix{Float64}, label_vector::Vector{Int64}, model)
    δ = 0.0001
    new_centers = zeros(model.k, size(X, 2))

    for i in 1:model.k

        # Mask for selecting points belonging to the i-th cluster
        mask = label_vector .== i

        # Points in the i-th cluster
        cluster_points = X[mask, :]

        # Number of points in the i-th cluster
        num_points = size(cluster_points, 1)

        # If no points are assigned to the cluster, skip the update
        if num_points == 0
            continue
        end

        # Compute the log-potential for each point in the cluster
        log_potentials = zeros(num_points)
        for j in 1:num_points
            d = cluster_points[j, :]

            temp = (cluster_points.-transpose(d)).^2

            temp2 = sum(temp,dims=2).+δ

            log_potential = sum(log.(temp2))

            log_potentials[j] = log_potential
        end

        # Find the point with the minimum log-potential
        min_index = argmin(log_potentials)
        new_centers[i, :] = cluster_points[min_index, :]

    end

    return new_centers

end

"""
    fit!(model::DC, X)

Runs the Distributional Clustering algorithm for the given data and model.

# Arguments
- `model::DC`: The DC model to be trained.
- `X`: The input data matrix where each row is a data point.
- `k`: Exponent of the norm

# Examples
```julia-repl

model = DC(3, "dc",20,1e-4,zeros(Float64, 0, 0), Int[])
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
function fit!(model::DC, X::Matrix{Float64})

    k = model.k

    if size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("X must be a non-empty matrix"))
    end

    model.centroids = init_centroids(X, model.k, model.mode)

    for i in 1:model.max_try

        D = compute_objective_function(X, model.centroids,k)

        model.labels = assign_center(D)

        new_centroids = update_centroids_dc(X, model.labels, model)


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
function predict(model::DC,X::Matrix{Float64})
    D = compute_objective_function(X, model.centroids,2)
    return assign_center(D)
end

end