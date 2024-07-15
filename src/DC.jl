"""
## Distributional Clustering Method

References:
- [Krishna, A., Mak, S. and Joseph, R., 2019. Distributional clustering: A distribution-preserving clustering method. arXiv preprint arXiv:1911.05940](https://arxiv.org/abs/1911.05940)
"""

mutable struct DC
    k::Int
    mode::String
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end

# Constructor
"""
    DC(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> DC

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
        throw(ArgumentError("mode must be either 'kmeans', 'kmeanspp', or 'dc'"))
    end
    return DC(k, mode, max_try, tol, zeros(Float64, 0, 0), Int[])
end

"""
    compute_objective_function(data, centroids, k) -> Array

Computes the distance from each data point to each centroid.

# Arguments
- `data`: The input data matrix where each row is a data point.
- `centroids`: The current centroids.
- `k`: the exponent of norm

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
D = compute_objective_function(X, centroids)
```
"""
function compute_objective_function(X::Matrix{Float64}, centroids::Matrix{Float64}; p=2, delta=0.0001)
    x = size(X)
    y = size(centroids)
    D = zeros(x[1], y[1])

    for (i, centroid) in enumerate(eachrow(centroids))
        for (j, x_row) in enumerate(eachrow(X))
            D[j, i] = sqrt(sum(abs.(x_row .- centroid) .^ p .+ delta))
        end
    end
    return D
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
function update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model::DC; delta=0.0001)
    new_centers = zeros(model.k, size(X, 2))
    for i in 1:model.k
        mask = label_vector .== i
        cluster_points = X[mask, :]
        num_points = size(cluster_points, 1)
        if num_points == 0
            continue
        end
        log_potentials = zeros(num_points)
        for j in 1:num_points
            d = cluster_points[j, :]
            temp = (cluster_points .- transpose(d)) .^ 2
            temp2 = sum(temp, dims=2) .+ delta
            log_potential = sum(log.(temp2))
            log_potentials[j] = log_potential
        end
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
model = DC(3, "dc", 20, 1e-4, zeros(Float64, 0, 0), Int[])
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
    if size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("X must be a non-empty matrix"))
    end

    model.centroids = init_centroids(X, model.k, model.mode)

    for i in 1:model.max_try
        D = compute_objective_function(X, model.centroids)
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
    predict(model::DC, X) -> Array

Returns the cluster labels for the given data points.

# Arguments
- `model::DC`: The trained DC model.
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
model = DC(k=3)
fit!(model, data)
labels = predict(model, test_data)
```
"""
function predict(model::DC, X::Matrix{Float64})
    D = compute_objective_function(X, model.centroids)
    return assign_center(D)
end
