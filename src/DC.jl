"""
    mutable struct DC

A mutable struct for the Density-based Clustering (DC) algorithm.

### Fields
- `k::Int`: Number of clusters.
- `mode::Symbol`: Initialization mode, either `:random` or `:kmeanspp`.
- `max_try::Int`: Maximum number of iterations.
- `tol::Float64`: Tolerance for convergence.
- `centroids::Array{Float64,2}`: Matrix of centroid coordinates.
- `labels::Array{Int,1}`: Vector of labels for each data point.

### Examples
```julia-repl
julia> model = DC(k=3, mode=:random, max_try=100, tol=1e-4)
DC(3, :random, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[])
```
"""
mutable struct DC
    k::Int
    mode::Symbol
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end

"""
    DC(; k::Int=3, mode::Symbol=:kmeanspp, max_try::Int=100, tol::Float64=1e-4)

Constructor for the DC struct.

### Input
- `k::Int`: Number of clusters (default: 3).
- `mode::Symbol`: Initialization mode, either `:random` or `:kmeanspp`(default: :kmeanspp).
- `max_try::Int`: Maximum number of iterations (default: 100).
- `tol::Float64`: Tolerance for convergence (default: 1e-4).

### Output
- Returns an instance of `DC`.

### Examples
```julia-repl
julia> model = DC(k=3, mode=:kmeanspp, max_try=100, tol=1e-4)
DC(3, :kmeanspp, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[])
```
"""
function DC(; k::Int=3, mode::Symbol=:kmeanspp, max_try::Int=100, tol::Float64=1e-4)
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(max_try, Int) || max_try <= 0
        throw(ArgumentError("max_try must be a positive integer"))
    end
    if !isa(tol, Float64) || tol <= 0
        throw(ArgumentError("tol must be a positive number"))
    end
    if mode != :random && mode != :kmeanspp
        throw(ArgumentError("mode must be either :random or :kmeanspp"))
    end
    return DC(k, mode, max_try, tol, zeros(Float64, 0, 0), Int[])
end

"""
    compute_objective_function(X::Matrix{Float64}, centroids::Matrix{Float64}; p=2, delta=0.0001)

Computes the objective function for the DC algorithm.

### Input
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.
- `centroids::Matrix{Float64}`: Matrix of centroid coordinates.
- `p`: Power parameter for the distance metric (default: 2).
- `delta`: Small constant to avoid division by zero (default: 0.0001).

### Output
- Returns a distance matrix where element (i, j) is the distance between the i-th data point and the j-th centroid.

### Examples
```julia-repl
julia> X = rand(100, 2)
julia> centroids = rand(3, 2)
julia> D = compute_objective_function(X, centroids)
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
    update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model::DC; delta=0.0001)

Updates the centroids based on the current assignment of data points to centroids.

### Input
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.
- `label_vector::Vector{Int64}`: Vector of labels for each data point.
- `model::DC`: An instance of `DC`.
- `delta`: Small constant to avoid division by zero (default: 0.0001).

### Output
- Returns a matrix of updated centroid coordinates.

### Examples
```julia-repl
julia> X = rand(100, 2)
julia> labels = rand(1:3, 100)
julia> model = DC(k=3)
julia> centroids = update_centroids(X, labels, model)
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
    fit!(model::DC, X::Matrix{Float64})

Fits the DC model to the data matrix X.

### Input
- `model::DC`: An instance of `DC`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Modifies the `model` in-place to fit the data.

### Algorithm
1. Initialize centroids.
2. Iterate up to `max_try` times:
    a. Compute the objective function between data points and centroids.
    b. Assign each data point to the nearest centroid.
    c. Update centroids based on the current assignment.
    d. Check for convergence based on `tol`.

### Examples
```julia-repl
julia> model = DC(k=3)
julia> X = rand(100, 2)
julia> fit!(model, X)
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
    predict(model::DC, X::Matrix{Float64})

Predicts the cluster labels for new data points based on the fitted DC model.

### Input
- `model::DC`: An instance of `DC`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Returns a vector of predicted labels for each data point.

### Examples
```julia-repl
julia> model = DC(k=3)
julia> X_train = rand(100, 2)
julia> fit!(model, X_train)
julia> X_test = rand(10, 2)
julia> labels = predict(model, X_test)
```
"""
function predict(model::DC, X::Matrix{Float64})
    D = compute_objective_function(X, model.centroids)
    return assign_center(D)
end
