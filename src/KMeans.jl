"""
    mutable struct KMeans

A mutable struct for the KMeans clustering algorithm.

### Fields
- `k::Int`: Number of clusters.
- `mode::Symbol`: Initialization mode, either `:random` or `:kmeanspp`.
- `max_try::Int`: Maximum number of iterations.
- `tol::Float64`: Tolerance for convergence.
- `centroids::Array{Float64,2}`: Matrix of centroid coordinates.
- `labels::Array{Int,1}`: Vector of labels for each data point.

### Examples
```julia-repl
julia> model = KMeans(k=3, mode=:kmeanspp, max_try=100, tol=1e-4)
KMeans(3, :kmeanspp, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[])
```
"""
mutable struct KMeans
    k::Int
    mode::Symbol
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end

"""
    KMeans(; k::Int=3, mode::Symbol=:kmeanspp, max_try::Int=100, tol::Float64=1e-4)

Constructor for the KMeans struct.

### Input
- `k::Int`: Number of clusters (default: 3).
- `mode::Symbol`: Initialization mode, either `:random` or `:kmeanspp` (default: :kmeanspp).
- `max_try::Int`: Maximum number of iterations (default: 100).
- `tol::Float64`: Tolerance for convergence (default: 1e-4).

### Output
- Returns an instance of `KMeans`.

### Examples
```julia-repl
julia> model = KMeans(k=3, mode=:kmeanspp, max_try=100, tol=1e-4)
KMeans(3, :kmeanspp, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[])
```
"""
function KMeans(; k::Int=3, mode::Symbol=:kmeanspp, max_try::Int=100, tol::Float64=1e-4)
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
    return KMeans(k, mode, max_try, tol, zeros(Float64, 0, 0), Int[])
end

"""
    fit!(model::KMeans, X::Matrix{Float64})

Fits the KMeans model to the data matrix X.

### Input
- `model::KMeans`: An instance of `KMeans`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Modifies the `model` in-place to fit the data.

### Algorithm
1. Initialize centroids.
2. Iterate up to `max_try` times:
    a. Compute distances between data points and centroids.
    b. Assign each data point to the nearest centroid.
    c. Update centroids based on the mean of assigned data points.
    d. Check for convergence based on `tol`.

### Examples
```julia-repl
julia> model = KMeans(k=3)
julia> X = rand(100, 2)
julia> fit!(model, X)
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
    compute_distance(X::Matrix{Float64}, centroids::Matrix{Float64})

Computes the distance between each data point in X and each centroid.

### Input
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.
- `centroids::Matrix{Float64}`: Matrix of centroid coordinates.

### Output
- Returns a distance matrix where element (i, j) is the distance between the i-th data point and the j-th centroid.

### Examples
```julia-repl
julia> X = rand(100, 2)
julia> centroids = rand(3, 2)
julia> D = compute_distance(X, centroids)
100×3 Matrix{Float64}:
 0.181333  0.539578  0.306867
 0.754863  0.48797   0.562147
 0.205116  0.360735  0.127107
 0.154926  0.552747  0.323433
 ⋮
 0.434321  0.321914  0.261909
 0.773258  0.291669  0.513668
 0.607547  0.310411  0.38714
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
    assign_center(D::Matrix{Float64})

Assigns each data point to the nearest centroid based on the distance matrix D.

### Input
- `D::Matrix{Float64}`: Distance matrix where element (i, j) is the distance between the i-th data point and the j-th centroid.

### Output
- Returns a vector of labels where each element is the index of the nearest centroid for the corresponding data point.

### Examples
```julia-repl
julia> D = rand(100, 3)
julia> labels = assign_center(D)
```
"""
function assign_center(D::Matrix{Float64})
    return [argmin(row) for row in eachrow(D)]
end

"""
    update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model::KMeans)

Updates the centroids based on the current assignment of data points to centroids.

### Input
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.
- `label_vector::Vector{Int64}`: Vector of labels for each data point.
- `model::KMeans`: An instance of `KMeans`.

### Output
- Returns a matrix of updated centroid coordinates.

### Examples
```julia-repl
julia> X = rand(100, 2)
julia> labels = rand(1:3, 100)
julia> model = KMeans(k=3)
julia> centroids = update_centroids(X, labels, model)
```
"""
function update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model::KMeans)
    r, c = size(X)
    centroids = zeros(model.k, c)

    for label in 1:model.k
        mask = label_vector .== label
        centroids[label, :] = mean(X[mask, :], dims=1)
    end

    return centroids
end

"""
    predict(model::KMeans, X::Matrix{Float64})

Predicts the cluster labels for new data points based on the fitted model.

### Input
- `model::KMeans`: An instance of `KMeans`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Returns a vector of predicted labels for each data point.

### Examples
```julia-repl
julia> model = KMeans(k=3)
julia> X_train = rand(100, 2)
julia> fit!(model, X_train)
julia> X_test = rand(10, 2)
julia> labels = predict(model, X_test)
```
"""
function predict(model::KMeans, X::Matrix{Float64})
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end