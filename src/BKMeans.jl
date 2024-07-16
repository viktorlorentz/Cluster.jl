"""
    mutable struct BKMeans

A mutable struct for the Bisecting KMeans clustering algorithm.

### Fields
- `k::Int`: Number of clusters.
- `kmeans::KMeans`: An instance of the KMeans struct used for bisecting.
- `labels::Array{Int,1}`: Vector of labels for each data point.
- `centroids::Matrix{Float64}`: Matrix of centroid coordinates.

### Examples
```julia-repl
julia> kmeans_model = KMeans(k=2, mode=:kmeanspp)
julia> model = BKMeans(k=3, kmeans=kmeans_model)
BKMeans(3, KMeans(2, :kmeanspp, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[]), Int64[], Matrix{Float64}(undef, 0, 0))
```
"""
mutable struct BKMeans
    k::Int
    kmeans::KMeans
    labels::Array{Int,1}
    centroids::Matrix{Float64}
end

"""
    BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode=:kmeanspp))

Constructor for the BKMeans struct.

### Input
- `k::Int`: Number of clusters (default: 3).
- `kmeans::KMeans`: An instance of the KMeans struct used for bisecting (default: KMeans(k=2, mode=:kmeanspp)).

### Output
- Returns an instance of `BKMeans`.

### Examples
```julia-repl
julia> kmeans_model = KMeans(k=2, mode=:kmeanspp)
julia> model = BKMeans(k=3, kmeans=kmeans_model)
BKMeans(3, KMeans(2, :kmeanspp, 100, 0.0001, Matrix{Float64}(undef, 0, 0), Int64[]), Int64[], Matrix{Float64}(undef, 0, 0))
```
"""
function BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode=:kmeanspp))
    if !isa(k, Int) || k <= 0
        throw(ArgumentError("k must be a positive integer"))
    end
    if !isa(kmeans, KMeans)
        throw(ArgumentError("kmeans must be an instance of KMeans"))
    end
    return BKMeans(k, kmeans, Int[], Array{Float64}(undef, 0, 0))
end

"""
    fit!(model::BKMeans, X::Matrix{Float64})

Fits the BKMeans model to the data matrix X.

### Input
- `model::BKMeans`: An instance of `BKMeans`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Modifies the `model` in-place to fit the data.

### Algorithm
1. Initialize clusters with the entire dataset.
2. While the number of clusters is less than `k`:
    a. Compute the sum of squared errors (SSE) for each cluster.
    b. Select the cluster with the highest SSE.
    c. Apply KMeans to bisect the selected cluster.
    d. Replace the selected cluster with the two resulting clusters.
3. Assign labels and centroids based on the final clusters.

### Examples
```julia-repl
julia> kmeans_model = KMeans(k=2, mode=:kmeanspp)
julia> model = BKMeans(k=3, kmeans=kmeans_model)
julia> X = rand(100, 2)
julia> fit!(model, X)
```
"""
function fit!(model::BKMeans, X::Matrix{Float64})
    if size(X, 1) == 0 || size(X, 2) == 0
        throw(ArgumentError("X must be a non-empty matrix"))
    end

    clusters = [X]

    while length(clusters) < model.k
        sse = [sum(compute_distance(clusters[i], mean(clusters[i], dims=1)) .^ 2) for i in eachindex(clusters)]
        i = argmax(sse)
        sub_model = deepcopy(model.kmeans)
        fit!(sub_model, clusters[i])
        new_clusters = [clusters[i][sub_model.labels.==1, :], clusters[i][sub_model.labels.==2, :]]

        deleteat!(clusters, i)
        for new_cluster in new_clusters
            if size(new_cluster, 1) > 0
                push!(clusters, new_cluster)
            end
        end
    end
    model.labels = Int[]
    model.centroids = zeros(Float64, length(clusters), size(X, 2))
    for g1 in eachindex(clusters)
        model.centroids[g1, :] = mean(clusters[g1], dims=1)[:]
        rows, _ = size(clusters[g1])
        for g2 in 1:rows
            push!(model.labels, g1)
        end
    end
end

"""
    predict(model::BKMeans, X::Matrix{Float64})

Predicts the cluster labels for new data points based on the fitted BKMeans model.

### Input
- `model::BKMeans`: An instance of `BKMeans`.
- `X::Matrix{Float64}`: Data matrix where rows are data points and columns are features.

### Output
- Returns a vector of predicted labels for each data point.

### Examples
```julia-repl
julia> kmeans_model = KMeans(k=2, mode=:kmeanspp)
julia> model = BKMeans(k=3, kmeans=kmeans_model)
julia> X_train = rand(100, 2)
julia> fit!(model, X_train)
julia> X_test = rand(10, 2)
julia> labels = predict(model, X_test)
```
"""
function predict(model::BKMeans, X::Matrix{Float64})
    D = compute_distance(X, model.centroids)
    return assign_center(D)
end