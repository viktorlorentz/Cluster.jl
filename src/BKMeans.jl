

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
    for g1 in 1:length(clusters)
        model.centroids[g1, :] = mean(clusters[g1], dims=1)[:]
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
