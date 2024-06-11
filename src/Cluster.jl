module Cluster

using Random
using LinearAlgebra
using Statistics

# KMeans  definition
mutable struct KMeans
    k::Int
    mode::Symbol
    max_try::Int
    tol::Float64
    centroids::Array{Float64,2}
    labels::Array{Int,1}
end

# Constructor
KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) = KMeans(
    k,
    mode,
    max_try,
    tol,
    zeros(Float64, 0, 0),  # Initialize centroids as an empty 2D array
    Int[]                  # Initialize labels as an empty 1D array
)

"""

        centroids = init_centroids(data, numberofcluster, mode)

    Initialize Controids for chosen algorithm.

    currently available modes are kmeans and kmeans++

    example:

    X = [
    1.0 1.0;
    1.5 2.0;
    3.0 4.0;
    5.0 6.0;
    8.0 9.0;
    10.0 11.0]

    K = 2

    mode = :kmeans

"""
function init_centroids(X, K, mode)


    if mode == 1
        #generate random vector with the length of the data, then choose first K values
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == 2
        # kmeans++ initialization
        row, col = size(X)
        permutation = randperm(row)
        idx = permutation[1:K]
        centroids = X[idx, :]

        for k in 2:K
            # dist from data points to  centroids
            D = compute_distance(X, centroids[1:k-1, :])

            # assign each  data point to  it's nearest centroids
            Dm = minimum(D, dims=2)

            # Choose the next centroid
            probabilities = vec(Dm) / sum(Dm)

            cummulative_probabilities = cumsum(probabilities)

            # perform  weighted random selection.
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

        fit!(Model,data)

    Runs KMeans algorithm for given Data and Model

"""
function fit!(model::KMeans, X)


    for i in 1:model.max_try

        D = compute_distance(X, model.centroids)

        labels = assign_center(D)

        new_centroids = update_centroids(X, labels, model)

        if maximum(sqrt.(sum((model.centroids .- new_centroids) .^ 2, dims=2))) < model.tol
            break
        end
        model.centroids = new_centroids


    end

end

"""

        compute_distance(data, centroids)

    Computes the distance from each Datapoint to each centroid and the Distances in a Matrix

    return a Matrix D size(length(datavector), number of centroids)

"""
function compute_distance(X, centroids)


    x = size(X)
    y = size(centroids)

    D = zeros(x[1], y[1])

    for i in 1:y[1]
        for j in 1:x[1]
            D[j, i] = sqrt(sum((X[j, :] .- centroids[i, :]) .^ 2))
        end
    end
    return D
end

"""
       assign_center(D)

   Returns the Minimum Argument of given Distance Matrix for every Datapoint.

"""
function assign_center(D)

    return [argmin(D[i, :]) for i in 1:size(D, 1)]
end

"""

       update_centroids(data, labelvector, model)

   Calculates new centroids based on given data and labelvector

"""
function update_centroids(X, label_vector, model)


    my_list = Vector{Any}()
    r, c = size(X)

    centroids = zeros(model.k, c)

    for label in 1:model.k
        # Create a mask for the current label
        mask = label_vector .== label

        # average the values using the mask
        centroids[label, :] = mean(X[mask, :], dims=1)

    end

    return centroids
end

"""
       predict(model,data)

   return cluster for given Datapoint

   # Examples
   ```julia-repl

   data_1 = [
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

   test_data = [
       1.1 1.1 1.2
   ]

   model = Kmeans()

   fit!(model,data)

   julia> predict(model,test_data)
   [1]
   ```
"""
function predict(model::KMeans, X)


    D = compute_distance(X, model.centroids)

    return assign_center(D)
end

end
