module Cluster

using Random
# using LinearAlgebra
# using Statistics

# Initialize centroids  kmeans++ or nor mal kmeans
function init_centroids(X, K; mode::Symbol=:kmeans)

    centroids = zeros(K, size(X))

    if mode == :kmeans
        centroids = x   #TODO assign the centroids
    elseif mode == :kmeans++
        centroids = X    #TODO assign the centroids
    # else
    #     throw(ArgumentError(" wrong mode."))

    end

    return centroids
end


# KMeans  definition
mutable struct KMeans
    k::Int
    mode::Symbol
    max_try::Int
    tol::Float64
    centroids::Array{Float64, 2}
    labels_::Array{Int, 1}
   
end

# Constructor 
KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=200, tol::Float64=1e-4) = KMeans(
   
)
#  fit the model to the data
function fit!(model::KMeans, X)

end 


# Predict each point in X belongs to cluster
function predict(model::KMeans, X)
end

end
