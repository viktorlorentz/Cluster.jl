module Cluster

# Write your package code here.
using Random
using LinearAlgebra #norm
using Statistics #mean

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
KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=5, tol::Float64=1e-4) = KMeans(
    k,
    mode,
    max_try,
    tol,
    zeros(Float64, 0, 0),  # Initialize centroids as an empty 2D array
    Int[]                  # Initialize labels_ as an empty 1D array
    #chatgpt
)


# Initialize centroids  kmeans++ or normal kmeans
function init_centroids(X, K, mode)#; mode::Symbol=:kmeans)

    
    println("Initializing centroids...")
    if mode == 1 ## TODO initialize not by numbers but by string or similiar!!
        row,col = size(X)
        permutation = randperm(row)#gpt
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == :kmeans_pp
        # Implement kmeans++ initialization here
        centroids = X # Placeholder for kmeans++ implementation

    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return centroids
end

#  fit the model to the data
function fit!(model::KMeans, X)
   
    for i in 1:model.max_try

        D = compute_distance(X, model.centroids,model)

        labels = assign_center(D)

        new_centroids = update_centroids(X,labels,model)

        model.centroids = new_centroids
        println(model.centroids)
    end
        
end

function compute_distance(X,center,model)
    #function that gets clustercenters and all data
    #returns matrix of distance to these clustercenters

    x = size(X)
    y = size(center)

    D=zeros(x[1],y[1])

    
    for i in 1:model.k
        D[:,i] = sqrt.(sum((X .- center[i]).^2, dims=2)) #30,1 #gpt
    end
   
    return D
end

function assign_center(D)
    #returns minimum argument of the distance matrix
    
    return [argmin(D[i, :]) for i in 1:size(D, 1)]#gpt
end

function update_centroids(X, label_vector,model)
    #creates a mask based on labels
    #accesses and calculate new clustercenter with that mask

 
    my_list = Vector{Any}()
    r, c = size(X)
    
    my_m = zeros(model.k,c)

    for label in 1:model.k
        # Create a mask for the current label
        mask = label_vector .== label

        m = mean(X[mask,:],dims= 1)

        my_m[label,:] = m

    end

    return my_m
end


data_1 = [
    # Cluster 1
    1.0 1.0;
    1.5 2.0;
    1.3 1.8;
    1.2 1.2;
    0.8 0.9;
    1.0 1.1;
    1.3 1.3;
    1.2 1.3;
    1.3 1.4;
    1.5 1.5;
    
    # Cluster 2
    5.0 7.0;
    5.5 7.5;
    6.0 7.0;
    5.8 7.2;
    6.2 7.5;
    5.9 6.8;
    5.6 7.1;
    6.3 7.6;
    5.8 6.7;
    5.8 7.7;
    
    # Cluster 3
    8.0 1.0;
    8.5 1.5;
    8.3 1.2;
    8.7 1.8;
    8.4 1.4;
    8.1 1.1;
    8.6 1.6;
    8.4 1.3;
    8.3 1.5;
    8.6 1.8
]

cent = init_centroids(data_1,3,1)

K = KMeans()

K.centroids = cent

fit!(K,data_1)






# Predict each point in X belongs to cluster
function predict(model::KMeans, X)
    #
    D = compute_distance(X, model.centroids,model)

    return assign_center(D)
end


end
