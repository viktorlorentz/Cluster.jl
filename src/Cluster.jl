module Cluster

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
    
)


# Initialize centroids  kmeans++ or normal kmeans
function init_centroids(X, K,model, mode)#; mode::Symbol=:kmeans)

    
    println("Initializing centroids...")
    if mode == "kmeans"
        row,col = size(X)
        permutation = randperm(row)#gpt
        idx = permutation[1:K]
        centroids = X[idx, :]

    elseif mode == "kmeanspp"
        
        row,col = size(X)
        permutation = randperm(row)#gpt
        idx = permutation[1:K]
        centroids = X[idx, :]

        for k in 2:K
        # dist from data points to  centroids
            D = compute_distance(X, centroids[1:k-1, :],model)
        # assign each  data point to  it's nearest centroids
            Dm = minimum(D, dims=2)
        # calcualte probabilities 
            probabilities = vec(Dm) / sum(Dm)
            cummulative_probabilities = cumsum(probabilities)
        # perform  weighted random selection.
            r_num = rand()
            next_centroid_ind = searchsortedfirst(cummulative_probabilities, r_num)
        # choose the next centroid
            centroids[k, :] = X[next_centroid_ind, :]       
        end 

    else
        throw(ArgumentError("Unknown mode: $mode"))
    end

    return centroids
end

#  fit the model to the data
function fit!(model::KMeans, X)
    model.centroids = init_centroids(data_1,3, K, "kmeans" )
    for i in 1:model.max_try
        println(model.centroids)
        D = compute_distance(X, model.centroids,model)
        println(D)
        labels = assign_center(D)
        println(labels)

        new_centroids = update_centroids(X,labels,model)  
        model.labels_ = labels

        if maximum(sqrt.(sum((model.centroids .- new_centroids).^2, dims=2))) < model.tol
            break
        end
        model.centroids = new_centroids
    
    end
        
end

function compute_distance(X,center,model)
    #function that gets clustercenters and all data
    #returns matrix of distance to these clustercenters

    x = size(X)
    y = size(center)
    D=zeros(Float64,x[1],y[1])
    for i in 1:y[1]
        for j in 1:x[1]
            D[j,i] = norm((X[j, :] .- center[i, :]))
        end
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

function predict(model::KMeans, X)
    #
    D = compute_distance(X, model.centroids,model)

    return assign_center(D)
end

data_1 = [
    # Cluster 1
    1.0 1.0;
    1.5 2.0;
    1.3 1.8;
    1.2 1.2;
    0.8 0.9;
    # 1.0 1.1;
    # 1.3 1.3;
    # 1.2 1.3;
    # 1.3 1.4;
    # 1.5 1.5;
    
    # Cluster 2
    5.0 7.0;
    5.5 7.5;
    6.0 7.0;
    5.8 7.2;
    6.2 7.5;
    # 5.9 6.8;
    # 5.6 7.1;
    # 6.3 7.6;
    # 5.8 6.7;
    # 5.8 7.7;
    
    # Cluster 3
    8.0 1.0;
    8.5 1.5;
    8.3 1.2;
    8.7 1.8;
    8.4 1.4;
    # 8.1 1.1;
    # 8.6 1.6;
    # 8.4 1.3;
    # 8.3 1.5;
    # 8.6 1.8
]

K = KMeans()

fit!(K,data_1)



println(K.labels_)




end
