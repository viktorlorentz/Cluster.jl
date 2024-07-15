module Cluster

using Random
using LinearAlgebra
using Statistics
using DocStringExtensions

include("utils.jl")
include("KMeans.jl")
include("BKMeans.jl")
include("DC.jl")


export

    # Algorithm Struts
    KMeans, BKMeans, DC,

    # Exported functions
    fit!, predict,

    # Utility functions
    init_centroids, compute_distance, assign_center, update_centroids


end