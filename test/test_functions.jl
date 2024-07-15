using Test
using Cluster
using Suppressor
include("utils.jl")
using .Utils

@testset "Test functions" begin
    @testset "init_centroids" begin

        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]


        for mode in ["kmeans", "kmeanspp", "dc"]
            centroids = init_centroids(data, 2, mode)

            # Test if the function returns the correct number of centroids
            @test size(centroids) == (2, 2)

            # Test if the function returns the correct centroids
            centroids = init_centroids(data, 3, mode)

            # check if the centroids are in the data even in differnt order. Sort first
            @test sort(centroids, dims=1) == sort(data, dims=1)

        end

        # Test if the function throws an error when the number of centroids is invalid
        @test_throws ArgumentError init_centroids(data, 0, "kmeans")
        @test_throws ArgumentError init_centroids(data, -1, "kmeans")
        @test_throws ArgumentError init_centroids(data, 2, "invalid_mode")


    end

end