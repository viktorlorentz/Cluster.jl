using Cluster
using Test

@testset "Cluster.jl" begin
    # Write your tests here.
    @testset "Test Interface" begin
        @testset "Kmeans Arguments" begin
        end
        @testset "Kmeans++ Arguments" begin
        end
    end

    @testset "Kmeans Dimension Tests" begin
        @testset "Kmeans 1D" begin
        end
        @testset "Kmeans 2D" begin
        end
        @testset "Kmeans 3D" begin
        end
        @testset "Kmeans 5D" begin
        end
        @testset "Kmeans 50D" begin
        end

    end

    @testset "Kmeans K Tests" begin
        @testset "Kmeans K = 1" begin
        end

        ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 1000]

        for k in ks
            @testset "Kmeans K = $k" begin
            end
        end


    @tetset "Edge Cases" begin

    end

    @testset "Benchmark against Clustering.jl"

    end


end
