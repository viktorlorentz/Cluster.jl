using Cluster
using Test


"""
    create_labeled_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; seed::Int=12345)

Create labeled data for testing K-means algorithm.

# Arguments
- `num_samples_per_class::Int`: Number of samples per class.
- `num_features::Int`: Number of features (dimensions) for each sample.
- `num_classes::Int`: Number of distinct classes (clusters).
- `spread::Number`: Spread of the data points around the center (default: 10).
- `seed::Int`: Seed for random number generator (default: 12345).

# Returns
- `data`: A matrix where each row is a data point.
- `labels`: A vector of labels corresponding to the data points.
"""
function create_labeled_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; spread::Number=10, seed::Int=12345)
    Random.seed!(seed)
    data = []
    labels = []
    for i in 1:num_classes

        center = randn((1, num_features)) * (spread * log(num_classes))  # Generate a random center for each class
        class_data = [center + randn((1, num_features)) for _ in 1:num_samples_per_class] # Create data points around the center
        append!(data, class_data)  # Add the generated data points to the data list
        append!(labels, fill(i, num_samples_per_class))  # Add labels for each data point
    end
    return vcat(data...), labels  # Return the data and labels as a matrix and a vector
end

num_samples = 50  # Number of samples per class
dimensions = [1, 2, 3, 5, 20]  # Selected dimensions to test
ks = [1, 2, 3, 5, 20]  # Selected values of k to test

@testset "Cluster.jl" begin

    @testset "K-means Tests" begin
        @testset "Basic Functionality Tests" begin
            for dim in dimensions
                @testset "Dimension: $dim" begin

                    # Test convergence on simple data
                    data, _ = create_labeled_data(num_samples, dim, 2)
                    kmeans_result = kmeans(data, 2)
                    @test kmeans_result.converged == true  # Ensure the algorithm converges

                    # Test centroid initialization
                    data, _ = create_labeled_data(num_samples, dim, 2)
                    kmeans_result = kmeans(data, 2, init=:random)
                    @test length(kmeans_result.centroids) == 2  # Check if the correct number of centroids is initialized

                    # Test handling of empty clusters
                    data, _ = create_labeled_data(1, dim, 3)
                    kmeans_result = kmeans(data, 3)
                    @test isempty(findall(==(0), kmeans_result.cluster_sizes)) == true  # Ensure no cluster is empty
                end
            end
        end

        @testset "Correctness Tests" begin
            for k in ks
                for dim in dimensions
                    @testset "k: $k, Dimension: $dim" begin
                        # Test correct number of clusters
                        data, _ = create_labeled_data(num_samples, dim, k)
                        kmeans_result = kmeans(data, k)
                        @test length(unique(kmeans_result.assignments)) == k  # Check if the correct number of clusters is found

                        # Test cluster assignment
                        data, _ = create_labeled_data(num_samples, dim, k)
                        kmeans_result = kmeans(data, k)
                        for i in 1:length(data)
                            cluster = argmin([norm(data[i, :] .- centroid) for centroid in kmeans_result.centroids])
                            @test kmeans_result.assignments[i] == cluster  # Ensure each point is assigned to the nearest centroid
                        end

                        # Test centroid update
                        data, _ = create_labeled_data(num_samples, dim, k)
                        kmeans_result = kmeans(data, k)
                        for c in 1:k
                            assigned_points = data[kmeans_result.assignments.==c, :]
                            centroid = mean(assigned_points, dims=1)
                            @test isapprox(kmeans_result.centroids[c], centroid, atol=1e-6)  # Check if centroids are updated correctly
                        end
                    end
                end
            end
        end

        @testset "Argument Validation Tests" begin
            data, _ = create_labeled_data(num_samples, 2, 3)
            @test_throws ArgumentError kmeans(data, 0)  # k must be at least 1
            @test_throws ArgumentError kmeans(data, -1)  # k must be a positive integer
            @test_throws ArgumentError kmeans(data, 2, max_iters=-10)  # max_iters must be non-negative
            @test_throws ArgumentError kmeans(data, 2, tol=-0.1)  # tol must be non-negative
            @test_throws ArgumentError kmeans(data, 2, tol="not_a_number")  # tol must be a number
            @test_throws ArgumentError kmeans(data, 2, max_iters="not_a_number")  # max_iters must be an integer
            @test_throws ArgumentError kmeans(data, "not_a_number")  # k must be an integer
            @test_throws ArgumentError kmeans("not_a_matrix", 2)  # data must be a matrix
            @test_throws ArgumentError kmeans(data, 2, mode=:invalid_mode)  # mode must be a valid symbol
            @text_throws ArgumentError kmeans(data, 2, mode="invalid_mode")  # mode must be a valid symbol
        end

        @testset "Edge Cases Tests" begin
            # Single point
            single_point_data = [1.0]
            kmeans_result = kmeans(single_point_data, 1)
            @test length(kmeans_result.centroids) == 1
            @test kmeans_result.centroids[1] == single_point_data

            # Identical points
            identical_points_data = repeat([1.0, 1.0], 100)
            kmeans_result = kmeans(identical_points_data, 1)
            @test length(kmeans_result.centroids) == 1
            @test kmeans_result.centroids[1] == [1.0, 1.0]

            # Very large values
            large_values_data = [1e10 * randn(2) for _ in 1:100]
            kmeans_result = kmeans(vcat(large_values_data...), 3)
            @test kmeans_result.converged == true  # Ensure the algorithm converges

            # Very small values
            small_values_data = [1e-10 * randn(2) for _ in 1:100]
            kmeans_result = kmeans(vcat(small_values_data...), 3)
            @test kmeans_result.converged == true  # Ensure the algorithm converges

            # All points are identical
            degenerate_data = [fill(1.0, 2) for _ in 1:100]
            kmeans_result = kmeans(vcat(degenerate_data...), 3)
            @test length(unique(kmeans_result.centroids)) == 1  # Only one unique centroid expected

            # Large number of clusters: k > number of points
            data, _ = create_labeled_data(10, 2, 2)  # 20 points total
            kmeans_result = kmeans(data, 30)  # k greater than the number of points
            @test length(unique(kmeans_result.assignments)) <= 20  # Cannot have more clusters than points

            # Cluster centers overlapping
            overlapping_data = [randn(2) for _ in 1:50] .+ repeat([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], inner=[50, 1])
            kmeans_result = kmeans(vcat(overlapping_data...), 3)
            @test kmeans_result.converged == true  # Ensure the algorithm converges

            # Sparse data
            sparse_data = sprand(100, 2, 0.1)  # Sparse matrix with 10% density
            kmeans_result = kmeans(sparse_data, 3)
            @test kmeans_result.converged == true  # Ensure the algorithm converges
        end
        @testset "Benchmark against Clustering.jl" begin

            @testset "Runtime Comparison" begin


                # Compare runtime of Cluster.jl and Clustering.jl
            end

            @testset "Accuracy Comparison" begin
                # Compare accuracy of Cluster.jl and Clustering.jl
            end
        end

    end
end
