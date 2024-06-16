using Test
using Cluster
include("DataGenerator.jl")
using .DataGenerator
using Suppressor

@testset "Cluster.jl Tests" begin
    @testset "K-means" begin

        @testset "K-means Basic Functionality" begin
            num_samples = 50
            num_features = 2
            num_classes = 3

            # Generate test data
            data, labels = create_labeled_data(num_samples, num_features, num_classes)

            # Create and fit KMeans model
            model = KMeans(k=num_classes, mode="kmeans")

            # Suppress and capture any output during fit!
            output = @capture_out begin
                fit!(model, data)
            end

            # Test if the model has the correct number of centroids
            @test size(model.centroids)[1] == num_classes
            @test size(model.centroids)[2] == num_features

            # Test if the model assigns labels to all data points
            @test length(model.labels) == num_samples * num_classes

            # Test if the model converges
            @test model.centroids != zeros(Float64, 0, 0)
        end

        # Ensure there is no output
        @testset "K-means no print output" begin
            @test isempty(output)
        end
    end
    @testset "K-means++" begin
        @testset "K-means++ Basic Functionality" begin
            num_samples = 50
            num_features = 2
            num_classes = 3

            # Generate test data
            data, labels = create_labeled_data(num_samples, num_features, num_classes)

            # Create and fit KMeans++ model
            model = KMeans(k=num_classes, mode="kmeanspp")

            # Suppress and capture any output during fit!
            output = @capture_out begin
                fit!(model, data)
            end

            # Test if the model has the correct number of centroids
            @test size(model.centroids)[1] == num_classes
            @test size(model.centroids)[2] == num_features

            # Test if the model assigns labels to all data points
            @test length(model.labels) == num_samples * num_classes

            # Test if the model converges
            @test model.centroids != zeros(Float64, 0, 0)
        end

        # Ensure there is no output
        @testset "K-means++ no print output" begin
            @test isempty(output)
        end
    end
    @testset "BKMeans" begin

        @testset "BKMeans Basic Functionality" begin
            num_samples = 50
            num_features = 2
            num_classes = 3

            # Generate test data
            data, labels = create_labeled_data(num_samples, num_features, num_classes)

            # Create and fit BKMeans model
            base_model = KMeans(k=2, mode="kmeans")
            model = BKMeans(k=num_classes, kmeans=base_model)

            # Suppress and capture any output during fit!
            output = @capture_out begin
                fit!(model, data)
            end

            # Test if the model assigns labels to all data points
            @test length(model.labels) == num_samples * num_classes

            # Test if the model converges
            @test model.labels != Int[]
        end

        # Ensure there is no output
        @testset "BKMeans no print output" begin
            @test isempty(output)
        end
    end
end