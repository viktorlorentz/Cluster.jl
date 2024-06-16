using Test
using Cluster
include("DataGenerator.jl")
using .DataGenerator

@testset "Cluster.jl Tests" begin

    @testset "K-means Basic Functionality" begin
        num_samples = 50
        num_features = 2
        num_classes = 3

        # Generate test data
        data, labels = create_labeled_data(num_samples, num_features, num_classes)

        # Create and fit KMeans model
        model = KMeans(k=num_classes, mode="kmeans")
        fit!(model, data)

        # Test if the model has the correct number of centroids
        @test size(model.centroids)[1] == num_classes
        @test size(model.centroids)[2] == num_features

        # Test if the model assigns labels to all data points
        @test length(model.labels) == num_samples * num_classes

        # Test if the model converges
        @test model.centroids != zeros(Float64, 0, 0)
    end

    @testset "K-means++ Basic Functionality" begin
        num_samples = 50
        num_features = 2
        num_classes = 3

        # Generate test data
        data, labels = create_labeled_data(num_samples, num_features, num_classes)

        # Create and fit KMeans++ model
        model = KMeans(k=num_classes, mode="kmeanspp")
        fit!(model, data)

        # Test if the model has the correct number of centroids
        @test size(model.centroids)[1] == num_classes
        @test size(model.centroids)[2] == num_features

        # Test if the model assigns labels to all data points
        @test length(model.labels) == num_samples * num_classes

        # Test if the model converges
        @test model.centroids != zeros(Float64, 0, 0)
    end

    @testset "BKMeans Basic Functionality" begin
        num_samples = 50
        num_features = 2
        num_classes = 3

        # Generate test data
        data, labels = create_labeled_data(num_samples, num_features, num_classes)

        # Create and fit BKMeans model
        base_model = KMeans(k=2, mode="kmeans")
        model = BKMeans(k=num_classes, kmeans=base_model)
        fit!(model, data)

        # Test if the model assigns labels to all data points
        @test length(model.labels) == num_samples * num_classes

        # Test if the model converges
        @test model.labels != Int[]
    end

end