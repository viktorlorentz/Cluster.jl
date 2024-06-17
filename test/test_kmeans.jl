using Test
using Cluster
using Suppressor
include("utils.jl")
using .Utils

@testset "K-means Tests" begin
    output = ""
    @testset "K-means Basic Functionality" begin

        for testCase in testCases
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes

                # Split the data
                train_data, train_labels, test_data, test_labels = split_data(data, labels, TRAIN_TEST_RATIO)

                # Create and fit KMeans model
                model = KMeans(k=num_classes, mode="kmeans")

                # Suppress and capture any output during fit!
                output = @capture_out begin
                    fit!(model, train_data)
                end

                # Test if the model has the correct number of centroids
                @test size(model.centroids)[1] == num_classes
                @test size(model.centroids)[2] == num_features

                # Test if the model assigns labels to all data points
                @test length(model.labels) == size(train_data, 1)

                # Test if the model converges
                @test model.centroids != zeros(Float64, 0, 0)
                @testset "Prediction Accuracy" begin
                    # check if test data is not empty
                    if num_samples != 1
                        # Predict on test data
                        test_pred_labels = predict(model, test_data)

                        # Calculate Rand Index
                        ri = randindex(test_labels, test_pred_labels)
                        @test ri > RAND_INDEX_THRESHOLD
                    else # single point
                        @test predict(model, test_data) == [1]
                    end
                end
            end
        end
    end

    @testset "Argument Validation Tests" begin
        @test_throws ArgumentError KMeans(k=0) # k must be at least 1
        @test_throws ArgumentError KMeans(k=-1) # k must be a positive integer
        @test_throws ArgumentError KMeans(k=2, max_try=-10) # max_try must be non-negative
        @test_throws ArgumentError KMeans(k=2, tol=-0.1) # tol must be non-negative
        @test_throws TypeError KMeans(k=2, tol="not_a_number") # tol must be a number
        @test_throws TypeError KMeans(k=2, max_try="not_a_number") # max_try must be an integer
        @test_throws TypeError KMeans(k="not_a_number") # k must be an integer
        @test_throws TypeError KMeans(k=2, mode=:invalid_mode) # mode must be a valid symbol
        @test_throws ArgumentError KMeans(k=2, mode="invalid_mode") # mode must be a valid symbol
    end

    @testset "K-means no print output" begin
        @test isempty(output)
    end
end