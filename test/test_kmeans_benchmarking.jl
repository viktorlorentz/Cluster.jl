using Test
using Cluster
include("utils.jl")
using .Utils

# rand_index
# what ration of data is correct
# some metrics to test benchmarking
# time
# we need to compare cluster.jl with out algorithm and check the time and accuracy

@testset "K-means Benchmarking" begin
    @testset "K-means Basic Functionality Benchmarking" begin

        for testCase in testCases
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes

                # Split the data
                train_data, train_labels, test_data, test_labels = split_data(data, labels, TRAIN_TEST_RATIO)

                # Create and fit KMeans model
                model = KMeans(k=num_classes, mode="kmeans")

                # Suppress and capture any output during fit!
                fit!(model, train_data)

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

end