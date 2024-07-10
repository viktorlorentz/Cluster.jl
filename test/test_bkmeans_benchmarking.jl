using Test
using Cluster
include("utils.jl")
using .Utils
include("Plotting.jl")
using .Plotting

@testset "BKMeans Benchmarking" begin
    @testset "BKMeans Basic Functionality Benchmarking" begin

        """
        Test for benchmarking K-means algorithm. Calucalates Rand index whether it's above certain threshold.
        Moreover, it calculates accuracy score and if it's above 80% returns pass to the test
        """

        for testCase in testCasesBenchmarking
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes, dataset_name = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes, testCase.name

                # Split the data
                train_data, train_labels, test_data, test_labels = split_data(data, labels, TRAIN_TEST_RATIO)

                # Create and fit KMeans model
                base_model = KMeans(k=2, mode="kmeans")
                model = BKMeans(k=num_classes, kmeans=base_model)

                # Suppress and capture any output during fit!
                fit!(model, train_data)

                # Test if the model has the correct number of centroids
                #@test size(model.centroids)[1] == num_classes
                #@test size(model.centroids)[2] == num_features

                # Test if the model assigns labels to all data points
                @test length(model.labels) == size(train_data, 1)

                # Test if the model converges
                @test model.labels != Int[]

                @testset "Prediction Accuracy for BKmeans" begin
                    # check if test data is not empty
                    if num_samples != 1
                        # Predict on test data
                        @test_broken test_pred_labels = predict(model, test_data)

                        # Remap predicted labels
                        # Note: If we will get accuracy of 0 percent probably our remapping works incorrectly
                        # We need to make remapping because we have different predicted labels and test labels
                        unique_true_labels = unique(test_labels)
                        unique_pred_labels = unique(test_pred_labels)
                        label_mapping = Dict(zip(unique_pred_labels, unique_true_labels))
                        remap_pred_labels = map(x -> label_mapping[x], test_pred_labels)

                        # Calculate Rand Index
                        @test_broken ri = randindex(test_labels, remap_pred_labels)
                        println("Test Case: $(testCase.name), Rand Index: $ri")

                        # Calculate accuracy
                        accuracy = sum(remap_pred_labels .== test_labels) / length(test_labels)
                        println("Test Case: $(testCase.name), Accuracy: $accuracy")

                        # if accuracy above value then our test was passed
                        @test accuracy >= ACCURACY_SCORE
                        
                        @test_broken ri > RAND_INDEX_THRESHOLD_BENCHMARKING
                        # visualize all predictions
                        Plotting.visualize_clusters(test_data, test_labels, test_pred_labels, dataset_name, "bkmeans-figures")
                    else # single point
                        @test_broken predict(model, test_data) == [1]
                    end
                end
            end
        end
    end
end