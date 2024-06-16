using Test
using Cluster
using Suppressor

@testset "K-means Tests" begin
    output = ""
    @testset "K-means Basic Functionality" begin

        for testCase in testCases
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes

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
        end
    end

    @testset "K-means no print output" begin
        @test isempty(output)
    end
end