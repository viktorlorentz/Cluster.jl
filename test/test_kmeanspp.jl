using Test
using Cluster
using Suppressor

@testset "K-means++ Tests" begin
    output = ""
    @testset "K-means++ Basic Functionality" begin

        for testCase in testCases
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes

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
        end
    end

    @testset "Argument Validation Tests" begin
        @test_throws ArgumentError KMeans(k=0, mode="kmeanspp") # k must be at least 1
        @test_throws ArgumentError KMeans(k=-1, mode="kmeanspp") # k must be a positive integer
        @test_throws ArgumentError KMeans(k=2, max_try=-10, mode="kmeanspp") # max_try must be non-negative
        @test_throws ArgumentError KMeans(k=2, tol=-0.1, mode="kmeanspp") # tol must be non-negative
        @test_throws TypeError KMeans(k=2, tol="not_a_number", mode="kmeanspp") # tol must be a number
        @test_throws TypeError KMeans(k=2, max_try="not_a_number", mode="kmeanspp") # max_try must be an integer
        @test_throws TypeError KMeans(k="not_a_number", mode="kmeanspp") # k must be an integer
        @test_throws TypeError KMeans(k=2, mode=:invalid_mode) # mode must be a valid symbol
        @test_throws ArgumentError KMeans(k=2, mode="invalid_mode") # mode must be a valid symbol
    end

    @testset "K-means++ no print output" begin
        @test isempty(output)
    end
end