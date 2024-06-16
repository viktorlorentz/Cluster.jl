using Test
using Cluster
using Suppressor

@testset "BKMeans Tests" begin
    output = ""
    @testset "BKMeans Basic Functionality" begin

        for testCase in testCases
            @testset "Test Case: $(testCase.name)" begin

                data, labels, num_samples, num_features, num_classes = testCase.data, testCase.labels, testCase.num_samples, testCase.num_features, testCase.num_classes

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
        end
    end

    @testset "Argument Validation Tests" begin
        @test_throws ArgumentError BKMeans(k=0) # k must be at least 1
        @test_throws ArgumentError BKMeans(k=-1) # k must be a positive integer
        @test_throws ArgumentError BKMeans(k=2, kmeans=KMeans(k=2, max_try=-10)) # max_try must be non-negative
        @test_throws ArgumentError BKMeans(k=2, kmeans=KMeans(k=2, tol=-0.1)) # tol must be non-negative
        @test_throws TypeError BKMeans(k=2, kmeans=KMeans(k=2, tol="not_a_number")) # tol must be a number
        @test_throws TypeError BKMeans(k=2, kmeans=KMeans(k=2, max_try="not_a_number")) # max_try must be an integer
        @test_throws TypeError BKMeans(k="not_a_number") # k must be an integer
        @test_throws TypeError BKMeans(k=2, kmeans=KMeans(k=2, mode=:invalid_mode)) # mode must be a valid symbol
        @test_throws ArgumentError BKMeans(k=2, kmeans=KMeans(k=2, mode="invalid_mode")) # mode must be a valid symbol
    end

    @testset "BKMeans no print output" begin
        @test isempty(output)
    end
end