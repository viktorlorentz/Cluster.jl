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

    @testset "BKMeans no print output" begin
        @test isempty(output)
    end
end