using Test
include("DataGenerator.jl")
include("DownloadDatabase.jl")
using .DataGenerator

const TRAIN_TEST_RATIO = 0.85
const RAND_INDEX_THRESHOLD = 0.8

struct testCase
    data::Array{Float64}
    labels::Array{Int}
    name::String
    num_samples::Int
    num_features::Int
    num_classes::Int

    function testCase(data, labels, name, num_samples, num_features, num_classes)
        new(data, labels, name, num_samples, num_features, num_classes)
    end

    function testCase(num_samples, num_features, num_classes, name)
        data, labels = create_labeled_data(num_samples, num_features, num_classes)
        new(data, labels, name, num_samples, num_features, num_classes)
    end
end

testCases = [
    # 50 samples, 2 features, 3 classes
    testCase(50, 2, 3, "Basic functionality"),
    testCase(50, 3, 2, "3D data"),
    testCase(50, 2, 5, "More clusters than dimensions"),
    testCase(50, 2, 2, "Two clusters"),
    testCase(50, 2, 4, "Four clusters"),
    testCase(50, 2, 6, "Six clusters"),
    testCase(50, 2, 8, "Eight clusters"),
    testCase(50, 2, 10, "Ten clusters"),
    testCase(50, 2, 20, "Twenty clusters"),
    testCase(1, 2, 1, "Single 2d point"),
    testCase(10, 1, 2, "Ten 1d points"),
    testCase(10, 2, 2, "Ten 2d points"),
    testCase(10, 3, 2, "Ten 3d points"),
    testCase(1, 1, 1, "Single 1d point"),
    testCase(10, 2, 30, "Large number of clusters"),
    testCase(200, 5, 4, "Higher dimensions"),
    testCase(1000, 10, 5, "Large dataset"),
    #testCase(10000, 20, 10, "Very large dataset")
]


@testset "Cluster.jl Tests" begin
    include("test_kmeans.jl")
    include("test_kmeanspp.jl")
    include("test_bkmeans.jl")
end

folder_name = "test/datasets"
datasetPath = joinpath(dirname(@__DIR__), "test/datasets")

# Download and preprocess data
DownloadDatabase.download_data("https://github.com/Omadzze/JlData.git", datasetPath, "wut", "x3")
data, labels, dataset = DownloadDatabase.data_preprocessing(datasetPath, "wut", "circles")

# Create test cases using the downloaded data
testCasesBenchmarking = [
    testCase(data, labels[1], dataset, size(data, 1), size(data, 2), length(unique(labels[1]))),
]

@testset "Cluster.jl Benchmarking" begin
    #print(testCasesBenchmarking)
    include("test_kmeans_benchmarking.jl")
    #include("test_kmeanspp.jl")
    #include("test_bkmeans.jl")
end