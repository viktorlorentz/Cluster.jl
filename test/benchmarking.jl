include("DownloadDatabase.jl")

const ACCURACY_SCORE = 0.2
const RAND_INDEX_THRESHOLD_BENCHMARKING = 0.2

function test_benchmark()

    """
    Function which runs our benchmarking algorithm on a dataset.
    We have different batteries that will be running from a "wut" dataset.
    """

    global testCasesBenchmarking = []

    datasetPath = joinpath(dirname(@__DIR__), "test/datasets")

    batteries = [
        "circles", "cross", "graph", "isolation", "labirynth",
        "mk1", "mk2", "mk3", "mk4", "olympic", "smile", "stripes",
        "trajectories", "trapped_lovers", "twosplashes", "windows",
        "x1", "x2", "x3", "z1", "z2", "z3"
    ]

    for battery in batteries
        DownloadDatabase.download_data("https://github.com/Omadzze/JlData.git", datasetPath, "wut", battery)
        data, labels = DownloadDatabase.data_preprocessing(datasetPath, "wut", battery)

        # Create test cases using the downloaded data
        push!(testCasesBenchmarking, testCase(data, labels[1], battery, size(data, 1), size(data, 2), length(unique(labels[1]))))
    end

end

test_benchmark()

@testset "Kmeans Benchmarking" begin
    include("test_kmeans_benchmarking.jl")
end

@testset "Kmeans++ Benchmarking" begin
    include("test_kmeansapp_benchmarking.jl")
end

@testset "BKmeans Benchmarking" begin
    include("test_bkmeans_benchmarking.jl")
end


function test_benchmark()

    """
    Function which runs our benchmarking algorithm on a dataset.
    We have different batteries that will be running from a "wut" dataset.
    """

    global testCasesBenchmarking = []

    datasetPath = joinpath(dirname(@__DIR__), "test/datasets")

    batteries = [
        "circles", "cross", "graph", "isolation", "labirynth",
        "mk1", "mk2", "mk3", "mk4", "olympic", "smile", "stripes",
        "trajectories", "trapped_lovers", "twosplashes", "windows",
        "x1", "x2", "x3", "z1", "z2", "z3"
    ]

    for battery in batteries
        DownloadDatabase.download_data("https://github.com/Omadzze/JlData.git", datasetPath, "wut", battery)
        data, labels = DownloadDatabase.data_preprocessing(datasetPath, "wut", battery)

        # Create test cases using the downloaded data
        push!(testCasesBenchmarking, testCase(data, labels[1], battery, size(data, 1), size(data, 2), length(unique(labels[1]))))
    end

end

test_benchmark()

@testset "Kmeans Benchmarking" begin
    include("test_kmeans_benchmarking.jl")
end

@testset "Kmeans++ Benchmarking" begin
    include("test_kmeansapp_benchmarking.jl")
end

@testset "BKmeans Benchmarking" begin
    include("test_bkmeans_benchmarking.jl")
end