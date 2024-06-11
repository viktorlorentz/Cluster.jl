#using Cluster
using Test
#using FileIO
include("../test/DownloadDatabase.jl")

url = "https://github.com/Omadzze/JlData.git"
folder_name_test = "test/datasets_test"
const datasetpathTest = joinpath(dirname(@__DIR__), folder_name_test)

@testset "download_data tests" begin

    # Removing folder datasets_test
    rm(datasetpathTest; force=true, recursive=true)

    # Test 1. Downloading dataset
    @test !isdir(datasetpathTest) # Directory not exist
    download_data(url, datasetpathTest)
    @test isdir(datasetpathTest) # Directory exist
    @test !isempty(readdir(datasetpathTest)) # Checks whether we have data inside directory
end
