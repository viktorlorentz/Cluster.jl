using LibGit2
using GZip
using DelimitedFiles
using Clustering

folder_name = "datasets"
const datasetpath = joinpath(dirname(@__DIR__), folder_name)

# Downloading dataset to test our algorithm from Github
function download_data(url, datasetPath)
    if !isdir(datasetPath)
        mkdir(datasetPath)
        LibGit2.clone(url, datasetPath)
        print("Creating folder and downloading data... \n")
    elseif isdir(datasetPath) && isempty(readdir(datasetPath))
        print("Downloading data... \n")
        LibGit2.clone(url, datasetPath)
    else
        print("Data exist \n")
        data_preprocessing("wut", "x2")
    end
end

# code taken from https://github.com/HolyLab/ClusteringBenchmarks.jl 
# Preprocess our data after downloading. Note that data are in .data.gz
# format so we need to extract it
function data_preprocessing(battery, dataset)
    basename = joinpath(datasetpath, battery, dataset)
    datafile = basename * ".data.gz"
    #print(datafile)
    isfile(datafile) || error("no such file: $datafile")
    data = Matrix(gzopen(datafile) do f
        readdlm(f, Float64)'
    end)
    labels = Vector{Int}[]
    i = 0
    while true
        labelfile = basename * ".labels$i.gz"
        if !isfile(labelfile)
            break
        end
        push!(labels, gzopen(labelfile) do f
            vec(readdlm(f, Int))
        end)
        i += 1
    end
    isempty(labels) && error("no label files found for $basename")
    return data, labels
end

# make calculation on data that was downloaded
function data_calculation(data; noise_factor=1e-6)
    mean = mean(data, dims=2)
    centroid = data .- mean
    std_dev = std(centroid, dims=2)
    scaled = centroid ./ std_dev
    noise = noise_factor * randn(size(data))
    return scaled .+ noise
end


download_data("https://github.com/Omadzze/JlData.git", datasetpath)