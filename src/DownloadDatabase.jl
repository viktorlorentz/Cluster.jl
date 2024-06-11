using LibGit2
using GZip
using DelimitedFiles
using Clustering
using Statistics

folder_name = "datasets"
const datasetPath = joinpath(dirname(@__DIR__), folder_name)


function download_data(url, datasetPath)
    """
    Downloads benchmarking data from Github repository

    Args:
        url (string): Url to download benchmark data
        datasetPath (string): Path to store downloaded data

    Returns:
        It should download data and start data_preprocessing() function
    """
    # if there is no folder, create it and download data
    if !isdir(datasetPath)
        mkdir(datasetPath)
        LibGit2.clone(url, datasetPath)
        print("Creating folder and downloading data... \n")
    # else if folder exist but it's empty, download data
    elseif isdir(datasetPath) && isempty(readdir(datasetPath))
        print("Downloading data... \n")
        LibGit2.clone(url, datasetPath)
    # start processing if data exist
    else
        print("Data exist. Preprocessing... \n")
        data_preprocessing()
    end
end


function data_preprocessing(dataset_path=datasetPath, battery="wut", dataset="x2")
    """
    Preprocess data and labels from Gzip compressed files

    Args:
        dataset_path (String): Path to the dataset
        battery (String, optional): Dataset name directory
        dataset (String, optional): Name of the dataset files

    Returns:
        Tuple{Matrix{Float64}, Vector{Vector{Int}}}: 
            Return a matrix with columns as features and rows as datapoints
    """
    
    full_path = joinpath(dataset_path, battery, dataset)
    data_file = full_path * ".data.gz"

    # Check whether dataset exist
    if !isfile(data_file)
        error("Data file not found: $data_file")
    end

    data = Matrix(gzopen(data_file) do f
        readdlm(f, Float64)
    end)

    # load labels files
    labels = Vector{Vector{Int}}()
    i = 0
    while true
        label_file = full_path * ".labels$i.gz"
        if !isfile(label_file)
            break
        end

        push!(labels, gzopen(label_file) do f
            vec(readdlm(f, Int))
        end)
        i += 1
    end

    # Check whether we have labels for the dataset
    if isempty(labels)
        error("No label files found for dataset: $dataset")
    end

    return data, labels
end


download_data("https://github.com/Omadzze/JlData.git", datasetPath)
data, lables = data_preprocessing()
