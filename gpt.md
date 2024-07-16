Our use of generative models is documented below:

# CoPilot
- Ideation of additional test cases
- Refactoring of Docstrings
- Help with plotting options
- Refactoring parameter types consistently

# ChatGPT

Chat Gpt was used to get a general overwiew for the structure the code should have(fit!, and update_centroids function). What functions are needed and how to call them in the training loop. The code was generated in Python and then translated into Julia.

Such as here in the update_centroids function of the DC algortihm.

function update_centroids(X::Matrix{Float64}, label_vector::Vector{Int64}, model::DC; delta=0.0001)
    new_centers = zeros(model.k, size(X, 2))
    for i in 1:model.k
        mask = label_vector .== i#gpt
        cluster_points = X[mask, :]
        num_points = size(cluster_points, 1)#gpt
        if num_points == 0#gpt
            continue#gpt
        end#gpt
        #translated gpt
        log_potentials = zeros(num_points)
        for j in 1:num_points
            d = cluster_points[j, :]
            temp = (cluster_points .- transpose(d)) .^ 2
            temp2 = sum(temp, dims=2) .+ delta
            log_potential = sum(log.(temp2))
            log_potentials[j] = log_potential
        end
        min_index = argmin(log_potentials)
        new_centers[i, :] = cluster_points[min_index, :]
        #translated gpt end
    end



Furthermore Chat Gpt was used to find some smart ways to implement simple things such as:

function assign_center(D)
    """
    Returns the Minimum Argument of given Distance Matrix for every Datapoint. 
    This is used further down as a labelvector
    """
    return [argmin(D[i, :]) for i in 1:size(D, 1)]#gpt
end

if mode == 1 ## TODO initialize not by numbers but by string or similiar!!
    row,col = size(X)
    permutation = randperm(row)#gpt
    idx = permutation[1:K]
    centroids = X[idx, :]
end


Moreover, we used GPT to preprocess the data in a benchmark. Code below:
function data_preprocessing(dataset_path, battery, dataset)
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

    return data, labels, dataset
    
end

