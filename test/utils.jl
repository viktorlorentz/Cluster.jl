module Utils
using Random

export split_data, randindex

"""
    split_data(data::Array{Float64, 2}, labels::Array{Int, 1}, ratio::Float64; seed::Int=12345)

Splits the data and labels into training and test sets based on the given ratio.

# Arguments
- `data::Array{Float64, 2}`: The data matrix where each row is a data point.
- `labels::Array{Int, 1}`: The vector of labels corresponding to the data points.
- `ratio::Float64`: The ratio of the data to be used for training (e.g., 0.8 for 80% training data).
- `seed::Int`: The seed for the random number generator to ensure reproducibility (default: 12345).

# Returns
- `train_data::Array{Float64, 2}`: The training data matrix.
- `train_labels::Array{Int, 1}`: The training labels vector.
- `test_data::Array{Float64, 2}`: The test data matrix.
- `test_labels::Array{Int, 1}`: The test labels vector.
"""
function split_data(data, labels, ratio; seed=12345)
    n = size(data, 1)
    if n == 1
        return data, labels, data, labels
    end
    Random.seed!(seed)
    idx = shuffle(1:n)
    train_size = Int(floor(ratio * n))
    train_idx = idx[1:train_size]
    test_idx = idx[train_size+1:end]

    return data[train_idx, :], labels[train_idx], data[test_idx, :], labels[test_idx]
end

"""
    randindex(labels_true::Array{Int, 1}, labels_pred::Array{Int, 1})

Calculates the Rand Index, a measure of the similarity between two data clusterings.

# Arguments
- `labels_true::Array{Int, 1}`: The ground truth labels.
- `labels_pred::Array{Int, 1}`: The predicted labels.

# Returns
- `ri::Float64`: The Rand Index, a value between 0 and 1 where 1 indicates perfect agreement.
"""
function randindex(labels_true, labels_pred)
    n = length(labels_true)
    a = 0
    b = 0
    for i in 1:n-1
        for j in i+1:n
            same_cluster_true = labels_true[i] == labels_true[j]
            same_cluster_pred = labels_pred[i] == labels_pred[j]
            if same_cluster_true && same_cluster_pred
                a += 1
            elseif !same_cluster_true && !same_cluster_pred
                b += 1
            end
        end
    end
    return (a + b) / (n * (n - 1) / 2)
end
end
