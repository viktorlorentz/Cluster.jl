module DataGenerator

using Random

export create_labeled_data

"""
    create_labeled_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; spread::Number=10, seed::Int=12345)

Create labeled data for testing K-means algorithm.

# Arguments
- `num_samples_per_class::Int`: Number of samples per class.
- `num_features::Int`: Number of features (dimensions) for each sample.
- `num_classes::Int`: Number of distinct classes (clusters).
- `spread::Number`: Spread of the data points around the center (default: 10).
- `seed::Int`: Seed for random number generator (default: 12345).

# Returns
- `data`: A matrix where each row is a data point.
- `labels`: A vector of labels corresponding to the data points.
"""
function create_labeled_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; spread::Number=10, seed::Int=12345)
    Random.seed!(seed)
    data = []
    labels = []
    for i in 1:num_classes
        center = randn((1, num_features)) * (spread * log(num_classes)) # Generate a random center for each class
        class_data = [center + randn((1, num_features)) for _ in 1:num_samples_per_class] # Create data points around the center
        append!(data, class_data) # Add the generated data points to the data list
        append!(labels, fill(i, num_samples_per_class)) # Add labels for each data point
    end
    return vcat(data...), labels # Return the data and labels as a matrix and a vector
end

end