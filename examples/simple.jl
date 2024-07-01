# Load the Cluster.jl package
using Pkg
Pkg.add("Plots")
Pkg.add("Random")
Pkg.add(url="https://github.com/viktorlorentz/Cluster.jl")
Pkg.activate()

using Cluster
using Random

# Set the random seed for reproducibility
Random.seed!(1234)

# Create a simple 30x2 dataset with distinct clusters
data = vcat(
    [randn(10, 2) .+ [1 1]; # Cluster 1 around [1, 1]
        randn(10, 2) .+ [5 5]; # Cluster 2 around [5, 5]
        randn(10, 2) .+ [9 1]] # Cluster 3 around [9, 1]
)

# Print the first 3 data points
println("Data: ", data[1:3, :])

# Print the shape of the data
println("Shape: ", size(data))

# Initialize the clustering algorithm
model = Cluster.KMeans(k=3, mode="kmeans") # also supports "kmeanspp"

# Fit the model to the data
Cluster.fit!(model, data)

# Print the centroids of the clusters
println("Centroids: ", model.centroids)

# Test Data
test_data = [
    1.0 1.0; # Cluster 1
    1.5 2.0; # Cluster 1
    0.5 1.5; # Cluster 1
    5.0 5.0; # Cluster 2
    8.0 9.0; # Cluster 2
    4.5 5.5; # Cluster 2
    9.0 1.0  # Cluster 3
    9.5 1.5  # Cluster 3
    8.5 1.0  # Cluster 3
]

println("Test Data: ", test_data)
println("Note: The labels might differ from the original cluster labels due to random initialization.")


# Predict the cluster for each data point
clusters = Cluster.predict(model, test_data)

# Print the resulting clusters
println("Cluster assignments: ", clusters)

# Plot the clusters (requires Plots.jl)
using Plots
scatter(data[:, 1], data[:, 2], color=model.labels, legend=false)
scatter!(model.centroids[:, 1], model.centroids[:, 2], color=:red, shape=:star, markersize=10)
savefig("simple-cluster.png")
