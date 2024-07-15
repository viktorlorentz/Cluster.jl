### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d7a3fe86-4355-434c-a33e-a0b64a2a546a
using Pkg

# ╔═╡ 2dba0803-7a19-4fae-a6ea-de6f13423129
begin
    Pkg.add("Plots")
    Pkg.add("Random")
    Pkg.add("PlutoUI")
    Pkg.activate()
end

# ╔═╡ d4687371-815f-4c67-9bdb-44a28c0d8122
Pkg.add(url="https://github.com/viktorlorentz/Cluster.jl")

# ╔═╡ f933995d-e6f2-415f-89c6-d21247d371e9
using Cluster

# ╔═╡ 8a5b6f90-2a68-4e6c-9d8c-3b49ca4ae4b5
using Plots

# ╔═╡ 557bdb83-40db-4383-9baa-ddf144bd6f8f
using Random

# ╔═╡ b55a6ff5-4035-46cf-926e-8142c533843b
using PlutoUI

# ╔═╡ 9516a229-1b61-4586-a1d6-4fa684e8811b

function create_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; spread::Number=10, seed::Int=12345)
    Random.seed!(seed)  # Set the random seed for reproducibility
    data = []  # Initialize an empty list for data points
    labels = []  # Initialize an empty list for labels
    for i in 1:num_classes
        center = randn((1,num_features)) * (spread * log(num_classes))  # Generate a random center for each class
        class_data = [center + randn((1,num_features)) for _ in 1:num_samples_per_class]  # Create data points around the center
        append!(data, class_data)  # Add the generated data points to the data list
        append!(labels, fill(i, num_samples_per_class))  # Add labels for each data point
    end
    return vcat(data...), labels  # Return the data and labels as a matrix and a vector
end

# ╔═╡ 6304380c-7139-4e21-97eb-f49e37c9b325
begin
# Function to plot data
function plot_sample_data!(ax, data, labels, num_classes, title = "Sample Data")
    for i in 1:num_classes
        class_data = data[labels .== i, :]
        scatter!(ax, class_data[:, 1], class_data[:, 2], label="Class $i")
    end
    xlabel!(ax, "Feature 1")
    ylabel!(ax, "Feature 2")
    title!(ax, title)
end

# Function to create subplots
function create_subplots(data, labels, predicted_labels, num_classes, model)
    fig = plot(layout = (3, 1), size=(800, 1400))
    plot_sample_data!(fig[1], data, labels, num_classes, "Sample Data")
    plot_sample_data!(fig[2], data, predicted_labels, num_classes, "Prediction")

    # Mark wrong predictions with an "x"
    wrong_predictions = labels .!= predicted_labels
    scatter!(fig[2], data[wrong_predictions, 1], data[wrong_predictions, 2], marker=:x, color=:black, label="Wrong Prediction")

    # Create a grid of points
    x_min, x_max = minimum(data[:, 1]), maximum(data[:, 1])
    y_min, y_max = minimum(data[:, 2]), maximum(data[:, 2])
    x_grid = range(x_min, x_max, length=100)
    y_grid = range(y_min, y_max, length=100)
    grid_x, grid_y = repeat(x_grid, inner=100), repeat(y_grid, outer=100)
    grid_points = hcat(grid_x, grid_y)

    # Predict labels for the grid points
    grid_labels = Cluster.predict(model, grid_points)

    # Plot the color map using scatter
    scatter!(fig[3], grid_x, grid_y, color=grid_labels, marker=:rect, markerstrokewidth=0,ms=4, title="Decision Boundary", legend=false)

    return fig
end

# Function to match cluster labels
function match_labels(true_labels, predicted_labels, num_classes)
    contingency_matrix = zeros(Int, num_classes, num_classes)
    for i in 1:num_classes
        for j in 1:num_classes
            contingency_matrix[i, j] = sum((true_labels .== i) .& (predicted_labels .== j))
        end
    end

    label_mapping = Dict{Int, Int}()
    for i in 1:num_classes
        max_overlap = argmax(contingency_matrix[:, i])
        label_mapping[i] = max_overlap
    end

    matched_labels = [label_mapping[label] for label in predicted_labels]
    return matched_labels
end
end

# ╔═╡ b62db7d3-ced4-4d36-8f36-308bd3ab9e00
md"# Cluster.jl Demo"

# ╔═╡ 35dc71ef-2fa3-4ec9-953e-61548f29562c
md"Seed:\
$(@bind seed Slider(0:1000000, default=123, show_value=true))\
Samples per class:\
$(@bind num_samples_per_class Slider(5:200, default=20, show_value=true))\
Features/ Dimensions:\
$(@bind num_features Slider(1:2 , default=2, show_value=true))\
Classes:\
$(@bind num_classes Slider(1:25 , default=3, show_value=true))\
Spread of cluster:\
$(@bind spread Slider(0:0.1:10 , default=4, show_value=true)))\
Clustering algorithm:\
$(@bind clustering_algorithm Select([\"kmeans\", \"kmeanspp\", \"bkmeans\"]))\
"

# ╔═╡ 726f9420-db77-442a-9fa3-09a1dad985c4
begin
    # Generate sample data
    data, labels = create_data(num_samples_per_class, num_features, 			num_classes, spread = spread, seed = seed)

    # Select clustering algorithm
    if clustering_algorithm == "kmeans"
        model = Cluster.KMeans(k=num_classes, mode="kmeans")

    elseif clustering_algorithm == "kmeanspp"
        model = Cluster.KMeans(k=num_classes, mode="kmeanspp")

    elseif clustering_algorithm == "bkmeans"
        base_model = KMeans(k=2, mode="kmeans")
        model = BKMeans(k=num_classes, kmeans=base_model)
    end

    # Fit the model
    Cluster.fit!(model, data)

    # Match the predicted labels to the true labels
    matched_labels = match_labels(labels, model.labels, num_classes)

    # Create subplots
    create_subplots(data, labels, matched_labels, num_classes, model)
end

# ╔═╡ Cell order:
# ╠═d7a3fe86-4355-434c-a33e-a0b64a2a546a
# ╠═d4687371-815f-4c67-9bdb-44a28c0d8122
# ╠═f933995d-e6f2-415f-89c6-d21247d371e9
# ╟─2dba0803-7a19-4fae-a6ea-de6f13423129
# ╟─8a5b6f90-2a68-4e6c-9d8c-3b49ca4ae4b5
# ╟─557bdb83-40db-4383-9baa-ddf144bd6f8f
# ╟─b55a6ff5-4035-46cf-926e-8142c533843b
# ╟─9516a229-1b61-4586-a1d6-4fa684e8811b
# ╟─6304380c-7139-4e21-97eb-f49e37c9b325
# ╟─b62db7d3-ced4-4d36-8f36-308bd3ab9e00
# ╟─35dc71ef-2fa3-4ec9-953e-61548f29562c
# ╠═726f9420-db77-442a-9fa3-09a1dad985c4
