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

# ╔═╡ fd7a69ee-de8d-45c4-99d3-fa4b72327d7c
Pkg.add(path="/Users/viktorlorentz/Development/ce/julia/Cluster.jl")

# ╔═╡ 8a5b6f90-2a68-4e6c-9d8c-3b49ca4ae4b5
using Plots

# ╔═╡ 557bdb83-40db-4383-9baa-ddf144bd6f8f
using Random

# ╔═╡ b55a6ff5-4035-46cf-926e-8142c533843b
using PlutoUI

# ╔═╡ 3204dce5-432c-4cd9-8586-60785e233212
using Cluster

# ╔═╡ 9516a229-1b61-4586-a1d6-4fa684e8811b

function create_labeled_data(num_samples_per_class::Int, num_features::Int, num_classes::Int; spread::Number=10, seed::Int=12345)
    Random.seed!(seed)  # Set the random seed for reproducibility
    data = []  # Initialize an empty list for data points
    labels = []  # Initialize an empty list for labels
    for i in 1:num_classes
	
        center = randn((1,num_features)) * (spread * log(num_classes))  # Generate a random center for each class

        class_data = [center + randn((1,num_features)) for _ in 1:num_samples_per_class]
	# Create data points around the center
        append!(data, class_data)  # Add the generated data points to the data list
        append!(labels, fill(i, num_samples_per_class))  # Add labels for each data point
    end
    return vcat(data...), labels  # Return the data and labels as a matrix and a vector
end


# ╔═╡ 0e11467a-ec17-43a0-bd29-4711dd5d8aee
@bind num_samples_per_class Slider(5:200, default=20)

# ╔═╡ 146996c3-ea17-49e2-930e-40e1e04d7e49
@bind num_features Slider(1:4 , default=2)

# ╔═╡ c060f1ea-0a86-4fa3-980a-57b08912e87e
@bind num_classes Slider(1:50 , default=3)

# ╔═╡ 853c6935-f73c-4076-98af-0896ca0cdae2
@bind spread Slider(0:0.1:10 , default=10)

# ╔═╡ 726f9420-db77-442a-9fa3-09a1dad985c4
begin
	# Generate sample data
	data, labels = create_labeled_data(num_samples_per_class, num_features, num_classes, spread = spread)

	
	# Plot the sample data
	function plot_sample_data(data, labels, num_classes)
	    scatter()
	    for i in 1:num_classes
	        class_data = data[labels .== i,:]
	        scatter!(class_data[:,1],class_data[:,2], label="Class $i")
	    end
	    xlabel!("Feature 1")
	    ylabel!("Feature 2")
	    title!("Sample Data")
	end
	
	plot_sample_data(data, labels, num_classes)
end

# ╔═╡ 3197f17d-61c9-430c-9025-8e1de79e8f57
Cluster.KMeans()

# ╔═╡ 469dac6d-3e5c-4431-8c23-1f61c02e1935
using Pkg

# ╔═╡ 38f8de51-cbea-40ac-96b7-8c357c54feb6
using Revise, Pkg

# ╔═╡ Cell order:
# ╠═8a5b6f90-2a68-4e6c-9d8c-3b49ca4ae4b5
# ╠═557bdb83-40db-4383-9baa-ddf144bd6f8f
# ╠═b55a6ff5-4035-46cf-926e-8142c533843b
# ╠═9516a229-1b61-4586-a1d6-4fa684e8811b
# ╠═0e11467a-ec17-43a0-bd29-4711dd5d8aee
# ╠═146996c3-ea17-49e2-930e-40e1e04d7e49
# ╠═c060f1ea-0a86-4fa3-980a-57b08912e87e
# ╠═853c6935-f73c-4076-98af-0896ca0cdae2
# ╠═726f9420-db77-442a-9fa3-09a1dad985c4
# ╠═38f8de51-cbea-40ac-96b7-8c357c54feb6
# ╠═469dac6d-3e5c-4431-8c23-1f61c02e1935
# ╠═fd7a69ee-de8d-45c4-99d3-fa4b72327d7c
# ╠═3204dce5-432c-4cd9-8586-60785e233212
# ╠═3197f17d-61c9-430c-9025-8e1de79e8f57
