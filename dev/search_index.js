var documenterSearchIndex = {"docs":
[{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"CurrentModule = Cluster","category":"page"},{"location":"benchmarks/#Benchmarks","page":"Benchmarks","title":"Benchmarks","text":"","category":"section"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"For benchmark our project we used the dataset provided by Marek Gagolewski. Specificallyh we are using wut dataset. Below you can see the table with our metrics of rand index and accuracy of our algorithms: kmeans and kmeans++ which was benchmarked on wut dataset. Also, we are providing visualization in order to give insights what data we used.","category":"page"},{"location":"benchmarks/#Comparison-table","page":"Benchmarks","title":"Comparison table","text":"","category":"section"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"Test Case Kmeans Rand Index Kmeans Accuracy Kmeans++ Rand Index Kmeans++ Accuracy Bkmeans Rand Index Bkmeans Accuracy\ncircles 1.00 1.00 1.00 1.00 1.0 1.0\ncross 0.5905 0.5967 0.5905 0.5967 0.7362 0.7167\ngraph 0.8835 0.2373 0.8852 0.6133 0.8687 0.3147\nisolation 0.5564 0.3474 0.5557 0.3444 0.5543 0.3489\nlabirynth 0.7526 0.4511 0.7526 0.4511 0.7839 0.2350\nmk1 1.00 1.00 1.00 1.00 1.0 1.0\nmk2 0.5010 0.5467 0.5010 0.5467 0.5010 0.5467\nmk3 0.9740 0.9778 0.9740 0.9778 0.9740 0.9778\nmk4 0.5682 0.2311 0.5766 0.2400 0.6224 0.5111\nolympic 0.7122 0.2907 0.7306 0.2707 0.7110 0.2840\nsmile 0.8263 0.4267 0.8379 0.5267 0.8335 0.5733\nstripes 0.4998 0.5160 0.4998 0.5160 0.4998 0.5160\ntrajectories 1.00 1.00 1.00 1.00 1.0 1.0\ntrapped_lovers 0.5973 0.5840 0.5955 0.5813 0.5833 0.4533\ntwosplashes 0.6186 0.7500 0.6186 0.7500 0.6186 0.7500\nwindows 0.5745 0.3333 0.5677 0.3535 0.5820 0.4855\nx1 1.00 1.00 1.00 1.00 1.0 1.0\nx2 0.8301 0.8889 0.6078 0.6667 0.6078 0.6667\nx3 0.9259 0.9286 0.9339 0.6429 0.9339 0.6429\nz1 0.6256 0.4828 0.6256 0.4828 0.6650 0.5862\nz2 0.7803 0.3333 0.7803 0.3333 0.7772 0.2963\nz3 1.00 1.00 1.00 1.00 1.0 1.0","category":"page"},{"location":"benchmarks/#Visualization-K-means","page":"Benchmarks","title":"Visualization K-means","text":"","category":"section"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"<div style=\"text-align: center;\">\n  <img src=\"https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/circles.png?raw=true\" alt=\"Clustering Circles\">\n  <p>Figure 1: Clustering Circles with K-means</p>\n</div>\n\n<div style=\"text-align: center;\">\n  <img src=\"https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/mk1.png?raw=true\" alt=\"Mk1\">\n  <p>Figure 2: Mk1 with K-means</p>\n</div>\n\n<div style=\"text-align: center;\">\n  <img src=\"https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/mk3.png?raw=true\" alt=\"Mk1\">\n  <p>Figure 3: Mk3 with K-means</p>\n</div>\n\n<div style=\"text-align: center;\">\n  <img src=\"https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/z3.png?raw=true\" alt=\"Mk1\">\n  <p>Figure 4: Z3 with K-means</p>\n</div>\n\n<div style=\"text-align: center;\">\n  <img src=\"https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/trajectories.png?raw=true\" alt=\"Mk1\">\n  <p>Figure 5: Trajectories with K-means</p>\n</div>","category":"page"},{"location":"api/","page":"API","title":"API","text":"CurrentModule = Cluster","category":"page"},{"location":"api/#API-Reference","page":"API","title":"API Reference","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Pages = [\"api.md\"]\nDepth = 5","category":"page"},{"location":"api/#Utility-functions","page":"API","title":"Utility functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"All functions used by all algorithms.","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Cluster]\nPages = [\"utils.jl\"]","category":"page"},{"location":"api/#Cluster.init_centroids-Tuple{Matrix{Float64}, Int64, String}-api","page":"API","title":"Cluster.init_centroids","text":"Initialize centroids for the chosen algorithm.\n\nArguments\n\nX::Array{Float64,2}: The input data matrix where each row is a data point.\nK::Int: The number of clusters.\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++).\n\nFields\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nK = 2\nmode = :kmeans\ncentroids = init_centroids(X, K, mode)\n\n\n\n\n\n","category":"method"},{"location":"api/#KMeans-/-KMeans-Clustering-Algorithm","page":"API","title":"KMeans / KMeans++ Clustering Algorithm","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Initializes centroids using either random selection or KMeans++.\nIteratively assigns points to the nearest centroid.\nUpdates centroids based on the mean of assigned points.\nStops when centroids converge or after a maximum number of iterations.","category":"page"},{"location":"api/","page":"API","title":"API","text":"References:","category":"page"},{"location":"api/","page":"API","title":"API","text":"Scikit-Learn KMeans Documentation","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Cluster]\nPages = [\"KMeans.jl\"]","category":"page"},{"location":"api/#Cluster.BKMeans-Tuple{}-api-2","page":"API","title":"Cluster.BKMeans","text":"BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode=\"kmeans\"))\n\nCreates a new BKMeans clustering model.\n\nKeyword Arguments:\n\nk::Int: The number of clusters (default: 3).\nkmeans::KMeans: An instance of the KMeans model used for cluster splitting (default: KMeans with 2 clusters).\n\nReturns: A BKMeans model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.KMeans-Tuple{}-api-2","page":"API","title":"Cluster.KMeans","text":"KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> KMeans\n\nCreates a new KMeans clustering model.\n\nKeyword Arguments\n\nk::Int: The number of clusters (default: 3).\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++, default: :kmeans).\nmax_try::Int: The maximum number of iterations for the algorithm (default: 100).\ntol::Float64: The tolerance for convergence (default: 1e-4).\n\nReturns\n\nA KMeans model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.assign_center-Tuple{Matrix{Float64}}-api-2","page":"API","title":"Cluster.assign_center","text":"assign_center(D) -> Array\n\nReturns the index of the nearest centroid for each data point.\n\nArguments\n\nD: The distance matrix.\n\nReturns\n\nAn array of indices indicating the nearest centroid for each data point.\n\nExamples\n\nD = [\n    0.0 2.0;\n    1.0 1.0;\n    2.0 0.0\n]\nlabels = assign_center(D)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.compute_distance-Tuple{Matrix{Float64}, Matrix{Float64}}-api-2","page":"API","title":"Cluster.compute_distance","text":"compute_distance(data, centroids) -> Array\n\nComputes the distance from each data point to each centroid.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\ncentroids: The current centroids.\n\nReturns\n\nA distance matrix D of size (number of data points, number of centroids), where D[i, j] is the distance from data point i to centroid j.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0\n]\ncentroids = [\n    1.0 1.0;\n    3.0 4.0\n]\nD = compute_distance(X, centroids)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{BKMeans, Matrix{Float64}}-api-2","page":"API","title":"Cluster.fit!","text":"fit!(model::BKMeans, X)\n\nRuns the Bisecting KMeans algorithm for the given data and model.\n\nArguments:\n\nmodel::BKMeans: The BKMeans model to be trained.\nX: The input data matrix where each row is a data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{KMeans, Matrix{Float64}}-api-2","page":"API","title":"Cluster.fit!","text":"fit!(model::KMeans, X)\n\nRuns the KMeans algorithm for the given data and model.\n\nArguments\n\nmodel::KMeans: The KMeans model to be trained.\nX: The input data matrix where each row is a data point.\n\nExamples\n\nmodel = KMeans(k=3, mode=:kmeans)\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nfit!(model, X)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{BKMeans, Any}-api-2","page":"API","title":"Cluster.predict","text":"predict(model::BKMeans, X) -> Array\n\nReturns the cluster labels for the given data points using the trained BKMeans model.\n\nArguments:\n\nmodel::BKMeans: The trained BKMeans model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns: An array of cluster labels for each data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{KMeans, Matrix{Float64}}-api-2","page":"API","title":"Cluster.predict","text":"predict(model, data) -> Array\n\nReturns the cluster labels for the given data points.\n\nArguments\n\nmodel::KMeans: The trained KMeans model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns\n\nAn array of cluster labels for each data point.\n\nExamples\n\ndata = [\n    # Cluster 1\n    1.0 1.0 1.5;\n    1.5 2.0 1.6;\n    1.3 1.8 1.4;\n    # Cluster 2\n    5.0 7.0 3.5;\n    5.5 7.5 3.5;\n    6.0 7.0 3.5;\n    # Cluster 3\n    8.0 1.0 6.5;\n    8.5 1.5 6.5;\n    8.3 1.2 7.5;\n]\ntest_data = [1.1 1.1 1.2]\nmodel = KMeans(k=3)\nfit!(\n\nmodel, data)\nlabels = predict(model, test_data)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.update_centroids-Tuple{Matrix{Float64}, Vector{Int64}, KMeans}-api-2","page":"API","title":"Cluster.update_centroids","text":"update_centroids(data, labelvector, model) -> Array\n\nCalculates new centroids based on the given data and label vector.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\nlabelvector: The current labels of the data points.\nmodel: The KMeans model.\n\nReturns\n\nAn array of new centroids.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0\n]\nlabels = [1, 1, 2, 2]\nmodel = KMeans(k=2)\nnew_centroids = update_centroids(X, labels, model)\n\n\n\n\n\n","category":"method"},{"location":"api/#Bisecting-KMeans-Clustering-Algorithm","page":"API","title":"Bisecting KMeans Clustering Algorithm","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Starts with a single cluster containing all data points.\nRecursively splits clusters based on the highest SSE until k clusters are obtained.\nUses standard KMeans for cluster splitting.","category":"page"},{"location":"api/","page":"API","title":"API","text":"References:","category":"page"},{"location":"api/","page":"API","title":"API","text":"Bisecting KMeans: An Improved Version of KMeans","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Cluster]\nPages = [\"BKMeans.jl\"]","category":"page"},{"location":"api/#Cluster.BKMeans-Tuple{}-api-3","page":"API","title":"Cluster.BKMeans","text":"BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode=\"kmeans\"))\n\nCreates a new BKMeans clustering model.\n\nKeyword Arguments:\n\nk::Int: The number of clusters (default: 3).\nkmeans::KMeans: An instance of the KMeans model used for cluster splitting (default: KMeans with 2 clusters).\n\nReturns: A BKMeans model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{BKMeans, Matrix{Float64}}-api-3","page":"API","title":"Cluster.fit!","text":"fit!(model::BKMeans, X)\n\nRuns the Bisecting KMeans algorithm for the given data and model.\n\nArguments:\n\nmodel::BKMeans: The BKMeans model to be trained.\nX: The input data matrix where each row is a data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{BKMeans, Any}-api-3","page":"API","title":"Cluster.predict","text":"predict(model::BKMeans, X) -> Array\n\nReturns the cluster labels for the given data points using the trained BKMeans model.\n\nArguments:\n\nmodel::BKMeans: The trained BKMeans model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns: An array of cluster labels for each data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributional-Clustering-Method","page":"API","title":"Distributional Clustering Method","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"References:","category":"page"},{"location":"api/","page":"API","title":"API","text":"Krishna, A., Mak, S. and Joseph, R., 2019. Distributional clustering: A distribution-preserving clustering method. arXiv preprint arXiv:1911.05940","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Cluster]\nPages = [\"DC.jl\"]","category":"page"},{"location":"api/#Cluster.DC-Tuple{}-api-4","page":"API","title":"Cluster.DC","text":"DC(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> DC\n\nCreates a new DC clustering model.\n\nKeyword Arguments\n\nk::Int: The number of clusters (default: 3).\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++, default: :kmeans).\nmax_try::Int: The maximum number of iterations for the algorithm (default: 100).\ntol::Float64: The tolerance for convergence (default: 1e-4).\n\nReturns\n\nA DC model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.compute_objective_function-Tuple{Matrix{Float64}, Matrix{Float64}}-api-4","page":"API","title":"Cluster.compute_objective_function","text":"compute_objective_function(data, centroids, k) -> Array\n\nComputes the distance from each data point to each centroid.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\ncentroids: The current centroids.\nk: the exponent of norm\n\nReturns\n\nA distance matrix D of size (number of data points, number of centroids), where D[i, j] is the distance from data point i to centroid j.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0\n\n\n]\ncentroids = [\n    1.0 1.0;\n    3.0 4.0\n]\nD = compute_objective_function(X, centroids)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{DC, Matrix{Float64}}-api-4","page":"API","title":"Cluster.fit!","text":"fit!(model::DC, X)\n\nRuns the Distributional Clustering algorithm for the given data and model.\n\nArguments\n\nmodel::DC: The DC model to be trained.\nX: The input data matrix where each row is a data point.\nk: Exponent of the norm\n\nExamples\n\nmodel = DC(3, \"dc\", 20, 1e-4, zeros(Float64, 0, 0), Int[])\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nfit!(model, X)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{DC, Matrix{Float64}}-api-4","page":"API","title":"Cluster.predict","text":"predict(model::DC, X) -> Array\n\nReturns the cluster labels for the given data points.\n\nArguments\n\nmodel::DC: The trained DC model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns\n\nAn array of cluster labels for each data point.\n\nExamples\n\ndata = [\n    # Cluster 1\n    1.0 1.0 1.5;\n    1.5 2.0 1.6;\n    1.3 1.8 1.4;\n    # Cluster 2\n    5.0 7.0 3.5;\n    5.5 7.5 3.5;\n    6.0 7.0 3.5;\n    # Cluster 3\n    8.0 1.0 6.5;\n    8.5 1.5 6.5;\n    8.3 1.2 7.5;\n]\ntest_data = [1.1 1.1 1.2]\nmodel = DC(k=3)\nfit!(model, data)\nlabels = predict(model, test_data)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.update_centroids-Tuple{Matrix{Float64}, Vector{Int64}, DC}-api-4","page":"API","title":"Cluster.update_centroids","text":"update_centroids(data, labelvector, model) -> Array\n\nCalculates new centroids based on the given data and label vector.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\nlabelvector: The current labels of the data points.\nmodel: The KMeans model.\n\nReturns\n\nAn array of new centroids.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0\n]\nlabels = [1, 1, 2, 2]\nmodel = KMeans(k=2)\nnew_centroids = update_centroids(X, labels, model)\n\n\n\n\n\n","category":"method"},{"location":"api/#Full-list-of-available-functions","page":"API","title":"Full list of available functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Cluster]","category":"page"},{"location":"api/#Cluster.BKMeans-Tuple{}","page":"API","title":"Cluster.BKMeans","text":"BKMeans(; k::Int=3, kmeans::KMeans=KMeans(k=2, mode=\"kmeans\"))\n\nCreates a new BKMeans clustering model.\n\nKeyword Arguments:\n\nk::Int: The number of clusters (default: 3).\nkmeans::KMeans: An instance of the KMeans model used for cluster splitting (default: KMeans with 2 clusters).\n\nReturns: A BKMeans model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.DC-Tuple{}","page":"API","title":"Cluster.DC","text":"DC(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> DC\n\nCreates a new DC clustering model.\n\nKeyword Arguments\n\nk::Int: The number of clusters (default: 3).\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++, default: :kmeans).\nmax_try::Int: The maximum number of iterations for the algorithm (default: 100).\ntol::Float64: The tolerance for convergence (default: 1e-4).\n\nReturns\n\nA DC model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.KMeans-Tuple{}","page":"API","title":"Cluster.KMeans","text":"KMeans(; k::Int=3, mode::Symbol=:kmeans, max_try::Int=100, tol::Float64=1e-4) -> KMeans\n\nCreates a new KMeans clustering model.\n\nKeyword Arguments\n\nk::Int: The number of clusters (default: 3).\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++, default: :kmeans).\nmax_try::Int: The maximum number of iterations for the algorithm (default: 100).\ntol::Float64: The tolerance for convergence (default: 1e-4).\n\nReturns\n\nA KMeans model with the specified parameters.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.assign_center-Tuple{Matrix{Float64}}","page":"API","title":"Cluster.assign_center","text":"assign_center(D) -> Array\n\nReturns the index of the nearest centroid for each data point.\n\nArguments\n\nD: The distance matrix.\n\nReturns\n\nAn array of indices indicating the nearest centroid for each data point.\n\nExamples\n\nD = [\n    0.0 2.0;\n    1.0 1.0;\n    2.0 0.0\n]\nlabels = assign_center(D)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.compute_distance-Tuple{Matrix{Float64}, Matrix{Float64}}","page":"API","title":"Cluster.compute_distance","text":"compute_distance(data, centroids) -> Array\n\nComputes the distance from each data point to each centroid.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\ncentroids: The current centroids.\n\nReturns\n\nA distance matrix D of size (number of data points, number of centroids), where D[i, j] is the distance from data point i to centroid j.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0\n]\ncentroids = [\n    1.0 1.0;\n    3.0 4.0\n]\nD = compute_distance(X, centroids)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.compute_objective_function-Tuple{Matrix{Float64}, Matrix{Float64}}","page":"API","title":"Cluster.compute_objective_function","text":"compute_objective_function(data, centroids, k) -> Array\n\nComputes the distance from each data point to each centroid.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\ncentroids: The current centroids.\nk: the exponent of norm\n\nReturns\n\nA distance matrix D of size (number of data points, number of centroids), where D[i, j] is the distance from data point i to centroid j.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0\n\n\n]\ncentroids = [\n    1.0 1.0;\n    3.0 4.0\n]\nD = compute_objective_function(X, centroids)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{BKMeans, Matrix{Float64}}","page":"API","title":"Cluster.fit!","text":"fit!(model::BKMeans, X)\n\nRuns the Bisecting KMeans algorithm for the given data and model.\n\nArguments:\n\nmodel::BKMeans: The BKMeans model to be trained.\nX: The input data matrix where each row is a data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{DC, Matrix{Float64}}","page":"API","title":"Cluster.fit!","text":"fit!(model::DC, X)\n\nRuns the Distributional Clustering algorithm for the given data and model.\n\nArguments\n\nmodel::DC: The DC model to be trained.\nX: The input data matrix where each row is a data point.\nk: Exponent of the norm\n\nExamples\n\nmodel = DC(3, \"dc\", 20, 1e-4, zeros(Float64, 0, 0), Int[])\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nfit!(model, X)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.fit!-Tuple{KMeans, Matrix{Float64}}","page":"API","title":"Cluster.fit!","text":"fit!(model::KMeans, X)\n\nRuns the KMeans algorithm for the given data and model.\n\nArguments\n\nmodel::KMeans: The KMeans model to be trained.\nX: The input data matrix where each row is a data point.\n\nExamples\n\nmodel = KMeans(k=3, mode=:kmeans)\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nfit!(model, X)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.init_centroids-Tuple{Matrix{Float64}, Int64, String}","page":"API","title":"Cluster.init_centroids","text":"Initialize centroids for the chosen algorithm.\n\nArguments\n\nX::Array{Float64,2}: The input data matrix where each row is a data point.\nK::Int: The number of clusters.\nmode::Symbol: The mode of initialization (:kmeans or :kmeans++).\n\nFields\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0;\n    8.0 9.0;\n    10.0 11.0\n]\nK = 2\nmode = :kmeans\ncentroids = init_centroids(X, K, mode)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{BKMeans, Any}","page":"API","title":"Cluster.predict","text":"predict(model::BKMeans, X) -> Array\n\nReturns the cluster labels for the given data points using the trained BKMeans model.\n\nArguments:\n\nmodel::BKMeans: The trained BKMeans model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns: An array of cluster labels for each data point.\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{DC, Matrix{Float64}}","page":"API","title":"Cluster.predict","text":"predict(model::DC, X) -> Array\n\nReturns the cluster labels for the given data points.\n\nArguments\n\nmodel::DC: The trained DC model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns\n\nAn array of cluster labels for each data point.\n\nExamples\n\ndata = [\n    # Cluster 1\n    1.0 1.0 1.5;\n    1.5 2.0 1.6;\n    1.3 1.8 1.4;\n    # Cluster 2\n    5.0 7.0 3.5;\n    5.5 7.5 3.5;\n    6.0 7.0 3.5;\n    # Cluster 3\n    8.0 1.0 6.5;\n    8.5 1.5 6.5;\n    8.3 1.2 7.5;\n]\ntest_data = [1.1 1.1 1.2]\nmodel = DC(k=3)\nfit!(model, data)\nlabels = predict(model, test_data)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.predict-Tuple{KMeans, Matrix{Float64}}","page":"API","title":"Cluster.predict","text":"predict(model, data) -> Array\n\nReturns the cluster labels for the given data points.\n\nArguments\n\nmodel::KMeans: The trained KMeans model.\nX::Array{Float64,2}: The input data matrix where each row is a data point.\n\nReturns\n\nAn array of cluster labels for each data point.\n\nExamples\n\ndata = [\n    # Cluster 1\n    1.0 1.0 1.5;\n    1.5 2.0 1.6;\n    1.3 1.8 1.4;\n    # Cluster 2\n    5.0 7.0 3.5;\n    5.5 7.5 3.5;\n    6.0 7.0 3.5;\n    # Cluster 3\n    8.0 1.0 6.5;\n    8.5 1.5 6.5;\n    8.3 1.2 7.5;\n]\ntest_data = [1.1 1.1 1.2]\nmodel = KMeans(k=3)\nfit!(\n\nmodel, data)\nlabels = predict(model, test_data)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.update_centroids-Tuple{Matrix{Float64}, Vector{Int64}, DC}","page":"API","title":"Cluster.update_centroids","text":"update_centroids(data, labelvector, model) -> Array\n\nCalculates new centroids based on the given data and label vector.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\nlabelvector: The current labels of the data points.\nmodel: The KMeans model.\n\nReturns\n\nAn array of new centroids.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0\n]\nlabels = [1, 1, 2, 2]\nmodel = KMeans(k=2)\nnew_centroids = update_centroids(X, labels, model)\n\n\n\n\n\n","category":"method"},{"location":"api/#Cluster.update_centroids-Tuple{Matrix{Float64}, Vector{Int64}, KMeans}","page":"API","title":"Cluster.update_centroids","text":"update_centroids(data, labelvector, model) -> Array\n\nCalculates new centroids based on the given data and label vector.\n\nArguments\n\ndata: The input data matrix where each row is a data point.\nlabelvector: The current labels of the data points.\nmodel: The KMeans model.\n\nReturns\n\nAn array of new centroids.\n\nExamples\n\nX = [\n    1.0 1.0;\n    1.5 2.0;\n    3.0 4.0;\n    5.0 6.0\n]\nlabels = [1, 1, 2, 2]\nmodel = KMeans(k=2)\nnew_centroids = update_centroids(X, labels, model)\n\n\n\n\n\n","category":"method"},{"location":"examples/","page":"Examples","title":"Examples","text":"CurrentModule = Cluster","category":"page"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#Basic-Example","page":"Examples","title":"Basic Example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Below is a basic usage example of the cluster.jl package. This example demonstrates how to set up and use the package with a dataset that has distinct clusters.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"# Load the Cluster.jl package\nusing Cluster\nusing Random\n\n# Set the random seed for reproducibility\nRandom.seed!(1234)\n\n# Create a simple 30x2 dataset with distinct clusters\ndata = vcat(\n    [randn(10, 2) .+ [1 1]; # Cluster 1 around [1, 1]\n        randn(10, 2) .+ [5 5]; # Cluster 2 around [5, 5]\n        randn(10, 2) .+ [9 1]] # Cluster 3 around [9, 1]\n)\n\n# Print the first 3 data points\nprintln(\"Data: \", data[1:3, :])\n\n# Print the shape of the data\nprintln(\"Shape: \", size(data))\n\n# Initialize the clustering algorithm\nmodel = Cluster.KMeans(k=3, mode=\"kmeans\") # also supports \"kmeanspp\"\n\n# Fit the model to the data\nCluster.fit!(model, data)\n\n# Print the centroids of the clusters\nprintln(\"Centroids: \", model.centroids)\n\n# Test Data\ntest_data = [\n    1.0 1.0; # Cluster 1\n    1.5 2.0; # Cluster 1\n    0.5 1.5; # Cluster 1\n    5.0 5.0; # Cluster 2\n    8.0 9.0; # Cluster 2\n    4.5 5.5; # Cluster 2\n    9.0 1.0  # Cluster 3\n    9.5 1.5  # Cluster 3\n    8.5 1.0  # Cluster 3\n]\n\nprintln(\"Test Data: \", test_data)\nprintln(\"Note: The labels might differ from the original cluster labels due to random initialization.\")\n\n\n# Predict the cluster for each data point\nclusters = Cluster.predict(model, test_data)\n\n# Print the resulting clusters\nprintln(\"Cluster assignments: \", clusters)\n\n# Plot the clusters\nusing Plots\nscatter(data[:, 1], data[:, 2], color=model.labels, legend=false)\nscatter!(model.centroids[:, 1], model.centroids[:, 2], color=:red, shape=:star, markersize=10)\nsavefig(\"simple-cluster.svg\"); nothing # hide","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"This is the resulting plot of the clusters:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: )","category":"page"},{"location":"examples/#Interactive-Example-Notebook","page":"Examples","title":"Interactive Example Notebook","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"This mike take some time to load, but you can run the example notebook in your browser by clicking the Binder link below. Here you can play around with generating different datasets and clustering them using the Cluster.jl package.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: Binder)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Cluster","category":"page"},{"location":"#Cluster.jl","page":"Home","title":"Cluster.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Cluster.jl. This is a package for clustering algorithms in Julia. It was created by a group of students as part of a project for the course \"Julia for Machine Learning\" at TU Berlin.","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Do not use this package\nPlease instead use the Clustering.jl package for a more complete and maintained package.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package can be installed by running the following command in the Julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/viktorlorentz/Cluster.jl\")","category":"page"}]
}
