module Plotting

using Plots

function visualize_clusters(data, true_labels, pred_labels, filename, directory_name)

    """
    Cretes 2D visualization of our cluster with true cluster (correct cluster) and predicted cluster (our cluster)
    and saves it in folder

    Args:
        data (Array): Datapoints
        true_labels (Array): Correct points 
        pred_labels (Array): Predicted points
        filename (String): Name for saving our plot. It should be name of our dataset
        directory_name (String): Nmae of directory in which our plots will be saved

    Returns:
        Saves two plots in the with true cluster and predicted cluster
    """
    
    unique_true_labels = unique(true_labels)
    unique_pred_labels = unique(pred_labels)
    
    colors = palette(:tab10, max(length(unique_true_labels), length(unique_pred_labels)))

    p = plot(layout=(1, 2))

    # We are crating two plots for both clusters
    scatter!(p[1], data[:, 1], data[:, 2], group=true_labels, title = "True Cluster",
    markersize=5, markerstrokewidth=0, legened=:topleft, palette=colors)

    scatter!(p[2], data[:, 1], data[:, 2], group=pred_labels, title="Predicted Cluster",
    markersize=5, markerstrokewidth=0, legend=:topleft, palette=colors)

    # Saving the plot inside the specific directory
    isdir(directory_name) || mkdir(directory_name)
    savepath = joinpath(directory_name, filename)
    savefig(p, savepath)

end
end