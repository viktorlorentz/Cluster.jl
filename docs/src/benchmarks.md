```@meta
CurrentModule = Cluster
```
# Benchmarks

For benchmark our project we used the dataset provided by [Marek Gagolewski](https://clustering-benchmarks.gagolewski.com/index.html). Specificallyh we are using [wut](https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html#sec-battery-wut) dataset. Below you can see the table with our metrics of rand index and accuracy of our algorithms: kmeans and kmeans++ which was benchmarked on wut dataset. Also, we are providing visualization in order to give insights what data we used.

## Comparison table
| Test Case | Kmeans Rand Index | Kmeans Accuracy | Kmeans++ Rand Index | Kmeans++ Accuracy | Bkmeans Rand Index | Bkmeans Accuracy |
|-----------------|-------------------|-----------------|--------------------|-------------------|-------------------|-------------------|
| circles | 1.00 | 1.00 | 1.00 | 1.00 | 1.0 | 1.0 |
| cross | 0.5905 | 0.5967 | 0.5905 | 0.5967 | 0.7362 | 0.7167 |
| graph | 0.8835 | 0.2373 | 0.8852 | 0.6133 | 0.8687 | 0.3147 |
| isolation | 0.5564 | 0.3474 | 0.5557 | 0.3444 | 0.5543 | 0.3489 |
| labirynth | 0.7526 | 0.4511 | 0.7526 | 0.4511 | 0.7839 | 0.2350 |
| mk1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.0 | 1.0 |
| mk2 | 0.5010 | 0.5467 | 0.5010 | 0.5467 | 0.5010 | 0.5467 |
| mk3 | 0.9740 | 0.9778 | 0.9740 | 0.9778 | 0.9740 | 0.9778 |
| mk4 | 0.5682 | 0.2311 | 0.5766 | 0.2400 | 0.6224 | 0.5111 |
| olympic | 0.7122 | 0.2907 | 0.7306 | 0.2707 | 0.7110 | 0.2840 |
| smile | 0.8263 | 0.4267 | 0.8379 | 0.5267 | 0.8335 | 0.5733 |
| stripes | 0.4998 | 0.5160 | 0.4998 | 0.5160 | 0.4998 | 0.5160 |
| trajectories | 1.00 | 1.00 | 1.00 | 1.00 | 1.0 | 1.0 |
| trapped_lovers | 0.5973 | 0.5840 | 0.5955 | 0.5813 | 0.5833 | 0.4533 |
| twosplashes | 0.6186 | 0.7500 | 0.6186 | 0.7500 | 0.6186 | 0.7500 |
| windows | 0.5745 | 0.3333 | 0.5677 | 0.3535 | 0.5820 | 0.4855 |
| x1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.0 | 1.0 |
| x2 | 0.8301 | 0.8889 | 0.6078 | 0.6667 | 0.6078 | 0.6667 |
| x3 | 0.9259 | 0.9286 | 0.9339 | 0.6429 | 0.9339 | 0.6429 |
| z1 | 0.6256 | 0.4828 | 0.6256 | 0.4828 | 0.6650 | 0.5862 |
| z2 | 0.7803 | 0.3333 | 0.7803 | 0.3333 | 0.7772 | 0.2963 |
| z3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.0 | 1.0 |


## Visualization K-means
```@raw html
<div style="text-align: center;">
  <img src="https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/circles.png?raw=true" alt="Clustering Circles">
  <p>Figure 1: Clustering Circles with K-means</p>
</div>

<div style="text-align: center;">
  <img src="https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/mk1.png?raw=true" alt="Mk1">
  <p>Figure 2: Mk1 with K-means</p>
</div>

<div style="text-align: center;">
  <img src="https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/mk3.png?raw=true" alt="Mk1">
  <p>Figure 3: Mk3 with K-means</p>
</div>

<div style="text-align: center;">
  <img src="https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/z3.png?raw=true" alt="Mk1">
  <p>Figure 4: Z3 with K-means</p>
</div>

<div style="text-align: center;">
  <img src="https://github.com/viktorlorentz/Cluster.jl/blob/benchmarking_alg/test/kmeans-figures/trajectories.png?raw=true" alt="Mk1">
  <p>Figure 5: Trajectories with K-means</p>
</div>
```