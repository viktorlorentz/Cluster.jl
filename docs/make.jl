using Cluster
using Documenter

DocMeta.setdocmeta!(Cluster, :DocTestSetup, :(using Cluster); recursive=true)

makedocs(;
    modules=[Cluster],
    authors="Mohamad Bassel Fatloun<fatloun@campus.tu-berlin.de>, Viktor Lorentz<lorentz@campus.tu-berlin.de>, Florentin Marquardt<marquardt@campus.tu-berlin.de>, Omadbek Meliev<meliev@campus.tu-berlin.de>",
    sitename="Cluster.jl",
    format=Documenter.HTML(;
        canonical="https://viktorlorentz.github.io/Cluster.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/viktorlorentz/Cluster.jl",
    devbranch="main",
)
