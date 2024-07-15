@testset "Example Notebook Tests" begin
    include("../examples/notebook.jl")

end
@testset "Simple Example Tests" begin
    include("../examples/simple.jl")

    # check if simple-cluster.png is created
    @test isfile("simple-cluster.png")

    # delete the file
    rm("simple-cluster.png", force=true)
end