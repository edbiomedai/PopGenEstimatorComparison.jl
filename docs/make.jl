using PopGenEstimatorComparison
using Documenter

DocMeta.setdocmeta!(PopGenEstimatorComparison, :DocTestSetup, :(using PopGenEstimatorComparison); recursive=true)

makedocs(;
    modules=[PopGenEstimatorComparison],
    authors="Olivier Labayle <olabayle@gmail.com> and contributors",
    repo="https://github.com/olivierlabayle/PopGenEstimatorComparison.jl/blob/{commit}{path}#{line}",
    sitename="PopGenEstimatorComparison.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://olivierlabayle.github.io/PopGenEstimatorComparison.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/olivierlabayle/PopGenEstimatorComparison.jl",
    devbranch="main",
)
