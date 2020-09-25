using Scalpels
using Documenter

makedocs(;
    modules=[Scalpels],
    authors="Eric Ford",
    repo="https://github.com/RvSpectML/Scalpels.jl/blob/{commit}{path}#L{line}",
    sitename="Scalpels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RvSpectML.github.io/Scalpels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RvSpectML/Scalpels.jl",
)
