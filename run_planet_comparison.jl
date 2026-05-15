#!/usr/bin/env julia
# Run notebooks/planet_model_comparison.jl for each data/DS? subdirectory,
# passing the data directory as a command-line argument to the notebook.
#
# Usage:
#   julia run_planet_comparison.jl              # all DS? dirs under data/
#   julia run_planet_comparison.jl data/DS1     # single directory
#   julia run_planet_comparison.jl data/DS1 data/DS3   # explicit list

repo_dir = dirname(abspath(@__FILE__))
notebook = joinpath(repo_dir, "notebooks", "planet_model_comparison_joint.jl")

# Build the list of data directories to process.
ds_dirs = if isempty(ARGS)
    data_root = joinpath(repo_dir, "data")
    sort([
        joinpath(data_root, d)
        for d in readdir(data_root)
        if isdir(joinpath(data_root, d)) && occursin(r"^DS.$", d)
    ])
else
    map(abspath, ARGS)
end

if isempty(ds_dirs)
    println("No DS? directories found — nothing to do.")
    exit(1)
end

println("Datasets to process: $(join(basename.(ds_dirs), ", "))")

errors = String[]

for ds_dir in ds_dirs
    if !isdir(ds_dir)
        @warn "Skipping $(ds_dir): not a directory"
        continue
    end
    label = basename(ds_dir)
    println("\n", "="^60)
    println("=== $label  ($(ds_dir))")
    println("="^60)
    cmd = `julia --project=$repo_dir $notebook $ds_dir`
    result = run(ignorestatus(cmd))
    if !success(result)
        @warn "$label failed with exit code $(result.exitcode)"
        push!(errors, label)
    end
end

println("\n", "="^60)
if isempty(errors)
    println("All datasets completed successfully.")
else
    println("Completed with errors in: $(join(errors, ", "))")
    exit(1)
end
