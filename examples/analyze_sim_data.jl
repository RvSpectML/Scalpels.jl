using Plots
include("generate_sim_data.jl")

#num_obs = repeat([30, 100, 1000], 3);  σ_ccf = vcat( fill(1e-7, 3), fill(1e-10, 3), fill(0, 3) )
num_obs = repeat([40, 80, 160, 320, 640, 1280], 1);  σ_ccf = vcat( fill(1e-9, 6) )
 results = Any[]
 label_list = String[]
 for i in 1:length(num_obs)
    (rvs, ccfs) = make_simulated_rvs_and_ccfs_to_test_scalpels(num_rvs=num_obs[i], σ_ccf=σ_ccf[i])
    push!(results, rms_clean_rvs_vs_num_basis_scalpels(rvs,ccfs, max_num_basis=8) )
    str = "N_obs=" * string(num_obs[i]) * " σ=" * string(σ_ccf[i])
    push!(label_list,  str )
 end

num_basis_vectors = collect(0:(length(first(results))-1))
 plt = plot()
 map(i->scatter!(num_basis_vectors,results[i], label=label_list[i], color=i), 1:length(results) )
 map(i->plot!(num_basis_vectors,results[i], label=:none, color=i), 1:length(results) )
 xlabel!("Number of basis vectors used")
 ylabel!("RMS RVs post Scalpels")
 display(plt)
 #yaxis!(:log10)
