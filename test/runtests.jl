using Scalpels
using Test

@testset "Scalpels.jl" begin

    @testset "Load code to generate simulated data" begin
        @test_nowarn include("../examples/generate_sim_data.jl")
    end
    include("../examples/generate_sim_data.jl")

    @testset "Check Code runs" begin
        @test_nowarn make_simulated_rvs_and_ccfs_to_test_scalpels()
        (rvs, ccfs) = make_simulated_rvs_and_ccfs_to_test_scalpels()
        @test_nowarn clean_rvs_scalpels(rvs,ccfs)
        @test_nowarn rms_clean_rvs_vs_num_basis_scalpels(rvs,ccfs)
    end

    @testset "Test accuracy" begin
        @test_nowarn make_simulated_rvs_and_ccfs_to_test_scalpels()
        (rvs, ccfs) = make_simulated_rvs_and_ccfs_to_test_scalpels()
        rms_vs_num_basis =  rms_clean_rvs_vs_num_basis_scalpels(rvs,ccfs)
        @test all(rms_vs_num_basis[5:end] .< 3.0)
    end

end
