using Test
using Scalpels
using Scalpels.Simulation
using Statistics, StatsBase, LinearAlgebra, Random

# ── Shared test dataset ────────────────────────────────────────────────────────
# Generated once and reused across test sets to keep the suite fast.

const v_grid     = range(-15000.0, stop=15000.0, step=300.0)
const resolution = 120000.0
const ccf_noise  = 0.0005
const ccf_depth  = 0.5
const num_obs_test = 40

const pl_model_test = RvSinusoidSimple(3.0, 5.0, 0.0, 0.0)

const _test_data = let
    gen_rv_ccf_dataset(
        num_obs_test;
        model      = pl_model_test,
        timespan   = 365.0,
        σ_ccf      = ccf_noise,
        seed       = 42,
        v_grid,
        resolution,
        ccf_noise
    )
end

const df_rvs_test = _test_data.df_rvs
const ccfs_test   = _test_data.ccfs

# ── RvSinusoid / RvSinusoidSimple ─────────────────────────────────────────────
@testset "RvSinusoid vs RvSinusoidSimple equivalence" begin
    m1 = RvSinusoidSimple(3.0, 100.0, 2.0, 1.0)
    m2 = RvSinusoid(; P=3.0, K=100.0, t0=2.0, C=1.0)

    data1 = gen_rv_dataset(num_obs_test; model=m1, timespan=30.0, σ_rv=0.0, seed=1234)
    data2 = gen_rv_dataset(num_obs_test; model=m2, timespan=30.0, σ_rv=0.0, seed=1234)

    max_abs_err = maximum(abs.(data1.rv.- data2.rv))
    @test max_abs_err < m1.K * 1e-12
end

# ── loocv: weighted vs unweighted mean (uniform σ) ───────────────────────────
@testset "loocv: weighted and unweighted means agree for uniform σ_rvs" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv
    rvs_centered = rvs.- mean(rvs)
    k = 5

    out_weighted   = loocv(rvs_centered, ccfs_test;
                           σ_rvs=σrv, max_scalpels_vectors=k, weighted_mean=true)
    out_unweighted = loocv(rvs_centered, ccfs_test;
                           σ_rvs=σrv, max_scalpels_vectors=k, weighted_mean=false)

    # When all σ_rvs are equal, both paths must give identical results.
    @test all(σrv.== first(σrv))  # confirm uniform σ in test data
    @test out_weighted.u_loocv ≈ out_unweighted.u_loocv   atol=1e-10
    @test out_weighted.α_loocv ≈ out_unweighted.α_loocv   atol=1e-10
end

# ── loocv: basic shape checks ─────────────────────────────────────────────────
@testset "loocv output shapes" begin
    rvs_centered = df_rvs_test.rv_template.- mean(df_rvs_test.rv_template)
    k = 10
    out = loocv(rvs_centered, ccfs_test; max_scalpels_vectors=k)

    @test size(out.u_loocv) == (num_obs_test, k)
    @test size(out.α_loocv) == (num_obs_test, k)
    @test eltype(out.u_loocv) == Float64
    @test eltype(out.α_loocv) == Float64
end

# ── reorder_uloocv ────────────────────────────────────────────────────────────
@testset "reorder_uloocv" begin
    rvs_centered = df_rvs_test.rv_template.- mean(df_rvs_test.rv_template)
    σrv = df_rvs_test.σrv
    k = 6
    out = loocv(rvs_centered, ccfs_test; max_scalpels_vectors=k)

    k_list, χ²_list, aic_list, rms_list = reorder_uloocv(
        out.u_loocv, out.α_loocv, rvs_centered, σrv;
        max_scalpels_vectors=k
    )

    @test length(k_list)   == k
    @test length(χ²_list)  == k + 1
    @test length(aic_list) == k + 1
    @test length(rms_list) == k + 1

    # χ² should be non-increasing (each step can only help or be neutral).
    # But uloocv @info χ²_list
    #@test all(diff(χ²_list).<= 1e-10)

    # All selected column indices should be unique and in range.
    @test length(unique(k_list)) == k
    @test all(1 .<= k_list.<= k)
end

# ── vscalpels_loocv ───────────────────────────────────────────────────────────
@testset "vscalpels_loocv" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv

    out = vscalpels_loocv(ccfs_test, rvs, σrv; max_scalpels_vectors=10)

    @test length(out.rv_clean) == num_obs_test
    @test size(out.u_loocv, 1) == num_obs_test
    @test size(out.α_loocv, 1) == num_obs_test
    @test size(out.u_loocv, 2) == size(out.α_loocv, 2)

    # Cleaned RVs should have lower or equal RMS than raw RVs.
    # TODO DECIDE WHAT THRESHOLD TO SET HERE
    @test std(out.rv_clean) <= std(rvs.- mean(rvs)) + 1e-10
end

# ── vscalpels_recover_loocv ───────────────────────────────────────────────────
@testset "vscalpels_recover_loocv" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv
    t   = df_rvs_test.t

    out = vscalpels_recover_loocv(t, rvs, σrv, ccfs_test, [3.0];
                                  max_scalpels_vectors=5)

    @test length(out.rv_centered) == num_obs_test
    @test length(out.rvclean)     == num_obs_test
    @test length(out.rvorbit)     == num_obs_test
    @test length(out.rvresid)     == num_obs_test
    @test length(out.amp)         == 1
    @test length(out.amperr)      == 1
    @test out.amp[1] > 0
    @test out.χ² >= 0

    # Scalar period convenience method should give identical results.
    out_scalar = vscalpels_recover_loocv(t, rvs, σrv, ccfs_test, 3.0;
                                         max_scalpels_vectors=5)
    @test out.amp    ≈ out_scalar.amp
    @test out.amperr ≈ out_scalar.amperr
    @test out.χ²     ≈ out_scalar.χ²

    # RV decomposition identity: rv_centered = rvshape + rvclean
    @test out.rv_centered ≈ out.rvshape.+ out.rvclean   atol=1e-10

    # Residual identity: rvresid = rvclean - rvorbit
    # TODO DECIDE WHAT THRESHOLD TO SET HERE
    # @test out.rvresid ≈ out.rvclean.- out.rvorbit   atol=1e-10

    # Amplitude should be in the right ballpark for a 5 m/s planet.
    # (Loose bound — LOOCV won't be perfect on small datasets.)
    @test 0.5 < out.amp[1] < 20.0
end

# ── mask_outliers ─────────────────────────────────────────────────────────────
@testset "mask_outliers" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv

    result = mask_outliers(ccfs_test, rvs, σrv; threshold=7, max_scalpels_vectors=10)

    @test length(result.obs_mask) == num_obs_test
    @test eltype(result.obs_mask) == Bool
    @test length(result.badfrac_vs_threshold) == 20

    # With a clean synthetic dataset and generous threshold, most obs should pass.
    @test sum(result.obs_mask) >= num_obs_test ÷ 2

    # badfrac should be non-increasing as threshold increases.
    @test all(diff(result.badfrac_vs_threshold).<= 1e-10)
end

# ── quality_control ───────────────────────────────────────────────────────────
@testset "quality_control" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv

    result = quality_control(ccfs_test, rvs, σrv;
                             max_scalpels_vectors=10, threshold=7)

    @test length(result.obs_mask) == num_obs_test
    @test result.kopt >= 0
    @test length(result.madratio) == 10
    @test all(result.madratio.>= 0)
end

# ── make_period_list ──────────────────────────────────────────────────────────
@testset "make_period_list" begin
    t = sort(rand(50).* 365.0)

    periods = make_period_list(t, 2.0, 100.0)

    @test issorted(periods)
    @test first(periods) >= 2.0 - 1e-10
    @test last(periods)  <= 100.0 + 1e-10
    @test length(periods) > 1

    # Oversampling should give more periods.
    periods_os = make_period_list(t, 2.0, 100.0; oversample_factor=2.0)
    @test length(periods_os) > length(periods)

    # Argument checks.
    @test_throws AssertionError make_period_list(t, 100.0, 2.0)
    @test_throws AssertionError make_period_list([1.0], 2.0, 100.0)
    @test_throws AssertionError make_period_list(reverse(t), 2.0, 100.0)
end

# ── svd_reconstruction ────────────────────────────────────────────────────────
@testset "svd_reconstruction" begin
    A = randn(50, 20)

    # Full reconstruction should recover A exactly.
    @test svd_reconstruction(A) ≈ A   atol=1e-10

    # Rank-1 reconstruction should have rank 1.
    A1 = svd_reconstruction(A; n=1)
    @test size(A1) == size(A)
    F  = svd(A1)
    @test F.S[2] < 1e-10 * F.S[1]

    # Reconstruction error should decrease monotonically with n.
    errs = [norm(svd_reconstruction(A; n=k).- A) for k in 1:min(10, size(A,2))]
    @test all(diff(errs).<= 1e-10)
end

# ── Original Scalpels functions (weighted_mean parameter) ─────────────────────
@testset "calc_basis_scores_scalpels weighted_mean" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv
    rvs_centered = rvs.- mean(rvs, weights(1.0./ σrv.^2))

    # With uniform σ, weighted and unweighted should agree.
    @test all(σrv.== first(σrv))
    s_w  = calc_basis_scores_scalpels(rvs_centered, ccfs_test;
                                      σ_rvs=σrv, num_basis=3,
                                      assume_centered=true, weighted_mean=true)
    s_uw = calc_basis_scores_scalpels(rvs_centered, ccfs_test;
                                      σ_rvs=σrv, num_basis=3,
                                      assume_centered=true, weighted_mean=false)
    @test s_w.scores ≈ s_uw.scores   atol=1e-10
    @test s_w.basis  ≈ s_uw.basis    atol=1e-10
end

@testset "clean_rvs_scalpels reduces RMS" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv

    rvs_clean = clean_rvs_scalpels(rvs, ccfs_test; σ_rvs=σrv, num_basis=1)
    @test std(rvs_clean) <= std(rvs) + 1e-10
end

@testset "rms_clean_rvs_vs_num_basis_scalpels" begin
    rvs = df_rvs_test.rv_template
    σrv = df_rvs_test.σrv
    k   = 5

    rms_list = rms_clean_rvs_vs_num_basis_scalpels(rvs, ccfs_test;
                                                    σ_rvs=σrv, max_num_basis=k)
    @test length(rms_list) == k + 1
    @test rms_list[1] ≈ std(rvs)   atol=1e-10
    @test all(rms_list.>= 0)
end

#= Old tests

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

=#