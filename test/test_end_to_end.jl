# test/test_end_to_end.jl
#
# End-to-end tests for vSCALPELS with injected planet signals and
# Gauss-Hermite CCF shape perturbations mimicking stellar activity.
#
# These tests are NOT included in test/runtests.jl because they are
# computationally expensive (n_obs ≥ 80, full LOOCV). Run them manually:
#
#   julia --project=. test/test_end_to_end.jl
#
# or selectively in CI by setting the environment variable RUN_E2E_TESTS=true.

using Test
using Scalpels
using Scalpels.Simulation
using Statistics, StatsBase, LinearAlgebra, Random

# ── Gauss-Hermite basis functions ─────────────────────────────────────────────

"""
    physicists_hermite(n, x) -> Real

Evaluate the physicists' Hermite polynomial Hₙ(x) using the three-term
recurrence relation:
    H₀(x) = 1
    H₁(x) = 2x
    Hₙ(x) = 2x·Hₙ₋₁(x) - 2(n-1)·Hₙ₋₂(x)
"""
function physicists_hermite(n::Integer, x::Real)
    n == 0 && return one(x)
    n == 1 && return 2x
    h_prev2 = one(x)
    h_prev1 = 2x
    for k in 2:n
        h_curr  = 2x * h_prev1 - 2(k-1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    end
    return h_prev1
end

"""
    gauss_hermite_basis(v, n, σ) -> Vector

Evaluate the Gauss-Hermite basis function of order `n` on velocity grid `v`,
using the CCF's own width parameter `σ`:

    GHₙ(v) = Hₙ(v/σ) · exp(-v²/(2σ²))

where Hₙ is the physicists' Hermite polynomial.
"""
function gauss_hermite_basis(v::AbstractVector, n::Integer, σ::Real)
    x = v./ σ
    @. physicists_hermite(n, x) * exp(-x^2 / 2)
end

# ── Activity-modulated CCF perturbation ───────────────────────────────────────

"""
    make_gh_perturbations(v_grid, t, gh_orders, σ_ccf_shape;
                          P_act=28.0, perturbation_scale=1e-4,
                          seed=nothing)
        -> (perturbations, activity_phase, gh_amplitudes)

Generate additive CCF perturbations based on Gauss-Hermite functions,
modulated by a sinusoidal activity cycle.

Each GH order shares the same activity phase but has an independently drawn
amplitude. The perturbation for observation i is:

    δCCF(v, tᵢ) = perturbation_scale · sin(2π·tᵢ/P_act + φ)
                   · Σₙ aₙ · GHₙ(v)

where φ is a single random phase and aₙ ~ N(0,1) independently for each
order n.

The activity modulation also induces a spurious RV signal via the CCF
centroid shift, which is what vSCALPELS is designed to remove.

# Returns
- `perturbations`: Matrix of shape `(num_vel_bins, num_obs)` — additive
  CCF perturbations for each observation.
- `activity_phase`: Vector of length `num_obs` — the sinusoidal activity
  modulation value `sin(2π·t/P_act + φ)` at each observation time.
- `gh_amplitudes`: Vector of GH coefficients aₙ (one per order).
"""
function make_gh_perturbations(
        v_grid::AbstractVector,
        t::AbstractVector,
        gh_orders::AbstractVector{<:Integer},
        σ_ccf_shape::Real;
        P_act::Real = 28.0,
        perturbation_scale::Real = 1e-4,
        seed::Union{Nothing,Integer} = nothing
    )
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Single shared activity phase, independent amplitudes per order.
    φ              = 2π * rand(rng)
    gh_amplitudes  = randn(rng, length(gh_orders))
    activity_phase = sin.(2π.* t./ P_act.+ φ)

    # Build the combined GH shape template (num_vel_bins,).
    gh_shape = zeros(length(v_grid))
    for (i, n) in enumerate(gh_orders)
        gh_shape.+= gh_amplitudes[i].* gauss_hermite_basis(v_grid, n, σ_ccf_shape)
    end

    # Outer product: scale by activity modulation at each observation.
    perturbations = perturbation_scale.* gh_shape * activity_phase'

    return perturbations, activity_phase, gh_amplitudes
end

# ── Shared simulation parameters ──────────────────────────────────────────────

const E2E_SEED        = 20240101
const E2E_NUM_OBS     = 80
const E2E_TIMESPAN    = 365.0        # days
const E2E_P_PLANET    = 3.0          # days
const E2E_K_PLANET    = 5.0          # m/s
const E2E_P_ACT       = 28.0         # days  (activity cycle)
const E2E_RESOLUTION  = 120_000.0
const E2E_CCF_FWHM    = 7_200.0      # m/s
const E2E_CCF_DEPTH   = 0.5
const E2E_CCF_NOISE   = 0.0005
const E2E_PERTURB_SCALE = 1e-4       # 0.01% of continuum
const E2E_V_GRID      = range(-15_000.0, stop=15_000.0, step=300.0)

# σ of the Gaussian envelope — derived from the CCF FWHM.
const E2E_CCF_σ       = E2E_CCF_FWHM / (2 * sqrt(2 * log(2)))

# ── Core simulation helper ────────────────────────────────────────────────────

"""
    run_e2e_simulation(gh_orders; seed=E2E_SEED) -> NamedTuple

Generate a synthetic dataset with:
- A sinusoidal planet signal (P=$(E2E_P_PLANET) d, K=$(E2E_K_PLANET) m/s)
- Additive Gauss-Hermite CCF shape perturbations modulated by a
  $(E2E_P_ACT)-day activity cycle
- Template RVs computed by CCF derivative projection (biased by activity)

Returns a named tuple with all quantities needed for the tests.
"""
function run_e2e_simulation(
        gh_orders::AbstractVector{<:Integer};
        seed::Integer = E2E_SEED
    )
    rng = MersenneTwister(seed)

    pl_model = RvSinusoidSimple(
        Float64(E2E_P_PLANET),
        Float64(E2E_K_PLANET),
        0.0, 0.0
    )

    # ── Generate base CCF dataset (no GH perturbations yet) ──────────────────
    # Use a sub-seed so the base dataset is independent of gh_orders choice.
    base_seed = rand(rng, UInt32)
    base_out  = gen_rv_ccf_dataset(
        E2E_NUM_OBS;
        model      = pl_model,
        timespan   = E2E_TIMESPAN,
        σ_ccf      = E2E_CCF_NOISE,
        seed       = Int(base_seed),
        v_grid     = E2E_V_GRID,
        resolution = E2E_RESOLUTION,
        ccf_noise  = E2E_CCF_NOISE
    )
    df_rvs = base_out.df_rvs
    ccfs   = copy(base_out.ccfs)   # copy so we can add perturbations

    # ── Add Gauss-Hermite perturbations ──────────────────────────────────────
    perturb_seed = rand(rng, UInt32)
    perturbations, activity_phase, gh_amplitudes = make_gh_perturbations(
        collect(E2E_V_GRID),
        df_rvs.t,
        gh_orders,
        E2E_CCF_σ;
        P_act             = E2E_P_ACT,
        perturbation_scale = E2E_PERTURB_SCALE,
        seed              = Int(perturb_seed)
    )
    ccfs.+= perturbations

    # ── Recompute template RVs on the perturbed CCFs ─────────────────────────
    # This is the biased estimator that vSCALPELS should correct.
    α = 2
    mean_ccf, mean_dfdx, _ = calc_ccf_derivs(
        collect(E2E_V_GRID),
        vec(mean(ccfs, dims=2)),
        fill(E2E_CCF_NOISE, size(ccfs, 1));
        scale  = sqrt(α) / E2E_CCF_NOISE,
        length = α * 299_792_458.0 / E2E_RESOLUTION
    )
    ccf_invvar = 1.0 / E2E_CCF_NOISE^2
    rv_template = map(
        i -> -dot((ccfs[:, i].- mean_ccf).* ccf_invvar, mean_dfdx) /
              sum(abs2.(mean_dfdx).* ccf_invvar),
        1:E2E_NUM_OBS
    )
    σrv = sqrt(1.0 / sum(abs2.(mean_dfdx).* ccf_invvar))
    σrv_vec = fill(σrv, E2E_NUM_OBS)

    # True noiseless planet RVs (what we want to recover).
    rv_true = pl_model.(df_rvs.t)

    return (;
        df_rvs,
        ccfs,
        rv_template,
        σrv_vec,
        rv_true,
        pl_model,
        gh_orders,
        gh_amplitudes,
        activity_phase,
        perturbations
    )
end

# ── Single-orders test helper ─────────────────────────────────────────────────

"""
    run_e2e_testset(label, gh_orders)

Run the full end-to-end test for a given set of Gauss-Hermite orders.
Encapsulated as a function so the three @testset blocks share identical logic.
"""
function run_e2e_testset(label::String, gh_orders::AbstractVector{<:Integer})
    @testset "$label" begin
        sim = run_e2e_simulation(gh_orders)

        df_rvs      = sim.df_rvs
        ccfs        = sim.ccfs
        rv_template = sim.rv_template
        σrv_vec     = sim.σrv_vec
        rv_true     = sim.rv_true
        pl_model    = sim.pl_model

        # ── Pre-compute ACFs once for all calls ──────────────────────────────
        acfs = Scalpels._compute_acfs(ccfs)

        # ── Run vscalpels_recover_loocv at the true planet period ─────────────
        out = vscalpels_recover_loocv(
            df_rvs.t,
            rv_template,
            σrv_vec,
            ccfs,
            Float64(E2E_P_PLANET);
            max_scalpels_vectors = 10,
            acfs                 = acfs
        )

        # ── Test 1: amplitude recovery within 2σ of true K ───────────────────
        # The formal uncertainty amperr is the least-squares 1σ error on amp.
        K_true      = pl_model.K
        K_recovered = out.amp[1]
        K_err       = out.amperr[1]

        @test isfinite(K_recovered)
        @test isfinite(K_err)
        @test K_err > 0

        amp_residual = abs(K_recovered - K_true)
        @info "|K_recovered - K_true| = $(round(amp_residual, digits=3)) m/s " *
            "≥ 2σ = $(round(2*K_err, digits=3)) m/s " *
            "(K_true=$(K_true), K_recovered=$(round(K_recovered,digits=3)))"
        @test amp_residual < 2 * K_err  # "Amplitude recovery failed: " 
            

        # ── Test 2: bias reduction ────────────────────────────────────────────
        # vSCALPELS should reduce the RMS difference between the cleaned RVs
        # and the true planet signal compared to the raw template RVs.
        rv_template_centered = rv_template.- mean(rv_template)
        rms_before = std(rv_template_centered.- rv_true)
        rms_after  = std(out.rvclean.- rv_true)

        @info "RMS after=$(round(rms_after,digits=3)) m/s ≥ " *
            "RMS before=$(round(rms_before,digits=3)) m/s"

        @test rms_after < rms_before  # "Bias reduction failed: " 

        # ── Test 3: RV decomposition identity ─────────────────────────────────
        # rv_centered = rvshape + rvclean  (always true by construction,
        # but worth asserting to catch any future refactoring bugs).
        @test out.rv_centered ≈ out.rvshape.+ out.rvclean  atol=1e-10

        # ── Test 4: residuals are smaller than cleaned RVs ────────────────────
        # After fitting and removing the planet model, residuals should be
        # smaller than the cleaned RVs.
        @test std(out.rvresid) <= std(out.rvclean) + 1e-10

        # ── Diagnostic output (visible with --verbose) ────────────────────────
        @info "[$label] GH orders: $gh_orders" *
              "\n  K_true=$(K_true) m/s" *
              "\n  K_recovered=$(round(K_recovered,digits=3)) ± $(round(K_err,digits=3)) m/s" *
              "\n  |ΔK|/σ = $(round(amp_residual/K_err, digits=2))" *
              "\n  RMS before=$(round(rms_before,digits=3)) m/s" *
              "\n  RMS after=$(round(rms_after,digits=3)) m/s" *
              "\n  Vectors selected: $(size(out.u_loocv,2))"
    end
end

# ── End-to-end test sets ──────────────────────────────────────────────────────

@testset "vSCALPELS end-to-end: planet recovery with GH activity perturbations" begin

    @testset "Gauss-Hermite perturbation basis functions" begin
        # H₀(x) = 1
        @test physicists_hermite(0, 0.0) ≈ 1.0
        @test physicists_hermite(0, 1.7) ≈ 1.0

        # H₁(x) = 2x
        @test physicists_hermite(1, 0.0) ≈ 0.0
        @test physicists_hermite(1, 1.0) ≈ 2.0
        @test physicists_hermite(1, 2.0) ≈ 4.0

        # H₂(x) = 4x² - 2
        @test physicists_hermite(2, 0.0) ≈ -2.0
        @test physicists_hermite(2, 1.0) ≈  2.0

        # H₃(x) = 8x³ - 12x
        @test physicists_hermite(3, 0.0) ≈  0.0
        @test physicists_hermite(3, 1.0) ≈ -4.0

        # H₄(x) = 16x⁴ - 48x² + 12
        @test physicists_hermite(4, 0.0) ≈  12.0
        @test physicists_hermite(4, 1.0) ≈ -20.0

        # H₅(x) = 32x⁵ - 160x³ + 120x
        @test physicists_hermite(5, 0.0) ≈   0.0
        @test physicists_hermite(5, 1.0) ≈  -8.0

        # H₆(x) = 64x⁶ - 480x⁴ + 720x² - 120
        @test physicists_hermite(6, 0.0) ≈ -120.0
        @test physicists_hermite(6, 1.0) ≈  184.0

        # Recurrence sanity: Hₙ(x) = 2x·Hₙ₋₁(x) - 2(n-1)·Hₙ₋₂(x)
        for n in 3:8, x in [-2.0, -0.5, 0.0, 0.5, 2.0]
            @test physicists_hermite(n, x) ≈
                2x * physicists_hermite(n-1, x) -
                2(n-1) * physicists_hermite(n-2, x)  atol=1e-10
        end

        # GH basis: peak should be near v=0 for even orders.
        v_grid = collect(range(-15_000.0, stop=15_000.0, step=300.0))
        gh4    = gauss_hermite_basis(v_grid, 4, E2E_CCF_σ)
        @test argmax(abs.(gh4)) != 1       # not at the edge
        @test argmax(abs.(gh4)) != length(v_grid)

        # GH basis: odd orders should be antisymmetric about v=0.
        gh3 = gauss_hermite_basis(v_grid, 3, E2E_CCF_σ)
        @test gh3 ≈ -reverse(gh3)  atol=1e-10

        # GH basis: even orders should be symmetric about v=0.
        gh6 = gauss_hermite_basis(v_grid, 6, E2E_CCF_σ)
        @test gh6 ≈  reverse(gh6)  atol=1e-10
    end

    @testset "Perturbation amplitude is at the correct scale" begin
        sim = run_e2e_simulation([3, 4, 5, 6])

        # Maximum perturbation should be O(perturbation_scale) relative
        # to the CCF continuum (which is 1.0).
        max_perturb = maximum(abs.(sim.perturbations))
        @test max_perturb < 10 * E2E_PERTURB_SCALE   # not wildly large
        @test max_perturb >  0.01 * E2E_PERTURB_SCALE # not negligibly small

        # Perturbations should be correlated with the activity phase.
        # The column norms of the perturbation matrix should track
        # |activity_phase|.
        col_norms    = vec(sqrt.(sum(sim.perturbations.^2, dims=1)))
        abs_act      = abs.(sim.activity_phase)
        corr         = cor(col_norms, abs_act)
        @test corr > 0.9  # "Activity-perturbation correlation too low: $corr"
    end

    @testset "Template RVs are biased by activity before cleaning" begin
        # Before cleaning, the template RVs should show excess scatter
        # compared to the true planet signal — i.e., the activity
        # perturbations leak into the measured RVs.
        sim = run_e2e_simulation([3, 4, 5, 6])

        rv_template_centered = sim.rv_template.- mean(sim.rv_template)
        rms_template_vs_true = std(rv_template_centered.- sim.rv_true)
        rms_true             = std(sim.rv_true)

        # The template RV error should be larger than pure photon noise
        # (i.e., activity is adding scatter).
        @test rms_template_vs_true > first(sim.σrv_vec)
    end

    # ── Planet recovery: odd GH orders only (3, 5) ───────────────────────────
    run_e2e_testset("Odd GH orders [3,5]", [3, 5])

    # ── Planet recovery: even GH orders only (4, 6) ──────────────────────────
    run_e2e_testset("Even GH orders [4,6]", [4, 6])

    # ── Planet recovery: all GH orders (3–6) ─────────────────────────────────
    run_e2e_testset("All GH orders [3,4,5,6]", [3, 4, 5, 6])

end