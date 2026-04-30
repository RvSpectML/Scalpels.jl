# src/Simulation/data_generation.jl
# Functions for generating synthetic RV and CCF datasets.

"""
    gen_rv_dataset(n; model, timespan=5*days_in_year, σ_rv=1.0,
                   num_obs=1, seed=nothing) -> DataFrame

Generate a synthetic RV dataset with Gaussian noise.

Observation times are drawn uniformly at random over `[0, timespan]`.

# Arguments
- `n`: Number of observations.

# Keyword Arguments
- `model`: A callable `model(t)` returning the noiseless RV at time `t`.
- `timespan`: Total time baseline (days).
- `σ_rv`: RV measurement uncertainty (m/s).
- `num_obs`: Number of distinct instruments/observers (used to assign
  random `obsid` labels).
- `seed`: Random seed for reproducibility. `nothing` uses the current RNG state.

# Returns
A `DataFrame` with columns `:t`, `:rv`, `:σrv`, `:obsid`.
"""
function gen_rv_dataset(
        n::Integer;
        model,
        timespan::Real = 5 * days_in_year,
        σ_rv::Real = 1.0,
        num_obs::Integer = 1,
        seed::Union{Nothing, Integer} = nothing
    )
    if !isnothing(seed)
        Random.seed!(seed)
    end
    t     = sort(timespan.* rand(n))
    rv    = model.(t).+ σ_rv.* randn(n)
    obsid = rand(1:num_obs, n)
    DataFrame(:t => t, :rv => rv, :σrv => fill(σ_rv, n), :obsid => obsid)
end

"""
    gen_ccf(v; fwhm=7200.0, depth=1.0, v_offset=0.0, σ_ccf=0.0) -> Vector

Generate a synthetic Gaussian CCF profile on a velocity grid.

The CCF has the form:

    CCF(v) = 1 - depth · exp(-(v - v_offset)² / (2σ²))

where `σ = fwhm / (2√(2 ln 2))`.

# Arguments
- `v`: Velocity grid (m/s).

# Keyword Arguments
- `fwhm`: Full width at half maximum of the CCF (m/s).
- `depth`: Line depth (dimensionless, 0–1).
- `v_offset`: Velocity offset of the line centre (m/s).
- `σ_ccf`: Standard deviation of additive Gaussian noise on the CCF.
"""
function gen_ccf(
        v::AbstractVector;
        fwhm::Real = 7200.0,
        depth::Real = 1.0,
        v_offset::Real = 0.0,
        σ_ccf::Real = 0.0
    )
    σ   = fwhm / (2 * sqrt(2 * log(2)))
    ccf = @. 1 - depth * exp(-(v - v_offset)^2 / (2 * σ^2))
    if !iszero(σ_ccf)
        ccf.+= σ_ccf.* randn(size(ccf))
    end
    return ccf
end

"""
    calc_ccf_derivs(x, y, var; scale=1, length=1, eps=10,
                    continuum=1.0) -> NamedTuple

Estimate the first and second derivatives of a CCF using a Gaussian process
smoother (Matérn 5/2 kernel via `TemporalGPs`).

Returns a named tuple `(f, dfdx, d2fdx2)`:
- `f`: GP posterior mean evaluated at `x`.
- `dfdx`: First derivative estimated by finite differences of the GP mean.
- `d2fdx2`: Second derivative estimated by finite differences of the GP mean.

# Arguments
- `x`: Velocity grid (m/s).
- `y`: Observed CCF values.
- `var`: Observation variance at each point (scalar or vector).

# Keyword Arguments
- `scale`: GP output scale parameter.
- `length`: GP length scale parameter (m/s).
- `eps`: Step size for finite-difference derivative estimation (m/s).
- `continuum`: Mean function value (CCF continuum level).

!!! note
    This function uses `TemporalGPs.to_sde` for efficient GP inference on
    the regularly-spaced velocity grid.
"""
function calc_ccf_derivs(
        x, y, var;
        scale::Real = 1,
        length::Real = 1,
        eps::Real = 10,
        continuum::Real = 1.0
    )
    gppr = to_sde(
        GP(ConstMean(continuum),
           scale * with_lengthscale(Matern52Kernel(), length))
    )
    gp    = posterior(gppr(x, var), y)
    f     = mean(gp(x))
    dfdx  = (mean(gp(x.+ eps/2)).- mean(gp(x.- eps/2)))./ eps
    d2fdx2 = (mean(gp(x.+ eps/2)).- 2.0 .* mean(gp(x)).+ mean(gp(x.- eps/2)))./ eps^2
    return (; f, dfdx, d2fdx2)
end

"""
    gen_rv_ccf_dataset(n; model, timespan=5*days_in_year, σ_rv=0.0,
                       num_obs=1, seed=nothing, fwhm=7200.0,
                       depth=1.0, σ_ccf=0.0,
                       v_grid, resolution, ccf_noise) -> NamedTuple

Generate a synthetic dataset of RVs and CCFs with correlated CCF shape
perturbations mimicking stellar activity.

Each observation has a randomly perturbed CCF width and line centre, producing
correlated shape and RV variations. Template RVs are computed by projecting
each CCF onto the derivative of the mean CCF (the standard template-matching
estimator).

Returns a named tuple `(df_rvs, ccfs, ccf_shape_perturb_mag)`:
- `df_rvs`: DataFrame with columns `:t`, `:rv`, `:σrv`, `:obsid`,
  `:rv_template`.
- `ccfs`: CCF matrix, size `(num_vel_bins, n)`.
- `ccf_shape_perturb_mag`: Vector of per-observation shape perturbation magnitudes.  Random perturbation vector.

# Arguments
- `n`: Number of observations.

# Keyword Arguments
- `model`: RV model callable.
- `timespan`: Observation baseline (days).
- `σ_rv`: Must be zero (noise enters only through `σ_ccf`).
- `num_obs`: Number of instruments.
- `seed`: Random seed.
- `fwhm`: Nominal CCF FWHM (m/s).
- `depth`: CCF line depth.
- `σ_ccf`: Per-pixel CCF noise.
- `v_grid`: Velocity grid (m/s).
- `resolution`: Spectral resolution R = λ/Δλ, used to set GP length scale.
- `ccf_noise`: CCF noise level used for GP inference of derivatives.
"""
function gen_rv_ccf_dataset(
        n::Integer;
        model,
        timespan::Real = 5 * days_in_year,
        σ_rv::Real = 0.0,
        num_obs::Integer = 1,
        seed::Union{Nothing, Integer} = nothing,
        fwhm::Real = 7200.0,
        depth::Real = 1.0,
        σ_ccf::Real = 0.0,
        v_grid::AbstractVector,
        resolution::Real,
        ccf_noise::Real
    )
    @assert iszero(σ_rv)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    df_rvs = gen_rv_dataset(n; model, timespan, num_obs)
    ccf_shape_perturb_mag = randn(n)

    ccfs = stack(
        i -> gen_ccf(
            v_grid;
            fwhm    = fwhm + 500 * ccf_shape_perturb_mag[i],
            depth,
            v_offset = df_rvs.rv[i] + 10 * ccf_shape_perturb_mag[i],
            σ_ccf
        ),
        1:n
    )

    α = 2
    mean_ccf, mean_dfdx, _ = calc_ccf_derivs(
        v_grid,
        vec(mean(ccfs, dims=2)),
        fill(ccf_noise, size(ccfs, 1));
        scale  = sqrt(α) / ccf_noise,
        length = α * c_mps / resolution
    )

    ccf_invvar = 1.0 / ccf_noise^2
    rv_proj = map(
        i -> -dot((ccfs[:, i].- mean_ccf).* ccf_invvar, mean_dfdx) /
              sum(abs2.(mean_dfdx).* ccf_invvar),
        1:n
    )
    σrv_proj = sqrt(1.0 / sum(abs2.(mean_dfdx).* ccf_invvar))

    df_rvs.rv_template = rv_proj
    df_rvs.σrv.= σrv_proj

    return (; df_rvs, ccfs, ccf_shape_perturb_mag)
end


function make_simulated_rvs_and_ccfs_to_test_scalpels(; num_rvs::Integer = 40, num_vels::Integer=200, v_max::Real = 20000,
                σ_ccf::Real = 0, depth::Real = 0.5, σv::Real = 7000, periods = [1/2, 1/3, 1/4], amplitudes = fill(0.001,length(periods)) )
    @assert length(periods) == length(amplitudes)
    @assert 1 <= length(amplitudes) <= 5
    times = range(0.0,stop=1,length=num_rvs)
    coeffs = amplitudes'.*cos.(2π.*times./periods')
    rotated_coeffs = coeffs

    if length(amplitudes)>1
        rot_angle = π/6
        rot_matrix = make_rotation_matrix(rot_angle, ndims=length(amplitudes), x=1, y=2)
        for i in 2:length(amplitudes)-1
             rot_matrix *= make_rotation_matrix(rot_angle, ndims=length(amplitudes), x=i, y=i+1)
        end
        rotated_coeffs *= rot_matrix
    end
    #println(rotated_coeffs)
    v_grid = range(-v_max,stop=v_max,length=num_vels)
    hermite_functions = [H1,H2,H3,H4]
    basis = mapreduce(h->h.(v_grid./σv), hcat, hermite_functions[1:length(amplitudes)] )

    ccfs = one(depth) .- depth .* (H0.(v_grid./σv) .+ mapreduce(i->basis[:,i].*rotated_coeffs[:,i]', +, 1:size(basis,2) ) )
    ccf_shape_perturb_mag = depth .* vec(sum(abs.(rotated_coeffs),dims=1))

    if σ_ccf > zero(σ_ccf)
        ccfs .*= one(σ_ccf) .+ σ_ccf .* randn(size(ccfs))
    end
    rvs_dirty = map(t->est_rv_from_test_ccf(v_grid,ccfs[:,t]),1:num_rvs)
    σ_rv = zeroes(rvs_dirty)  # TODO:  Add code
    df_rvs = DataFrame(:t=>times, :rv=>rvs_dirty, :σrv=>σ_rv, :obsid=>ones(length(times)) )

    return (;df_rvs, ccfs, ccf_shape_perturb_mag)
end
