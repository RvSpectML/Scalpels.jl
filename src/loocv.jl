# src/loocv.jl
# Leave-one-out cross-validation (LOOCV) based SCALPELS functions.

"""
    loocv(rvs_centered, ccfs; σ_rvs=1.0,
          max_scalpels_vectors, weighted_mean=true) -> NamedTuple

Compute leave-one-out cross-validated CCF shape vectors and their RV
projections.

For each observation `j`, fits the CCF autocorrelation basis using all
observations *except* `j`, then projects observation `j` onto that basis.
This avoids overfitting the shape model to the RVs.

Returns a named tuple `(α_loocv, u_loocv)` where:
- `u_loocv`: `(num_obs × max_scalpels_vectors)` matrix. Row `j` contains the
  projection of observation `j`'s ACF onto the LOO basis.
- `α_loocv`: `(num_obs × max_scalpels_vectors)` matrix. Row `j` contains the
  projection of the LOO RVs onto the LOO basis left singular vectors.

# Arguments
- `rvs_centered`: Mean-subtracted observed RVs, length `num_obs`.
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.

# Keyword Arguments
- `σ_rvs`: RV uncertainties (scalar or vector of length `num_obs`).
- `max_scalpels_vectors`: Number of LOOCV shape vectors to compute.
  Defaults to `calc_max_vectors(ccfs)`.
- `weighted_mean`: If `true` (default), subtract the inverse-variance weighted
  mean ACF before SVD. If `false`, subtract the unweighted mean.
"""
function loocv(
        rvs_centered::AbstractVector{T1},
        ccfs::AbstractMatrix{T2};
        σ_rvs = 1.0,
        max_scalpels_vectors::Integer = calc_max_vectors(ccfs),
        weighted_mean::Bool = true
    ) where {T1<:Real, T2<:Real}

    num_obs = length(rvs_centered)
    @assert size(ccfs, 2) == num_obs

    @assert max_scalpels_vectors <= calc_max_vectors(ccfs)

    obs_weights = 1.0./ (isa(σ_rvs, Real) ? fill(σ_rvs, num_obs).^2 : σ_rvs.^2)
    @assert is_centered(rvs_centered, obs_weights)

    acfs = autocor(ccfs, 0:size(ccfs, 1)-1)

    # Compute global mean ACF for centering LOO subsets consistently.
    mean_acfs = if weighted_mean && !all(σ_rvs.== first(σ_rvs))
        mean(acfs, weights(obs_weights), dims=2)
    else
        mean(acfs, dims=2)
    end

    k = max_scalpels_vectors
    u_loocv = zeros(num_obs, k)
    α_loocv = zeros(num_obs, k)

    obs_list = 1:num_obs
    for j in obs_list
        mask = obs_list.!= j
        acfloo = view(acfs, :, mask)
        rvloo  = view(rvs_centered, mask)

        svd_acfs = svd(acfloo')
        U  = svd_acfs.U
        S  = svd_acfs.S
        Vt = svd_acfs.Vt

        sv_mat  = Diagonal(view(S, 1:k)) * view(Vt, 1:k, :)
        vsv_mat = Diagonal(view(S, 1:k))

        # Project observation j's ACF onto the LOO basis.
        asv_vec = view(Vt, 1:k, :) * view(acfs, :, j)
        uj_loo  = vsv_mat \ asv_vec

        # Project LOO RVs onto the LOO left singular vectors.
        alpha_loo = view(U, :, 1:k)' * rvloo

        α_loocv[j, :].= alpha_loo
        u_loocv[j, :].= uj_loo
    end

    return (; α_loocv, u_loocv)
end

"""
    reorder_uloocv(u, α, rv_obs, σ_rv; jitter=0.0,
                   max_scalpels_vectors) -> Tuple

Greedily select and order LOOCV shape vectors by their ability to reduce the
χ² of the RV residuals.

At each step, the vector that most reduces χ² is selected from the remaining
candidates. Returns the permutation index, χ² history, AIC history, and RMS
history.

# Arguments
- `u`: `(num_obs × num_vecs)` LOOCV shape vector matrix from [`loocv`](@ref).
- `α`: `(num_obs × num_vecs)` LOOCV RV projection matrix from [`loocv`](@ref).
- `rv_obs`: Mean-subtracted observed RVs, length `num_obs`.
- `σ_rv`: RV uncertainties, length `num_obs`.

# Keyword Arguments
- `jitter`: Additional jitter (m/s) added in quadrature to `σ_rv`.
- `max_scalpels_vectors`: Maximum number of vectors to select. Defaults to
  `size(u, 2)`.

# Returns
A tuple `(k_list, χ²_list, aic_list, rms_list)`:
- `k_list`: Indices into columns of `u`/`α` in selection order.
- `χ²_list`: χ² after adding each successive vector (length
  `max_scalpels_vectors + 1`, first entry is χ² with no vectors removed).
- `aic_list`: AIC values corresponding to `χ²_list`.
  # TODO: Consider also returning BIC for comparison.
- `rms_list`: RMS of RV residuals corresponding to `χ²_list`.
"""
function reorder_uloocv(
        u::AbstractMatrix{T1},
        α::AbstractMatrix{T2},
        rv_obs::AbstractVector{T3},
        σ_rv::AbstractVector{T4};
        jitter::Real = 0.0,
        max_scalpels_vectors::Integer = size(u, 2)
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    num_cols = size(u, 2)
    max_scalpels_vectors = min(max_scalpels_vectors, num_cols)
    good_cols = trues(num_cols)
    invar = 1.0./ (σ_rv.^2 .+ jitter^2)
    current_rv_residuals = copy(rv_obs)
    χ²_orig = sum(current_rv_residuals.^2 .* invar)

    χ²_list  = zeros(max_scalpels_vectors + 1)
    rms_list = zeros(max_scalpels_vectors + 1)
    k_list   = zeros(Int, max_scalpels_vectors)
    χ²_list[1]  = χ²_orig
    rms_list[1] = std(current_rv_residuals)

    χ²_candidates       = fill(Inf, num_cols)
    candidate_col_indices = zeros(Int, num_cols)
    rvk_candidates      = Vector{Vector{Float64}}(undef, num_cols)

    for j in 1:max_scalpels_vectors
        active_col_indices = (1:num_cols)[good_cols]
        χ²_candidates.= Inf
        candidate_col_indices.= 0
        resize!(rvk_candidates, length(active_col_indices))

        for (i, k) in enumerate(active_col_indices)
            rvk_candidates[i]       = u[:, k].* α[:, k]
            candidate_col_indices[i] = k
            χ²_candidates[i] = sum(
                (current_rv_residuals.- rvk_candidates[i]).^2 .* invar
            )
        end

        best_idx = argmin(χ²_candidates)
        χ²_list[j+1]  = χ²_candidates[best_idx]
        k_list[j]     = candidate_col_indices[best_idx]
        current_rv_residuals.-= rvk_candidates[best_idx]
        rms_list[j+1] = std(current_rv_residuals)
        good_cols[k_list[j]] = false
    end

    # AIC penalty: each selected vector adds (num_cols + 1 + num_obs) parameters.
    # TODO: Consider also computing BIC = χ²_list.+ penalty.* log(num_obs).
    penalty = (num_cols + 1 + length(rv_obs))
    aic_list = χ²_list.+ (0:max_scalpels_vectors).* penalty

    return k_list, χ²_list, aic_list, rms_list
end

"""
    vscalpels_loocv(ccfs, rv, σ_rv; max_scalpels_vectors,
                    jitter=0.0, resort=true,
                    weighted_mean=true) -> NamedTuple

Clean RVs of stellar activity using LOOCV-based SCALPELS (vSCALPELS).

Computes LOOCV shape vectors, optionally reorders them by χ² reduction
(using AIC to select the optimal number), and subtracts the inferred
activity signal from the RVs.

Returns a named tuple `(rv_clean, u_loocv, α_loocv)`:
- `rv_clean`: Activity-cleaned RVs, length `num_obs`.
- `u_loocv`: Selected LOOCV shape vectors, `(num_obs × k)`.
- `α_loocv`: Selected LOOCV RV projections, `(num_obs × k)`.

# Arguments
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `rv`: Observed RVs, length `num_obs`.
- `σ_rv`: RV uncertainties, length `num_obs`.

# Keyword Arguments
- `max_scalpels_vectors`: Maximum number of shape vectors to consider.
- `jitter`: Additional jitter (m/s) added in quadrature to `σ_rv` when
  computing χ².
- `resort`: If `true` (default), reorder vectors by χ² reduction and use
  AIC to select the optimal number.
- `weighted_mean`: If `true` (default), use inverse-variance weighted mean
  ACF subtraction inside [`loocv`](@ref).
"""
function vscalpels_loocv(
        ccfs::AbstractMatrix,
        rv::AbstractVector,
        σ_rv::AbstractVector;
        max_scalpels_vectors::Integer = calc_max_vectors(ccfs),
        jitter::Real = 0.0,
        resort::Bool = true,
        weighted_mean::Bool = true
    )

    kmax = max_scalpels_vectors
    mean_rv = mean(rv)
    rv_centered = rv.- mean_rv

    α_loocv, u_loocv = loocv(
        rv_centered, ccfs;
        max_scalpels_vectors,
        weighted_mean
    )

    if resort
        idx_perm, _, aic_list, rms_list = reorder_uloocv(
            u_loocv, α_loocv, rv_centered, σ_rv;
            jitter, max_scalpels_vectors
        )
        #kmax     = argmin(aic_list)
        kmax     = argmin(rms_list)
        α_loocv  = α_loocv[:, idx_perm[1:kmax-1]]
        u_loocv  = u_loocv[:, idx_perm[1:kmax-1]]
    else
        α_loocv = view(α_loocv, :, 1:kmax)
        u_loocv = view(u_loocv, :, 1:kmax)
    end

    v_resp   = u_loocv.* α_loocv
    rv_shape = sum(v_resp, dims=2)
    rv_clean = vec(rv_centered.- rv_shape) .+ mean_rv

    return (; rv_clean, u_loocv, α_loocv)
end

"""
    vscalpels_recover_loocv(bjd, rvs, σ_rv, ccfs, periods;
                            max_scalpels_vectors, resort=true,
                            jitter=0.0, weighted_mean=true) -> NamedTuple

Simultaneously remove stellar activity and recover Keplerian signals at
specified trial periods using LOOCV-based SCALPELS.

For each trial period, fits sinusoidal planet models in the subspace
orthogonal to the LOOCV activity basis, returning amplitude estimates and
their uncertainties.

Returns a named tuple with fields:
- `rv_centered`: Mean-subtracted input RVs.
- `rvshape`: Inferred activity signal.
- `rvclean`: RVs with activity removed (= `rv_centered - rvshape`).
- `rvorbit`: Best-fit Keplerian signal.
- `rvresid`: Residuals after removing both activity and orbit.
- `periods`: Input trial periods.
- `amp`: Keplerian velocity semi-amplitude for each period (m/s).
- `amperr`: Uncertainty on `amp`.
- `u_loocv`: Selected LOOCV shape vectors.
- `α_loocv`: Selected LOOCV RV projections.
- `fftrn`: Design matrix of sine/cosine basis functions.
- `χ²`: χ² of the residuals.

# Arguments
- `bjd`: Barycentric Julian dates of observations, length `num_obs`.
- `rvs`: Observed RVs (m/s), length `num_obs`.
- `σ_rv`: RV uncertainties (m/s), length `num_obs`.
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `periods`: Trial orbital period or vector of periods (days).

# Keyword Arguments
- `max_scalpels_vectors`: Maximum number of LOOCV shape vectors to consider.
- `resort`: If `true` (default), use AIC to select the optimal number of
  shape vectors.
- `jitter`: Additional jitter (m/s) added in quadrature to `σ_rv`.
- `weighted_mean`: If `true` (default), use inverse-variance weighted mean
  ACF subtraction inside [`loocv`](@ref).
- `mean_bjd`: Reference epoch (days) used as the time zero-point when
  building the sinusoidal design matrix. Defaults to `mean(bjd)`.

# Extended help

The method projects both the RVs and the Keplerian basis functions into the
subspace orthogonal to the LOOCV activity basis (`pperp`), then solves a
weighted least-squares problem for the Keplerian amplitudes. This ensures
the planet signal is not absorbed by the activity model.
"""
function vscalpels_recover_loocv(
        bjd::AbstractVector{<:Real},
        rvs::AbstractVector{<:Real},
        σ_rv::AbstractVector{<:Real},
        ccfs::AbstractMatrix{<:Real},
        periods::AbstractVector{<:Real};
        max_scalpels_vectors::Integer = calc_max_vectors(ccfs),
        resort::Bool = true,
        jitter::Real = 0.0,
        weighted_mean::Bool = true,
        mean_bjd::Real = mean(bjd)
    )

    @assert max_scalpels_vectors >= 0
    kmax = max_scalpels_vectors
    nobs = length(bjd)
    mean_rv = mean(rvs)
    rv_centered = rvs .- mean_rv

    loocv_out = loocv(
        rv_centered, ccfs;
        max_scalpels_vectors,
        weighted_mean
    )
    u_loocv = loocv_out.u_loocv
    α_loocv = loocv_out.α_loocv

    if resort
        idx_perm, χ²_list, aic_list, rms_list = reorder_uloocv(
            u_loocv, α_loocv, rv_centered, σ_rv;
            jitter, max_scalpels_vectors
        )
        if max_scalpels_vectors > 0
            #kmax    = argmin(aic_list)
            kmax    = argmin(rms_list)
            α_loocv = α_loocv[:, idx_perm[1:kmax-1]]
            u_loocv = u_loocv[:, idx_perm[1:kmax-1]]
        end
    end

    # Build sinusoidal design matrix for all trial periods.
    nplanets = length(periods)
    fftrn = Matrix{Float64}(undef, nobs, 2 * nplanets + 1)
    for planet in 1:nplanets
        phi = 2π / periods[planet].* (bjd.- mean_bjd)
        fftrn[:, 2*planet-1] = cos.(phi)
        fftrn[:, 2*planet]   = sin.(phi)
    end
    fftrn[:, 2*nplanets+1] .= 1.0

    # Projection operator onto the complement of the activity subspace.
    pperp = I(nobs)
    if kmax > 0
        pperp = pperp - u_loocv * u_loocv'
    end

    fperp = pperp * fftrn
    vperp = pperp * rv_centered

    # Weighted least-squares for Keplerian amplitudes.
    # pinv is used instead of \ and inv to handle singular amat, which can
    # occur when a trial period is commensurate with the observation baseline
    # or when the activity projector absorbs a sinusoidal direction.
    invar     = Diagonal(σ_rv.^(-2))
    amat      = (fperp' * invar) * fperp
    bvec      = (fperp' * invar) * vperp
    amat_pinv = pinv(amat)
    theta     = amat_pinv * bvec
    theterr   = sqrt.(max.(0.0, diag(amat_pinv)))

    # Convert (cos, sin) amplitudes to velocity semi-amplitudes.
    Kx     = Vector{Float64}(undef, nplanets)
    Ky     = Vector{Float64}(undef, nplanets)
    dKx    = Vector{Float64}(undef, nplanets)
    dKy    =  Vector{Float64}(undef, nplanets)
    amp    = Vector{Float64}(undef, nplanets)
    amperr = Vector{Float64}(undef, nplanets)
    phase  = Vector{Float64}(undef, nplanets)
    phaseerr = Vector{Float64}(undef, nplanets)
    t0 = Vector{Float64}(undef, nplanets)
    for planet in 1:nplanets
        icol = 2 * planet - 1
        Kx[planet], dKx[planet] = theta[icol],   theterr[icol]
        Ky[planet], dKy[planet] = theta[icol+1], theterr[icol+1]
        K  = sqrt(Kx[planet]^2 + Ky[planet]^2)
        dK = sqrt((Kx[planet] * dKx[planet])^2 + (Ky[planet] * dKy[planet])^2) / K
        amp[planet]    = K
        amperr[planet] = dK
        phase[planet]  = rad2deg(atan(Ky[planet], Kx[planet]))
        phaseerr[planet] = rad2deg(sqrt(Ky[planet]^2*dKx[planet]^2+Kx[planet]^2*dKy[planet]^2)/(Kx[planet]^2+Ky[planet]^2))
        t0 = mean_bjd + periods[planet]/(2π) * phase[planet]
    end
    C = last(theta)
    Cerr = last(theterr)
 
    rvorbit  = fftrn * theta
    rvclean  = vperp 
    rvshape  = rv_centered.- rvclean
    #rvclean  .+= mean_rv
    rvresid  = vperp.- fperp * theta
    χ²       = sum(abs2.(rvresid./ σ_rv))

    return (;
        rv_obs=rvs, rvshape, rvclean, rvorbit, rvresid,
        periods, amp, amperr, Kx, Ky, dKx, dKy, C, Cerr, phase, phaseerr, t0, u_loocv, α_loocv, fftrn, χ²
    )
end

# Single-period convenience method.
"""
    vscalpels_recover_loocv(bjd, rvs, σ_rv, ccfs, period::Real; kwargs...)

Convenience method accepting a single trial `period` (days) as a scalar.
See the vector-period method for full documentation.
"""
function vscalpels_recover_loocv(
        bjd::AbstractVector{<:Real},
        rvs::AbstractVector{<:Real},
        σ_rv::AbstractVector{<:Real},
        ccfs::AbstractMatrix{<:Real},
        period::Real;
        kwargs...
    )
    vscalpels_recover_loocv(bjd, rvs, σ_rv, ccfs, [period]; kwargs...)
end

"""
    fit_planets_loocv(bjd, rvs, σ_rv, ccfs, periods;
                      k_search, max_num_basis=16,
                      resort=false, jitter=0.0,
                      weighted_mean=true) -> NamedTuple

Fit a fixed list of planetary periods jointly with stellar activity using
LOOCV-based SCALPELS. Performs the full three-step pipeline:

1. **Initial fit** — `vscalpels_recover_loocv` at `k_search` activity vectors.
2. **k optimisation** — `rms_clean_rvs_with_planets_vs_num_basis_scalpels`
   sweeps from 0 to `max_num_basis` vectors; `k_opt` is chosen by minimum AIC.
3. **Final fit** — `vscalpels_recover_loocv` at `k_opt` vectors.

This function can be called directly with user-supplied periods or is used
internally by [`search_planets_loocv`](@ref) after the period search step.

Returns a NamedTuple with fields:
- `periods`: input period list (days).
- `k_search`: number of activity vectors used in the initial fit.
- `k_opt`: optimal number of activity vectors selected by AIC.
- `init_fit`: full output of `vscalpels_recover_loocv` at `k_search`.
- `sweep`: full output of `rms_clean_rvs_with_planets_vs_num_basis_scalpels`.
- `final_fit`: full output of `vscalpels_recover_loocv` at `k_opt`.

# Arguments
- `bjd`: Observation times (days).
- `rvs`: Observed RVs (m/s).
- `σ_rv`: RV uncertainties (m/s).
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `periods`: Vector of orbital periods to fit simultaneously (days).

# Keyword Arguments
- `k_search`: Activity vectors for the initial fit; should be ≥ 1.
- `max_num_basis`: Upper bound on the k sweep (default 16).
- `resort`, `jitter`, `weighted_mean`: passed to `vscalpels_recover_loocv`.
- `mean_bjd`: Reference epoch (days) passed to `vscalpels_recover_loocv`.
  Defaults to `mean(bjd)`.
"""
function fit_planets_loocv(
        bjd::AbstractVector{<:Real},
        rvs::AbstractVector{<:Real},
        σ_rv::AbstractVector{<:Real},
        ccfs::AbstractMatrix{<:Real},
        periods::AbstractVector{<:Real};
        k_search::Integer,
        max_num_basis::Integer = 16,
        resort::Bool = false,
        fixed_k::Bool = false,
        jitter::Real = 0.0,
        weighted_mean::Bool = true,
        mean_bjd::Real = mean(bjd)
    )
    # Step 1: initial fit at k_search vectors.
    init_fit = vscalpels_recover_loocv(
        bjd, rvs, σ_rv, ccfs, periods;
        max_scalpels_vectors = k_search,
        resort, jitter, weighted_mean, mean_bjd
    )

    # Step 2: sweep k from 0 to max_num_basis and select by AIC.
    sweep = rms_clean_rvs_with_planets_vs_num_basis_scalpels(
        bjd, rvs, ccfs, periods;
        σ_rvs         = σ_rv,
        min_num_basis = 0,
        max_num_basis = max_num_basis,
        resort        = false,
        weighted_mean,
    )
    k_opt = fixed_k ? k_search : sweep.num_basis[argmin(sweep.rms)]

    # Step 3: final fit at k_opt.
    final_fit = k_opt == k_search ? init_fit : vscalpels_recover_loocv(
        bjd, rvs, σ_rv, ccfs, periods;
        max_scalpels_vectors = max(1, k_opt),
        resort, jitter, weighted_mean, mean_bjd
    )

    return (; periods = collect(periods), k_search, k_opt, init_fit, sweep, final_fit)
end

"""
    search_planets_loocv(bjd, rvs, σ_rv, ccfs, period_list;
                         max_num_pl=3, k_search,
                         max_num_basis=16, min_period_ratio=1.0,
                         resort=false, jitter=0.0,
                         weighted_mean=true) -> NamedTuple

Greedily search for up to `max_num_pl` planets using LOOCV-based SCALPELS,
then fit each cumulative period list with [`fit_planets_loocv`](@ref).

For each planet count `num_pl` from 1 to `max_num_pl`:
1. **Period scan** — call `vscalpels_recover_loocv` at `k_search` vectors for
   every trial period in `period_list` not excluded by `min_period_ratio`
   (with previously found periods fixed), retaining the full output at each
   point for diagnostics.
2. **Best period selection** — choose the trial period minimising χ².
3. **Full fit** — call [`fit_planets_loocv`](@ref) with the accumulated period
   list; this runs the initial fit, k sweep, and final fit.

Returns a NamedTuple `(; results)` where `results` is a `Vector` of
NamedTuples (one per `num_pl`) with fields:
- `period_list`: trial periods scanned at this step (after `min_period_ratio` filtering).
- `search_outs`: `Vector` of full `vscalpels_recover_loocv` outputs (one per
  trial period) — contains all fields including `χ²`, `amp`, `rvresid`, etc.
- `fit_result`: output of `fit_planets_loocv` for the chosen periods —
  contains `periods`, `k_search`, `k_opt`, `init_fit`, `sweep`, `final_fit`.

# Arguments
- `bjd`: Observation times (days).
- `rvs`: Observed RVs (m/s).
- `σ_rv`: RV uncertainties (m/s).
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `period_list`: Vector of trial periods (days) to scan at each step.

# Keyword Arguments
- `max_num_pl`: Number of planets to search for (default 3).
- `k_search`: Activity vectors used during the period scan; passed to
  `fit_planets_loocv` as well. Should be ≥ 1.
- `max_num_basis`: Upper bound on the k sweep in `fit_planets_loocv` (default 16).
- `min_period_ratio`: Trial periods within a factor of `min_period_ratio` of any
  already-found period are skipped. A period `P` is skipped when
  `max(P, P_best) / min(P, P_best) < min_period_ratio` for any `P_best` in the
  accumulated list. Default `1.0` (no filtering).
- `resort`, `jitter`, `weighted_mean`: passed through to underlying calls.
- `mean_bjd`: Reference epoch (days) passed to `vscalpels_recover_loocv`
  and `fit_planets_loocv`. Defaults to `mean(bjd)`.
"""
function search_planets_loocv(
        bjd::AbstractVector{<:Real},
        rvs::AbstractVector{<:Real},
        σ_rv::AbstractVector{<:Real},
        ccfs::AbstractMatrix{<:Real},
        period_list::AbstractVector{<:Real};
        max_num_pl::Integer = 3,
        k_search::Integer,
        max_num_basis::Integer = 16,
        min_period_ratio::Real = 1.0,
        resort::Bool = false,
        fixed_k::Bool = false,
        jitter::Real = 0.0,
        weighted_mean::Bool = true,
        mean_bjd::Real = mean(bjd)
    )
    best_periods = Float64[]
    results = NamedTuple[]

    for num_pl in 1:max_num_pl

        # Step 1: scan the period grid, storing full vscalpels_recover_loocv
        # output at every trial period for downstream diagnostics.
        # Skip periods too close to already-found periods (avoids fitting aliases).
        filtered_period_list = filter(period_list) do P
            all(max(P, P_best) / min(P, P_best) >= min_period_ratio
                for P_best in best_periods)
        end
        search_outs = map(filtered_period_list) do P
            vscalpels_recover_loocv(
                bjd, rvs, σ_rv, ccfs, vcat(best_periods, [P]);
                max_scalpels_vectors = k_search,
                resort, jitter, weighted_mean, mean_bjd
            )
        end

        # Step 2: pick the period with the lowest χ².
        best_P = filtered_period_list[argmin([out.χ² for out in search_outs])]
        push!(best_periods, best_P)

        # Step 3: full fit (initial + k sweep + final) for the accumulated periods.
        fit_result = fit_planets_loocv(
            bjd, rvs, σ_rv, ccfs, copy(best_periods);
            k_search, max_num_basis, resort, fixed_k, jitter, weighted_mean, mean_bjd
        )

        push!(results, (;
            period_list = collect(filtered_period_list),
            search_outs,
            fit_result,
        ))
    end

    return (; results)
end

"""
    search_planets_loocv_joint(inst_data, period_list;
                               max_num_pl=3, k_search,
                               max_num_basis=16, min_period_ratio=1.0,
                               resort=false, jitter=0.0,
                               weighted_mean=true) -> NamedTuple

Greedily search for up to `max_num_pl` planets by summing χ² across all
instruments at each trial period, so every instrument shares the same
`best_periods` list.  Per-instrument fits are performed independently once
the joint period selection is complete.

For each planet count `num_pl` from 1 to `max_num_pl`:
1. **Period scan** — call `vscalpels_recover_loocv` at `k_search` vectors for
   every unfiltered trial period and every instrument; sum χ² across instruments.
2. **Best period selection** — choose the trial period minimising the total χ².
3. **Full fit** — call [`fit_planets_loocv`](@ref) independently for each
   instrument with the accumulated shared period list.

Returns a NamedTuple `(; results)` where `results` is a `Vector` of
NamedTuples (one per `num_pl`) with fields:
- `period_list`: trial periods scanned at this step (after `min_period_ratio`
  filtering).
- `chi2_total`: total χ² summed across instruments for each trial period —
  used to select `best_P`.
- `inst_search_outs`: `Vector{Vector}` of per-instrument `vscalpels_recover_loocv`
  outputs. `inst_search_outs[i]` is a vector over trial periods for instrument
  `i`, mirroring `search_outs` from [`search_planets_loocv`](@ref).
- `fit_results`: `Vector` of [`fit_planets_loocv`](@ref) outputs, one per
  instrument, for the chosen (shared) periods.

# Arguments
- `inst_data`: `AbstractVector` of NamedTuples, one per instrument.  Each
  entry must have fields `bjd`, `rvs`, `σ_rv`, and `ccfs`.
- `period_list`: Vector of trial periods (days) to scan at each step.

# Keyword Arguments
- `max_num_pl`: Number of planets to search for (default 3).
- `k_search`: Activity vectors used during the period scan and passed to
  `fit_planets_loocv`. Should be ≥ 1.
- `max_num_basis`: Upper bound on the k sweep in `fit_planets_loocv` (default 16).
- `min_period_ratio`: Trial periods within a factor of `min_period_ratio` of any
  already-found period are skipped (see [`search_planets_loocv`](@ref)).
- `resort`, `fixed_k`, `jitter`, `weighted_mean`: passed through to underlying
  calls.
- `mean_bjd`: Reference epoch (days) passed to `vscalpels_recover_loocv` and
  `fit_planets_loocv`. Defaults to the mean of all BJDs across all instruments.
"""
function search_planets_loocv_joint(
        inst_data::AbstractVector,
        period_list::AbstractVector{<:Real};
        max_num_pl::Integer = 3,
        k_search::Integer,
        max_num_basis::Integer = 16,
        min_period_ratio::Real = 1.0,
        resort::Bool = false,
        fixed_k::Bool = false,
        jitter::Real = 0.0,
        weighted_mean::Bool = true,
        mean_bjd::Union{Real,Nothing} = nothing
    )
    ninst = length(inst_data)
    actual_mean_bjd::Float64 = isnothing(mean_bjd) ?
        mean(reduce(vcat, d.bjd for d in inst_data)) : Float64(mean_bjd)
    best_periods = Float64[]
    results = NamedTuple[]

    for num_pl in 1:max_num_pl

        # Filter periods too close to any already-found period.
        filtered_period_list = filter(period_list) do P
            all(max(P, P_best) / min(P, P_best) >= min_period_ratio
                for P_best in best_periods)
        end
        np = length(filtered_period_list)

        # Scan each (period, instrument) pair.
        # Outer index = period, inner index = instrument for easy χ² summation.
        scan = [
            vscalpels_recover_loocv(
                inst_data[i].bjd, inst_data[i].rvs, inst_data[i].σ_rv,
                inst_data[i].ccfs, vcat(best_periods, [P]);
                max_scalpels_vectors = k_search,
                resort, jitter, weighted_mean, mean_bjd = actual_mean_bjd
            )
            for P in filtered_period_list, i in 1:ninst
        ]
        # scan[ip, i] = output for period ip, instrument i

        # Sum χ² across instruments and pick the best period.
        chi2_total = [sum(scan[ip, i].χ² for i in 1:ninst) for ip in 1:np]
        best_P = filtered_period_list[argmin(chi2_total)]
        push!(best_periods, best_P)

        # Per-instrument full fit (initial + k sweep + final) at shared periods.
        fit_results = [
            fit_planets_loocv(
                inst_data[i].bjd, inst_data[i].rvs, inst_data[i].σ_rv,
                inst_data[i].ccfs, copy(best_periods);
                k_search, max_num_basis, resort, fixed_k, jitter, weighted_mean,
                mean_bjd = actual_mean_bjd
            )
            for i in 1:ninst
        ]

        # Reorganise scan so inst_search_outs[i] mirrors search_outs from the
        # single-instrument version (a vector over trial periods).
        inst_search_outs = [
            [scan[ip, i] for ip in 1:np]
            for i in 1:ninst
        ]

        push!(results, (;
            period_list     = collect(filtered_period_list),
            chi2_total,
            inst_search_outs,
            fit_results,
        ))
    end

    return (; results)
end

"""
    aic_zero_planet_vs_num_basis_loocv(u_loocv, α_loocv, rv_centered, σ_rv;
                                        jitter=0.0, max_num_basis) -> NamedTuple

Compute χ², AIC, and RMS of the 0-planet LOOCV model as a function of the
number of activity basis vectors used.

`u_loocv` and `α_loocv` must already be in the desired selection order (e.g.,
as permuted by [`reorder_uloocv`](@ref)). Each successive column pair
`(u[:,k], α[:,k])` is subtracted from the running RV residuals.

AIC is computed as χ²(k) + 2k, where k counts only the activity vectors.
This is consistent with `rms_clean_rvs_with_planets_vs_num_basis_scalpels`
(which uses χ² + 2*(k + 2*nplanets)) specialised to nplanets = 0.

# Arguments
- `u_loocv`: `(num_obs × num_vecs)` ordered LOOCV shape vectors.
- `α_loocv`: `(num_obs × num_vecs)` ordered LOOCV RV projections.
- `rv_centered`: Mean-subtracted observed RVs, length `num_obs`.
- `σ_rv`: RV uncertainties, length `num_obs`.

# Keyword Arguments
- `jitter`: Additional jitter (m/s) added in quadrature to `σ_rv`.
- `max_num_basis`: Maximum k to evaluate. Defaults to `size(u_loocv, 2)`.

# Returns
NamedTuple with fields:
- `num_basis`: `0:max_num_basis` — the k values evaluated.
- `χ²`: χ² of residuals at each k.
- `aic`: AIC = χ²(k) + 2k at each k.
- `rms`: RMS of residuals at each k.
"""
function aic_zero_planet_vs_num_basis_loocv(
        u_loocv::AbstractMatrix{<:Real},
        α_loocv::AbstractMatrix{<:Real},
        rv_centered::AbstractVector{<:Real},
        σ_rv::AbstractVector{<:Real};
        jitter::Real = 0.0,
        max_num_basis::Integer = size(u_loocv, 2)
    )
    @assert size(u_loocv, 2) == size(α_loocv, 2)
    @assert size(u_loocv, 1) == length(rv_centered) == length(σ_rv)

    max_num_basis = min(max_num_basis, size(u_loocv, 2))
    invar = 1.0 ./ (σ_rv.^2 .+ jitter^2)

    num_basis = 0:max_num_basis
    χ²_list  = zeros(max_num_basis + 1)
    aic_list = zeros(max_num_basis + 1)
    rms_list = zeros(max_num_basis + 1)

    rv_resid = copy(rv_centered)
    χ²_list[1]  = sum(rv_resid.^2 .* invar)
    rms_list[1] = std(rv_resid)
    aic_list[1] = χ²_list[1]   # k = 0 parameters

    for k in 1:max_num_basis
        rv_resid  .-= view(u_loocv, :, k) .* view(α_loocv, :, k)
        χ²_list[k+1]  = sum(rv_resid.^2 .* invar)
        rms_list[k+1] = std(rv_resid)
        aic_list[k+1] = χ²_list[k+1] + 2 * k
    end

    return (; num_basis, χ²=χ²_list, aic=aic_list, rms=rms_list)
end
