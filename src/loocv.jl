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
    rv_centered = rv.- mean(rv)

    α_loocv, u_loocv = loocv(
        rv_centered, ccfs;
        max_scalpels_vectors,
        weighted_mean
    )

    if resort
        idx_perm, _, aic_list, _ = reorder_uloocv(
            u_loocv, α_loocv, rv_centered, σ_rv;
            jitter, max_scalpels_vectors
        )
        kmax     = argmin(aic_list)
        α_loocv  = α_loocv[:, idx_perm[1:kmax-1]]
        u_loocv  = u_loocv[:, idx_perm[1:kmax-1]]
    else
        α_loocv = view(α_loocv, :, 1:kmax)
        u_loocv = view(u_loocv, :, 1:kmax)
    end

    v_resp   = u_loocv.* α_loocv
    rv_shape = sum(v_resp, dims=2)
    rv_clean = vec(rv_centered.- rv_shape)

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
        weighted_mean::Bool = true
    )

    @assert max_scalpels_vectors >= 0
    kmax = max_scalpels_vectors
    nobs = length(bjd)
    mean_bjd   = mean(bjd)
    rv_centered = rvs.- mean(rvs)

    loocv_out = loocv(
        rv_centered, ccfs;
        max_scalpels_vectors,
        weighted_mean
    )
    u_loocv = loocv_out.u_loocv
    α_loocv = loocv_out.α_loocv

    if resort
        idx_perm, χ²_list, aic_list, _ = reorder_uloocv(
            u_loocv, α_loocv, rv_centered, σ_rv;
            jitter, max_scalpels_vectors
        )
        if max_scalpels_vectors > 0
            kmax    = argmin(aic_list)
            α_loocv = α_loocv[:, idx_perm[1:kmax-1]]
            u_loocv = u_loocv[:, idx_perm[1:kmax-1]]
        end
    end

    # Build sinusoidal design matrix for all trial periods.
    nplanets = length(periods)
    fftrn = Matrix{Float64}(undef, nobs, 2 * nplanets)
    for planet in 1:nplanets
        phi = 2π / periods[planet].* (bjd.- mean_bjd)
        fftrn[:, 2*planet-1] = cos.(phi)
        fftrn[:, 2*planet]   = sin.(phi)
    end

    # Projection operator onto the complement of the activity subspace.
    pperp = I(nobs)
    if kmax > 0
        pperp = pperp - u_loocv * u_loocv'
    end

    fperp = pperp * fftrn
    vperp = pperp * rv_centered

    # Weighted least-squares for Keplerian amplitudes.
    invar = Diagonal(σ_rv.^(-2))
    amat  = (fperp' * invar) * fperp
    bvec  = (fperp' * invar) * vperp
    theta    = amat \ bvec
    theterr  = sqrt.(diag(inv(amat)))

    # Convert (cos, sin) amplitudes to velocity semi-amplitudes.
    amp    = Vector{Float64}(undef, nplanets)
    amperr = Vector{Float64}(undef, nplanets)
    for planet in 1:nplanets
        icol = 2 * planet - 1
        Kx, dKx = theta[icol],   theterr[icol]
        Ky, dKy = theta[icol+1], theterr[icol+1]
        K  = sqrt(Kx^2 + Ky^2)
        dK = sqrt((Kx * dKx)^2 + (Ky * dKy)^2) / K
        amp[planet]    = K
        amperr[planet] = dK
    end

    rvorbit  = fftrn * theta
    rvclean  = vperp
    rvshape  = rv_centered.- rvclean
    rvresid  = vperp.- fperp * theta
    χ²       = sum(abs2.(rvresid./ σ_rv))

    return (;
        rv_centered, rvshape, rvclean, rvorbit, rvresid,
        periods, amp, amperr, u_loocv, α_loocv, fftrn, χ²
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