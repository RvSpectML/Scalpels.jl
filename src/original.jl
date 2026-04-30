# src/original.jl
# Original SCALPELS functions (Collier Cameron et al. 2021).
# Extended with optional weighted mean subtraction.

"""
    calc_basis_scores_scalpels(rvs_centered, ccfs; σ_rvs=1.0,
                               num_basis, sort_by_responce=true,
                               assume_centered=false,
                               weighted_mean=true) -> NamedTuple

Compute the SCALPELS basis vectors and their scores (projections onto the
centered RVs) from a matrix of CCFs.

Returns a named tuple `(scores, basis)` where:
- `scores`: `(num_obs × num_basis)` matrix of basis scores.
- `basis`: `(num_basis × num_vel_bins)` matrix of basis vectors.

# Arguments
- `rvs_centered`: Mean-subtracted observed RVs, length `num_obs`.
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.

# Keyword Arguments
- `σ_rvs`: RV uncertainties (scalar or vector). Used for weighting.
- `num_basis`: Number of basis vectors to retain.
- `sort_by_responce`: If `true`, sort basis vectors by their correlation with
  the RVs rather than by singular value.
- `assume_centered`: If `true`, skip the centering assertion check.
- `weighted_mean`: If `true` (default), subtract the inverse-variance weighted
  mean ACF. If `false`, subtract the unweighted mean ACF.

!!! note
    Prior to this parameter being added, the original implementation used
    unweighted mean subtraction (equivalent to `weighted_mean=false`).
    Results may differ slightly when `σ_rvs` are non-uniform.
"""
function calc_basis_scores_scalpels(
        rvs_centered::AbstractVector{T1},
        ccfs::AbstractMatrix{T2};
        σ_rvs = 1.0,
        num_basis::Integer,
        sort_by_responce::Bool = true,
        assume_centered::Bool = false,
        weighted_mean::Bool = true
    ) where {T1<:Real, T2<:Real}

    num_obs = length(rvs_centered)
    @assert size(ccfs, 2) == num_obs
    obs_weights = 1.0./ σ_rvs.^2
    if !assume_centered
        @assert is_centered(rvs_centered, obs_weights)
    end

    acfs = autocor(ccfs, 0:size(ccfs, 1)-1)
    acfs_minus_mean = if weighted_mean && !all(σ_rvs.== first(σ_rvs))
        center_acfs(acfs, vec(obs_weights))
    else
        center_acfs(acfs)
    end

    svd_acfs = svd(acfs_minus_mean')
    U = svd_acfs.U
    alpha = U' * rvs_centered

    idx = if sort_by_responce
        sortperm(abs.(alpha), rev=true)
    else
        1:size(U, 2)
    end

    U_keep = view(U, :, idx[1:num_basis])
    P_keep = view(svd_acfs.Vt, idx[1:num_basis], :)
    return (scores = U_keep, basis = P_keep)
end

"""
    clean_rvs_scalpels(rvs, ccfs; σ_rvs=1.0, num_basis,
                       sort_by_responce=true,
                       weighted_mean=true) -> AbstractVector

Remove activity-driven RV variations using the SCALPELS method.

Projects the observed RVs onto the subspace spanned by the leading CCF
autocorrelation basis vectors, then subtracts that projection.

# Arguments
- `rvs`: Observed RVs, length `num_obs`.
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.

# Keyword Arguments
- `σ_rvs`: RV uncertainties (scalar or vector).
- `num_basis`: Number of basis vectors to use for cleaning.
- `sort_by_responce`: Sort basis vectors by RV correlation before truncating.
- `weighted_mean`: If `true` (default), use inverse-variance weighted mean ACF
  subtraction. If `false`, use unweighted mean.

!!! note
    Prior to this parameter being added, the original implementation used
    unweighted mean subtraction (equivalent to `weighted_mean=false`).
    Results may differ slightly when `σ_rvs` are non-uniform.
"""
function clean_rvs_scalpels(
        rvs::AbstractVector{T1},
        ccfs::AbstractMatrix{T2};
        σ_rvs = 1.0,
        num_basis::Integer,
        sort_by_responce::Bool = true,
        weighted_mean::Bool = true
    ) where {T1<:Real, T2<:Real}

    obs_weights = 1.0./ σ_rvs.^2
    rvs_mean = mean(rvs, weights(obs_weights))
    rvs_centered = rvs.- rvs_mean

    scores_out = calc_basis_scores_scalpels(
        rvs_centered, ccfs;
        σ_rvs, num_basis, sort_by_responce,
        assume_centered = true,
        weighted_mean
    )
    U_keep = scores_out.scores
    Δrv_shape = U_keep * U_keep' * rvs_centered
    return rvs.- Δrv_shape
end

"""
    rms_clean_rvs_vs_num_basis_scalpels(rvs, ccfs; σ_rvs=1.0,
                                        max_num_basis,
                                        sort_by_responce=true,
                                        weighted_mean=true) -> Vector{Float64}

Compute the RMS of the cleaned RVs as a function of the number of SCALPELS
basis vectors used, from 0 up to `max_num_basis`.

Useful for diagnosing how many basis vectors are needed to adequately remove
stellar activity.

# Arguments
- `rvs`: Observed RVs, length `num_obs`.
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.

# Keyword Arguments
- `σ_rvs`: RV uncertainties (scalar or vector).
- `max_num_basis`: Maximum number of basis vectors to test.
- `sort_by_responce`: Sort basis vectors by RV correlation.
- `weighted_mean`: If `true` (default), use inverse-variance weighted mean ACF
  subtraction. If `false`, use unweighted mean.

!!! note
    Prior to this parameter being added, the original implementation used
    unweighted mean subtraction (equivalent to `weighted_mean=false`).
    Results may differ slightly when `σ_rvs` are non-uniform.
"""
function rms_clean_rvs_vs_num_basis_scalpels(
        rvs::AbstractVector{T1},
        ccfs::AbstractMatrix{T2};
        σ_rvs = 1.0,
        max_num_basis::Integer,
        sort_by_responce::Bool = true,
        weighted_mean::Bool = true
    ) where {T1<:Real, T2<:Real}

    rms_list = zeros(max_num_basis + 1)
    rms_list[1] = std(rvs)
    for k in 1:max_num_basis
        rvs_clean = clean_rvs_scalpels(
            rvs, ccfs;
            σ_rvs, num_basis = k,
            sort_by_responce,
            weighted_mean
        )
        rms_list[k+1] = std(rvs_clean)
    end
    return rms_list
end