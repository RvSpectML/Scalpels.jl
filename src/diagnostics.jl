# src/diagnostics.jl
# Quality control and outlier masking functions.

"""
    mask_outliers(ccfs, rvs, σrvs; threshold=7,
                  max_scalpels_vectors) -> NamedTuple

Identify outlier observations based on their deviation from the median in
the space of CCF autocorrelation SVD coefficients.

For each of the leading `max_scalpels_vectors` left singular vectors of the
ACF matrix, observations deviating more than `threshold` median absolute
deviations (MADs) from the median are flagged.

Returns a named tuple `(obs_mask, badfrac_vs_threshold)`:
- `obs_mask`: Boolean vector of length `num_obs`. `true` = good observation.
- `badfrac_vs_threshold`: Fraction of observations flagged at each integer
  threshold level from 1 to 20, useful for diagnosing the threshold choice.

# Arguments
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `rvs`: Observed RVs, length `num_obs`.
- `σrvs`: RV uncertainties, length `num_obs`.

# Keyword Arguments
- `threshold`: MAD multiplier above which an observation is flagged (default 7).
- `max_scalpels_vectors`: Number of SVD vectors to use for outlier detection.
"""
function mask_outliers(
        ccfs::AbstractMatrix,
        rvs::AbstractVector,
        σrvs::AbstractVector;
        threshold::Real = 7,
        max_scalpels_vectors::Integer = calc_max_vectors(ccfs)
    )

    acfs = autocor(ccfs, 0:size(ccfs, 1)-1)
    svd_out = svd(acfs')
    u = view(svd_out.U, :, 1:max_scalpels_vectors)

    colmed_u  = median(u, dims=1)
    absdev_u  = abs.(u.- colmed_u)
    colmad_u  = median(absdev_u, dims=1)
    # Normalize deviations by MAD.
    absdev_u./= colmad_u

    num_threshold_levels = 20
    badfrac = zeros(num_threshold_levels)
    for i in 1:num_threshold_levels
        goodmask =.!(absdev_u.> i.* colmad_u)
        rowmask  = all(goodmask, dims=2)
        badfrac[i] = sum(.!rowmask) / length(rowmask)
    end

    #goodmask =.!(absdev_u.> threshold.* colmad_u)
    goodmask =.!(absdev_u.> threshold)
    rowmask  = vec(all(goodmask, dims=2))

    return (; obs_mask = rowmask, badfrac_vs_threshold = badfrac)
end

"""
    quality_control(ccfs, rvs, σrvs; mad_factor=3, threshold=7,
                    max_scalpels_vectors) -> NamedTuple

Estimate the optimal number of LOOCV shape vectors and identify outlier
observations.

First masks outliers using [`mask_outliers`](@ref), then compares the LOOCV
shape vectors on the cleaned dataset to the plain SVD vectors. The optimal
number of vectors `kopt` is estimated as the point where the LOOCV vectors
begin to diverge significantly from the SVD vectors (as measured by the ratio
of their MADs).

Returns a named tuple `(obs_mask, kopt, madratio)`:
- `obs_mask`: Boolean vector, `true` = good observation.
- `kopt`: Estimated optimal number of LOOCV shape vectors (diagnostic only;
  use as a guide when setting `max_scalpels_vectors` in
  [`vscalpels_loocv`](@ref)).
- `madratio`: Vector of MAD ratios for each vector index, useful for
  visualising the divergence.

# Arguments
- `ccfs`: CCF matrix, size `(num_vel_bins, num_obs)`.
- `rvs`: Observed RVs, length `num_obs`.
- `σrvs`: RV uncertainties, length `num_obs`.

# Keyword Arguments
- `mad_factor`: Multiplier controlling sensitivity of the `kopt` estimator
  (default 3). Larger values are more conservative (select fewer vectors).
- `threshold`: MAD threshold passed to [`mask_outliers`](@ref) (default 7).
- `max_scalpels_vectors`: Maximum number of vectors to evaluate.
"""
function quality_control(
        ccfs::AbstractMatrix,
        rvs::AbstractVector,
        σrvs::AbstractVector;
        mad_factor::Real = 3,
        threshold::Real = 7,
        max_scalpels_vectors::Integer = calc_max_vectors(ccfs)
    )

    obs_mask = mask_outliers(ccfs, rvs, σrvs; threshold).obs_mask

    ccfs_clean = view(ccfs, :, obs_mask)
    rvs_clean  = view(rvs, obs_mask)
    σrvs_clean = view(σrvs, obs_mask)

    acfs    = autocor(ccfs_clean, 0:size(ccfs_clean, 1)-1)
    svd_out = svd(acfs')

    scalpels_out = vscalpels_loocv(
        ccfs_clean, rvs_clean, σrvs_clean;
        max_scalpels_vectors,
        resort = false
    )
    h = scalpels_out.u_loocv
    d = view(svd_out.U, :, 1:max_scalpels_vectors).- h

    colmad_d  = median(abs.(d.- median(d, dims=1)), dims=1)
    colmad_h  = median(abs.(h.- median(h, dims=1)), dims=1)
    madratio  = vec(colmad_d./ colmad_h)

    kopt = 0
    for kk in 1:length(madratio)-1
        if (madratio[kk] > 0.25) &&
           (madratio[kk] > 1 - mad_factor * median(view(madratio, kk+1:length(madratio))))
            kopt = kk
            break
        end
    end

    return (; obs_mask, kopt, madratio)
end