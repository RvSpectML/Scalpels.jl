# src/internals.jl
# Internal helper functions. Not exported.

"""
    calc_max_vectors(ccfs::AbstractMatrix) -> Int

Return the maximum number of basis vectors that can be computed from a CCF
matrix, defined as `min(num_obs, num_vel_bins - 1)`.
"""
calc_max_vectors(ccfs::AbstractMatrix) = min(size(ccfs, 1), size(ccfs, 2) - 1)

"""
    is_centered(x, [w]; tol=1e-6) -> Bool

Return `true` if the (optionally weighted) mean of `x` is within `tol` of zero.
"""
function is_centered(x::AbstractVector; tol::Real = 1e-6)
    abs(mean(x)) <= tol
end

function is_centered(x::AbstractVector, w::AbstractVector; tol::Real = 1e-6)
    abs(mean(x, weights(w))) <= tol
end

function is_centered(x::AbstractVector, w::Real; tol::Real = 1e-6)
    abs(mean(x)) <= tol
end

"""
    center_acfs(acfs, [w]) -> AbstractMatrix

Subtract the (optionally weighted) row-wise mean from `acfs`.

# Arguments
- `acfs`: Matrix of autocorrelation functions, size `(num_vel_bins, num_obs)`.
- `w`: Optional weight vector of length `num_obs`. If omitted or scalar,
  unweighted mean is used.
"""
function center_acfs(acfs::AbstractMatrix; mean_acf = mean(acfs, dims=2))
    acfs.- mean_acf
end

function center_acfs(acfs::AbstractMatrix, w::AbstractVector)
    mean_acf = mean(acfs, weights(w), dims=2)
    acfs.- mean_acf
end

function center_acfs(acfs::AbstractMatrix, w::Real)
    mean_acf = mean(acfs, dims=2)
    acfs.- mean_acf
end