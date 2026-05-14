# src/utils.jl
# Miscellaneous utility functions.

"""
    make_period_list(t, Pmin, Pmax; oversample_factor=1.0) -> Vector{Float64}

Generate a list of trial periods for a period search, uniformly spaced in
frequency with spacing set by the observation baseline.

The frequency step is `Δf = 0.5 / (t[end] - t[1]) / oversample_factor`,
following standard Lomb-Scargle frequency grid conventions.

# Arguments
- `t`: Sorted vector of observation times (days), length ≥ 2.
- `Pmin`: Minimum trial period (days).
- `Pmax`: Maximum trial period (days).

# Keyword Arguments
- `oversample_factor`: Factor by which to oversample the frequency grid
  relative to the Nyquist-like spacing (default 1.0).

# Returns
Vector of trial periods in days, ordered from longest to shortest.
"""
function make_period_list(
        t::AbstractVector,
        Pmin::Real,
        Pmax::Real;
        oversample_factor::Real = 1.0
    )
    @assert Pmin < Pmax
    @assert length(t) >= 2
    @assert issorted(t)
    obs_span = last(t) - first(t)
    Δfreq    = 0.5 / obs_span / oversample_factor
    freq_list = reverse(range(1/Pmax, stop=1/Pmin, step=Δfreq))
    return 1.0./ freq_list
end

"""
    estimate_continuum(v_grid, ccf; quantile_level=0.9,
                       line_half_width=15000.0, v_center=0.0) -> Float64

Estimate the continuum level of a single CCF by taking a high quantile of the
CCF values at velocities outside a central exclusion window.

Pixels with `|v - v_center| < line_half_width` are excluded before computing
the quantile, preventing the line core from biasing the continuum estimate
downward.

# Arguments
- `v_grid`: Velocity grid in m/s, length `num_vel_bins`.
- `ccf`: CCF values, length `num_vel_bins`.

# Keyword Arguments
- `quantile_level`: Quantile used for the continuum estimate (default 0.9).
- `line_half_width`: Half-width of the line-core exclusion region in m/s
  (default 15000.0, i.e. ±15 km/s). Must be narrow enough to leave at least
  one unmasked velocity bin.
- `v_center`: Velocity of the line centre in m/s (default 0.0). Pixels within
  `line_half_width` of this value are excluded.
"""
function estimate_continuum(
        v_grid::AbstractVector,
        ccf::AbstractVector;
        quantile_level::Real = 0.9,
        line_half_width::Real = 15000.0,
        v_center::Real = 0.0
    )
    @assert length(v_grid) == length(ccf)
    mask = abs.(v_grid .- v_center) .>= line_half_width
    @assert any(mask) "line_half_width=$line_half_width excludes all velocity bins; reduce it"
    return mean(view(ccf, mask)) #, quantile_level)
end

"""
    estimate_continuum(v_grid, ccfs::AbstractMatrix; quantile_level=0.9,
                       line_half_width=15000.0, v_center=0.0) -> Vector{Float64}

Estimate the continuum level for each observation (column) in a 2-D CCF matrix
of shape `(num_vel_bins, num_obs)`.

Returns a vector of length `num_obs`.

See also: [`estimate_continuum`](@ref) for the single-CCF method.
"""
function estimate_continuum(
        v_grid::AbstractVector,
        ccfs::AbstractMatrix{T};
        quantile_level::Real = 0.9,
        line_half_width::Real = 15000.0,
        v_center::Real = 0.0
    ) where {T<:Real}
    continuum = zeros(T, size(ccfs,2))
    for ord in 1:size(ccfs,2)
        continuum[ord] = estimate_continuum(v_grid, view(ccfs, :, ord);
                                quantile_level, line_half_width, v_center)
        end
    return continuum
end

"""
    estimate_continuum(v_grid, ccfs::AbstractArray{T,3}; quantile_level=0.9,
                       line_half_width=15000.0, v_center=0.0) -> Matrix{Float64}

Estimate the continuum level for each `(order, observation)` pair in a 3-D CCF
array with axes `(num_vel_bins, num_orders, num_obs)`.

Returns a matrix of shape `(num_orders, num_obs)`.

See also: [`estimate_continuum`](@ref) for the single-CCF method.
"""
function estimate_continuum(
        v_grid::AbstractVector,
        ccfs::AbstractArray{T,3};
        quantile_level::Real = 0.9,
        line_half_width::Real = 15000.0,
        v_center::Real = 0.0
    ) where {T<:Real}

    continuum = zeros{T}(size(ccfs)[2:end])
    @info size(continuum)
    for ord in 1:size(ccfs,3)
        for obs in 1:size(ccfs,2)
            continuum[ord,obs] = estimate_continuum(v_grid, view(ccfs, :, ord, obs);
                                    quantile_level, line_half_width, v_center)
        end
    end
    return continuum
end

"""
    svd_reconstruction(ccf; n=size(ccf,2)) -> Matrix

Reconstruct a CCF matrix using only the leading `n` singular value components.

Useful for denoising or visualising the low-rank structure of the CCF matrix.

!!! warning
    This function is marked as **experimental**. Its interface or behaviour
    may change in a future version of Scalpels.

# Arguments
- `ccf`: Input matrix to decompose, size `(num_vel_bins, num_obs)`.

# Keyword Arguments
- `n`: Number of singular components to retain (default: full rank).
"""
function svd_reconstruction(ccf::AbstractMatrix; n::Integer = size(ccf, 2))
    F = svd(ccf)
    return view(F.U, :, 1:n) * Diagonal(view(F.S, 1:n)) * view(F.Vt, 1:n, :)
end
