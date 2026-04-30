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