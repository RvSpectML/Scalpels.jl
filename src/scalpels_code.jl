"""   `calc_basis_scores_scalpels(rvs, ccfs; σ_rvs, num_basis )`
Compute the CCF basis functions and scores for a Scalpels reconstruction of CCFs
Inputs:
- rvs:  vector of estimated radial velocities
- ccfs: 2d array of CCFS of size (num_velocities, num_spectra)
Optional Inputs:
- σ_rvs:  vector of measurement uncertainties for estimated radial velocities (default: ones)
- num_basis:  number of basis vectors to use for SVD reconstruction of CCFs
- sort_by_responce:  set true to sort basis vectors by RV responce (default: true)
Output:
(scores, basis): a NamedTuple

Notes:
- Currently Scalpels weights all velocity pixels equally and uses the σ_rvs to weight each observation.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function calc_basis_scores_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                num_basis::Integer = 3, assume_centered::Bool = false, sort_by_responce::Bool = true ) where { T1<:Real, T2<:Real, T3<:Real }
    @assert length(rvs) == length(σ_rvs)
    @assert length(rvs) == size(ccfs,2)
    @assert 1 <= num_basis < length(rvs)
    if assume_centered
        rvs_centered = rvs
    else
        mean_rv = mean(rvs, weights(1.0 ./ σ_rvs.^2 ))
        rvs_centered = rvs .- mean_rv
    end

    acfs = autocor(ccfs,0:size(ccfs,1)-1)
    if all(σ_rvs.==first(σ_rvs))
        acfs_minus_mean = acfs .- mean(acfs,dims=2)
    else
        acfs_minus_mean = acfs .- mean(acfs,weights(1.0 ./ σ_rvs.^2),dims=2)
    end
    #Δv_grid = convert(Float64,v_grid.step).*(0:size(acfs,1)-1)
    svd_acfs = svd(acfs_minus_mean')
    #return (basis=svd_acfs.U, scores=svd_acfs.S)  For ACF's SVD eigenvectors and eigenvalues

    alpha = svd_acfs.U'*rvs_centered
    if sort_by_responce
        idx = sortperm(abs.(alpha),rev=true)
        #println("# Ordered by alphas: ", idx[1:num_basis])
    else
        # To throw away sorting by projection onto RVs
        idx = 1:length(alpha)
    end
    U_keep = view(svd_acfs.U,:,idx[1:num_basis])
    P_keep = view(svd_acfs.Vt',:,idx[1:num_basis])
    return (scores=U_keep, basis=P_keep) # , responce=alpha[idx[1:num_basis]])

    #U_keep = view(svd_acfs.U,:,1:num_basis)
    #return (basis=U_keep, scores = view(alpha,1:num_basis) )
    #=  To clean RVs
    Δrv_shape = U_keep*U_keep'*rvs_centered
    rvs_clean = rvs .- Δrv_shape
    return rvs_clean
    =#
end


"""   `clean_rvs_scalpels(rvs, ccfs; σ_rvs, num_basis )`
Inputs:
- rvs:  vector of estimated radial velocities
- ccfs: 2d array of CCFS of size (num_velocities, num_spectra)
Optional Inputs:
- σ_rvs:  vector of measurement uncertainties for estimated radial velocities (default: ones)
- num_basis:  number of basis vectors to use for SVD reconstruction of CCFs
- sort_by_responce:  set true to sort basis vectors by RV responce (default: true)
Output:
rvs_clean: vector of estimated RVs after cleaning by scalpels

Notes:
- Currently Scalpels weights all observations equally and doesn't use the σ_rvs.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function clean_rvs_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                num_basis::Integer = 3, sort_by_responce::Bool = true ) where { T1<:Real, T2<:Real, T3<:Real }
    @assert length(rvs) == length(σ_rvs)
    @assert length(rvs) == size(ccfs,2)
    if num_basis == 0   return rvs   end
    @assert 0 <= num_basis < length(rvs)
    mean_rv = mean(rvs, weights(1.0 ./ σ_rvs.^2 ))
    rvs_centered = rvs .- mean_rv

    #=
    acfs = autocor(ccfs,0:size(ccfs,1)-1)
    acfs_minus_mean = acfs .- mean(acfs,dims=2)
    #Δv_grid = convert(Float64,v_grid.step).*(0:size(acfs,1)-1)
    svd_acfs = svd(acfs_minus_mean')
    alpha = svd_acfs.U'*rvs_centered

    idx = sortperm(abs.(alpha),rev=true)
    U_keep = view(svd_acfs.U,:,idx[1:num_basis])

    #(U, scores ) = calc_basis_scores_scalpels(rvs_centered, ccfs, σ_rvs=σ_rvs, num_basis=num_basis, assume_centered=true)
    idx = sortperm(abs.(alpha),rev=true)
    alpha = U'*rvs_centered
    U_keep = view(U,:,idx[1:num_basis])
    =#

    (U_keep, ) = calc_basis_scores_scalpels(rvs_centered, ccfs, σ_rvs=σ_rvs, num_basis=num_basis, assume_centered=true, sort_by_responce=sort_by_responce)
    Δrv_shape = U_keep*U_keep'*rvs_centered
    rvs_clean = rvs .- Δrv_shape
    return rvs_clean
end

"""   `rms_clean_rvs_vs_num_basis_scalpels(rvs, ccfs; σ_rvs, max_num_basis )`
Compute RMS of estimated RVs after cleaning raw RVS with Scalpels algorithm (based on CCFs)

Inputs:
- rvs:  vector of estimated radial velocities
- ccfs: 2d array of CCFS of size (num_velocities, num_spectra)
Optional Inputs:
- σ_rvs:  vector of measurement uncertainties for estimated radial velocities (default: ones)
- max_num_basis:  maximum number of basis vectors to use for SVD reconstruction of CCFs
- sort_by_responce:  set true to sort basis vectors by RV responce (default: true)
Output:
rms_scalpels: vector of RMS estimated RVs after cleaning by scalpels as a function of the number of basis vectors

Notes:
- Currently Scalpels weights all observations equally and doesn't use the σ_rvs.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function rms_clean_rvs_vs_num_basis_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                  max_num_basis::Integer = min(length(rvs)-1,default_max_num_basis),
                  sort_by_responce::Bool = true ) where { T1<:Real, T2<:Real, T3<:Real }
    rms_scalpels = map(b->std(clean_rvs_scalpels(rvs,ccfs,num_basis=b, sort_by_responce=sort_by_responce)), 0:max_num_basis)
end


"""   `calc_clean_rvs_scores_basis_scalpels(rvs, ccfs; σ_rvs, num_basis, sort_by_responce )`
Computes cleaned rvs as well as the CCF basis functions and scores for a Scalpels reconstruction of CCFs
Inputs:
- rvs:  vector of estimated radial velocities
- ccfs: 2d array of CCFS of size (num_velocities, num_spectra)
Optional Inputs:
- σ_rvs:  vector of measurement uncertainties for estimated radial velocities (default: ones)
- num_basis:  number of basis vectors to use for SVD reconstruction of CCFs
- sort_by_responce:  set true to sort basis vectors by RV responce (default: true)
Output:
(rvs, scores, basis): a NamedTuple

Notes:
- Currently Scalpels weights all velocity pixels equally and uses the σ_rvs to weight each observation.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function calc_clean_rvs_scores_basis_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                num_basis::Integer = 3,
                sort_by_responce::Bool = true ) where { T1<:Real, T2<:Real, T3<:Real }
    @assert length(rvs) == length(σ_rvs)
    @assert length(rvs) == size(ccfs,2)
    if num_basis == 0   return rvs   end
    @assert 0 <= num_basis < length(rvs)
    mean_rv = mean(rvs, weights(1.0 ./ σ_rvs.^2 ))
    rvs_centered = rvs .- mean_rv

    (U_keep, basis ) = calc_basis_scores_scalpels(rvs_centered, ccfs, σ_rvs=σ_rvs, num_basis=num_basis, assume_centered=true, sort_by_responce=sort_by_responce)
    Δrv_shape = U_keep*U_keep'*rvs_centered
    rvs_clean = rvs .- Δrv_shape

    return (rvs=rvs_clean, scores=U_keep, basis=basis )
end


"""   `rms_clean_rvs_with_planets_vs_num_basis_scalpels(bjd, rvs, ccfs, periods; ...)`
Jointly fit sinusoidal planet RV amplitudes and SCALPELS activity basis vectors,
returning diagnostics as a function of the number of SCALPELS feature vectors used.

For each value of k from `min_num_basis` to `max_num_basis`:
  1. Project RVs and the sinusoidal design matrix into the subspace orthogonal
     to the k leading SCALPELS ACF basis vectors.
  2. Solve a weighted least-squares problem for the planet amplitudes in that
     projected space.
  3. Record the RMS, χ², AIC, and planet semi-amplitudes of the residuals.

This mirrors the approach of `vscalpels_recover_loocv` but uses the standard
SVD-based SCALPELS basis instead of the LOOCV basis.

Inputs:
- bjd:     vector of observation times (days), length num_obs
- rvs:     vector of estimated radial velocities, length num_obs
- ccfs:    2d array of CCFs of size (num_velocities, num_obs)
- periods: vector of trial orbital periods (days), length nplanets

Optional Inputs:
- σ_rvs:          measurement uncertainties for RVs (default: ones)
- min_num_basis:  minimum number of SCALPELS basis vectors (default: 0)
- max_num_basis:  maximum number of SCALPELS basis vectors
                  (default: min(num_obs-1, default_max_num_basis))
- resort:         if true, sort SVD basis vectors by |α| = |U'*rvs_centered|
                  before sweeping over k (default: false)
- jitter:         additional jitter (same units as rvs) added in quadrature to
                  σ_rvs when computing WLS weights and χ² (default: 0.0)
- weighted_mean:  if true, subtract the inverse-variance weighted mean ACF;
                  if false, subtract the unweighted mean ACF (default: true)

Output:
NamedTuple with fields:
- num_basis: UnitRange min_num_basis:max_num_basis
- rms:       vector of RMS of residuals at each k
- chi2:      vector of χ² of residuals at each k
- aic:       vector of AIC = χ² + 2*(k + 2*nplanets) at each k
- amp:       matrix of planet velocity semi-amplitudes, size (nk × nplanets)
- amperr:    matrix of uncertainties on amp, size (nk × nplanets)
"""
function rms_clean_rvs_with_planets_vs_num_basis_scalpels(
        bjd::AbstractVector{<:Real},
        rvs::AbstractVector{<:Real},
        ccfs::AbstractMatrix{<:Real},
        periods::AbstractVector{<:Real};
        σ_rvs::AbstractVector = ones(length(rvs)),
        min_num_basis::Integer = 0,
        max_num_basis::Integer = min(length(rvs)-1, default_max_num_basis),
        resort::Bool = false,
        jitter::Real = 0.0,
        weighted_mean::Bool = true
    )

    nobs = length(rvs)
    nplanets = length(periods)
    @assert length(bjd) == nobs
    @assert size(ccfs, 2) == nobs
    @assert length(σ_rvs) == nobs
    @assert 0 <= min_num_basis <= max_num_basis < nobs

    # Center RVs using inverse-variance weights.
    acf_weights = 1.0 ./ σ_rvs.^2
    mean_rv = mean(rvs, weights(acf_weights))
    rv_centered = rvs .- mean_rv

    # Build sinusoidal design matrix: 2 columns per planet (cos φ, sin φ).
    mean_bjd = mean(bjd)
    fftrn = Matrix{Float64}(undef, nobs, 2 * nplanets)
    for p in 1:nplanets
        phi = (2π / periods[p]) .* (bjd .- mean_bjd)
        fftrn[:, 2*p-1] = cos.(phi)
        fftrn[:, 2*p]   = sin.(phi)
    end

    # Compute ACFs and mean-subtract.
    acfs = autocor(ccfs, 0:size(ccfs, 1)-1)
    acfs_minus_mean = if weighted_mean && !all(σ_rvs .== first(σ_rvs))
        acfs .- mean(acfs, weights(acf_weights), dims=2)
    else
        acfs .- mean(acfs, dims=2)
    end

    # SVD; optionally reorder columns by decreasing |α|.
    svd_acfs = svd(acfs_minus_mean')
    U = svd_acfs.U
    alpha = U' * rv_centered
    idx = resort ? sortperm(abs.(alpha), rev=true) : (1:size(U, 2))

    # Weights for WLS and χ² (include jitter in quadrature).
    wls_invar_vec = 1.0 ./ (σ_rvs.^2 .+ jitter^2)
    wls_invar = Diagonal(wls_invar_vec)

    # Allocate output arrays.
    nk = max_num_basis - min_num_basis + 1
    rms_list   = Vector{Float64}(undef, nk)
    chi2_list  = Vector{Float64}(undef, nk)
    aic_list   = Vector{Float64}(undef, nk)
    amp_mat    = Matrix{Float64}(undef, nk, nplanets)
    amperr_mat = Matrix{Float64}(undef, nk, nplanets)

    for (i, k) in enumerate(min_num_basis:max_num_basis)
        # Project RVs and design matrix into the orthogonal complement of the
        # k-dimensional activity subspace.
        if k == 0
            fperp = fftrn
            vperp = rv_centered
        else
            U_keep = view(U, :, idx[1:k])
            pperp  = I(nobs) - U_keep * U_keep'
            fperp  = pperp * fftrn
            vperp  = pperp * rv_centered
        end

        # Weighted least-squares for planet (cos, sin) amplitudes.
        # pinv handles singular amat (degenerate trial periods or projected
        # sinusoids absorbed by the activity basis).
        amat      = (fperp' * wls_invar) * fperp
        bvec      = (fperp' * wls_invar) * vperp
        amat_pinv = pinv(amat)
        theta     = amat_pinv * bvec
        theterr   = sqrt.(max.(0.0, diag(amat_pinv)))

        # Convert (cos, sin) coefficient pairs to velocity semi-amplitudes.
        for p in 1:nplanets
            icol = 2 * p - 1
            Kx, dKx = theta[icol],   theterr[icol]
            Ky, dKy = theta[icol+1], theterr[icol+1]
            K  = sqrt(Kx^2 + Ky^2)
            dK = K > 0 ? sqrt((Kx * dKx)^2 + (Ky * dKy)^2) / K : hypot(dKx, dKy)
            amp_mat[i, p]    = K
            amperr_mat[i, p] = dK
        end

        rvresid      = vperp .- fperp * theta
        chi2_list[i] = sum(abs2.(rvresid) .* wls_invar_vec)
        rms_list[i]  = std(rvresid)
        aic_list[i]  = chi2_list[i] + 2 * (k + 2 * nplanets)
    end

    num_basis = min_num_basis:max_num_basis
    return (; num_basis, rms=rms_list, chi2=chi2_list, aic=aic_list,
              amp=amp_mat, amperr=amperr_mat)
end
