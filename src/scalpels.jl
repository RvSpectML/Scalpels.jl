"""   `calc_basis_scores_scalpels(rvs, ccfs; σ_rvs, num_basis )`
Compute the CCF basis functions and scores for a Scalpels reconstruction of CCFs
Inputs:
- rvs:  vector of estimated radial velocities
- ccfs: 2d array of CCFS of size (num_velocities, num_spectra)
Optional Inputs:
- σ_rvs:  vector of measurement uncertainties for estimated radial velocities (default: ones)
- num_basis:  number of basis vectors to use for SVD reconstruction of CCFs
Output:
(basis, scores): a NamedTuple
basis: num_rvs*num_rvs

Notes:
- Currently Scalpels weights all velocity pixels equally and uses the σ_rvs to weight each observation.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function calc_basis_scores_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                num_basis::Integer = 3, assume_centered::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real }
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
    idx = sortperm(abs.(alpha),rev=true)
    #idx = 1:length(alpha)   # To throw away sorting by projection onto RVs
    U_keep = view(svd_acfs.U,:,idx[1:num_basis])
    return (basis=U_keep, scores = alpha[idx[1:num_basis]])

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
Output:
rvs_clean: vector of estimated RVs after cleaning by scalpels

Notes:
- Currently Scalpels weights all observations equally and doesn't use the σ_rvs.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function clean_rvs_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                num_basis::Integer = 3 ) where { T1<:Real, T2<:Real, T3<:Real }
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
    alpha = U'*rvs_centered
    idx = sortperm(abs.(alpha),rev=true)
    U_keep = view(U,:,idx[1:num_basis])
    =#

    (U_keep, scores ) = calc_basis_scores_scalpels(rvs_centered, ccfs, σ_rvs=σ_rvs, num_basis=num_basis, assume_centered=true)
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
Output:
rms_scalpels: vector of RMS estimated RVs after cleaning by scalpels as a function of the number of basis vectors

Notes:
- Currently Scalpels weights all observations equally and doesn't use the σ_rvs.
- First element of output starts with RMS with zero basis vectors (i.e., the input RVs)
"""
function rms_clean_rvs_vs_num_basis_scalpels(rvs::AbstractVector{T1}, ccfs::AbstractArray{T2,2}
                ; σ_rvs::AbstractVector{T3} = ones(length(rvs)),
                  max_num_basis::Integer = min(length(rvs)-1,default_max_num_basis) ) where { T1<:Real, T2<:Real, T3<:Real }
    rms_scalpels = map(b->std(clean_rvs_scalpels(rvs,ccfs,num_basis=b)), 0:max_num_basis)
end
