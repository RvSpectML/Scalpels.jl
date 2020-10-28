using Scalpels
using LinearAlgebra
import Polynomials

function calc_vertex_of_quadratic_fit(x::AbstractVector{T1}, y::AbstractVector{T2}) where {T1<:Real, T2<:Real}
    pfit = Polynomials.fit(x, y, 2)
    @assert length(Polynomials.coeffs(pfit)) >= 3   # just in case fails to fit a quadratic
    c, b, a = Polynomials.coeffs(pfit)
    v_at_min_of_quadratic = -b/(2*a)
    return v_at_min_of_quadratic
end

function est_rv_from_test_ccf(v_grid::AbstractVector{T1}, ccf::AbstractArray{T2,1}) where {T1<:Real, T2<:Real}
    idx_at_min = findmin(ccf)[2]
    @assert 1 <= idx_at_min <= length(v_grid)
    Δ_idx_to_include_in_fit = floor(Int,length(v_grid)//8)
    idx_around_min = (idx_at_min-Δ_idx_to_include_in_fit):(idx_at_min+Δ_idx_to_include_in_fit)
    @assert 1 <= minimum(idx_around_min) < maximum(idx_around_min) <= length(v_grid)
    (min_ccf,max_ccf) = extrema(ccf)
    ys = log.((ccf./max_ccf))
    v = calc_vertex_of_quadratic_fit(v_grid,ys)
    return v
end

# Hermite functions (physicists)
H0(x) = exp(-x^2)
H1(x) = 2*x*exp(-x^2)
H2(x) = (4*x^2-2)*exp(-x^2)
H3(x) = x*(8*x^2-12)*exp(-x^2)
H4(x) = (16*x^4-48*x^2+12)*exp(-x^2)

function make_rotation_matrix(angle::Real; ndims::Integer = 2, x::Integer = 1, y::Integer =2 )
    @assert( 1<=x<=ndims)
    @assert( 1<=y<=ndims)
    @assert( x!=y)
    matrix = diagm(ones(ndims))
    matrix[x,x] = cos(angle)
    matrix[y,y] = cos(angle)
    matrix[y,x] = sin(angle)
    matrix[x,y] = -matrix[y,x]
    return matrix
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
    if σ_ccf > zero(σ_ccf)
        ccfs .*= one(σ_ccf) .+ σ_ccf .* randn(size(ccfs))
    end
    rvs_dirty = map(t->est_rv_from_test_ccf(v_grid,ccfs[:,t]),1:num_rvs)
    return (rvs=rvs_dirty, ccfs=ccfs)
end
