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
