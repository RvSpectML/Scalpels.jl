# src/Simulation/models.jl
# Parametric RV models for simulation.

"""
    RvSinusoid{T<:Real}

A sinusoidal RV model parameterised by (A, B, C) coefficients, where the
RV at time `t` is:

    rv(t) = A·sin(2πt/P) + B·cos(2πt/P) + C

This parameterisation is linear in (A, B, C) and is convenient for fitting.

# Fields
- `P`: Orbital period (days).
- `A`: Sine coefficient (m/s).
- `B`: Cosine coefficient (m/s).
- `C`: RV offset (m/s).

# Constructor
    RvSinusoid(; P, K, t0, C)

Construct from physical parameters: period `P` (days), semi-amplitude `K`
(m/s), time of conjunction `t0` (days), and offset `C` (m/s).

# Example
```julia
m = RvSinusoid(P=3.0, K=5.0, t0=0.0, C=0.0)
m(1.5)  # RV at t=1.5 days
""" 

struct RvSinusoid{T<:Real} 
    P::T 
    A::T 
    B::T 
    C::T
end

function RvSinusoid(; P::Real, K::Real, t0::Real, C::Real) 
    RvSinusoid(P, K*cos(-t0*2π/P), K*sin(-t0*2π/P), C) 
end

function (m::RvSinusoid)(t)
     s, c = sincos(t * 2π / m.P)
    m.A * s + m.B * c + m.C
end

""" RvSinusoidSimple{T<:Real}

A sinusoidal RV model parameterised directly by period, semi-amplitude, time of conjunction, and offset:
rv(t) = K·sin(2π(t - t0)/P) + C
## Fields
- P: Orbital period (days).
- K: RV semi-amplitude (m/s).
- t0: Time of conjunction (days).
- C: RV offset (m/s).

## Example
```julia
m = RvSinusoidSimple(3.0, 5.0, 0.0, 0.0)
m(1.5)  # RV at t=1.5 days
```
# !!! note RvSinusoid and RvSinusoidSimple produce identical RVs for equivalent parameters. RvSinusoid is preferable when fitting, as its parameterisation is linear. 
""" 
struct RvSinusoidSimple{T<:Real} 
    P::T 
    K::T 
    t0::T 
    C::T
end

function (m::RvSinusoidSimple)(t) 
    m.K * sin((t - m.t0) * 2π / m.P) + m.C 
end
