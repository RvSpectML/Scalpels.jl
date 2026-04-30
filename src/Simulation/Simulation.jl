# src/Simulation/Simulation.jl
"""
    Scalpels.Simulation

Submodule providing tools for generating synthetic radial velocity and CCF
datasets, primarily for use in tests, examples, and method validation.

# Exported names
- [`RvSinusoid`](@ref)
- [`RvSinusoidSimple`](@ref)
- [`gen_rv_dataset`](@ref)
- [`gen_ccf`](@ref)
- [`gen_rv_ccf_dataset`](@ref)
- [`calc_ccf_derivs`](@ref)
"""
module Simulation

using Statistics, StatsBase
using LinearAlgebra, Polynomials
using Random
using DataFrames
using AbstractGPs, TemporalGPs, KernelFunctions

export RvSinusoid,
       RvSinusoidSimple,
       gen_rv_dataset,
       gen_ccf,
       gen_rv_ccf_dataset,
       calc_ccf_derivs

# Physical constants used in CCF generation.
const days_in_year = 365.2425
const c_mps        = 299_792_458.0

include("models.jl")
include("gasusshermite.jl")
include("data_generation.jl")

end # module Simulation
