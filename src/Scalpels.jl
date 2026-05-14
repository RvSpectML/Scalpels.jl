# src/Scalpels.jl
"""
    Scalpels

A Julia package for removing stellar activity signals from radial velocity (RV)
measurements using the SCALPELS method (Collier Cameron et al. 2021) and its
extensions.

SCALPELS exploits the shape information encoded in cross-correlation functions
(CCFs) to separate activity-driven RV variations from Keplerian signals. This
package extends the original Julia package by adding leave-one-out cross-validation (LOOCV)
based cleaning (`vscalpels_*` functions) and quality-control diagnostics.

# References
- Collier Cameron et al. (2021), MNRAS, 505, 1699

# Authors: Eric Ford, Andrew Collier Cameron, Claude
Date:   September 2020, April 2026

"""
module Scalpels

using Statistics, StatsBase
using LinearAlgebra

export clean_rvs_scalpels,
       calc_basis_scores_scalpels,
       rms_clean_rvs_vs_num_basis_scalpels,
       rms_clean_rvs_with_planets_vs_num_basis_scalpels,
       loocv,
       reorder_uloocv,
       vscalpels_loocv,
       vscalpels_recover_loocv,
       fit_planets_loocv,
       search_planets_loocv,
       mask_outliers,
       quality_control,
       make_period_list,
       svd_reconstruction,
       estimate_continuum

include("internals.jl")
include("scalpels_code.jl")
include("original.jl")
include("loocv.jl")
include("diagnostics.jl")
include("utils.jl")

include("Simulation/Simulation.jl")

end # module Scalpels
