"""
Module for performing Scalpels (Self-Correlation Analysis of Line Profiles for Extracting Low-amplitude Shifts)
based on a CCF timeseries.

For algorithm information, see Collier-Cameron, Ford, Shahaf et al. 2020

Author: Eric Ford
Date:   September 2020
"""
module Scalpels

using Statistics, StatsBase
using LinearAlgebra

default_max_num_basis = 16
default_num_basis = 4

include("scalpels_code.jl")
export clean_rvs_scalpels, calc_clean_rvs_scores_basis_scalpels, rms_clean_rvs_vs_num_basis_scalpels

end
