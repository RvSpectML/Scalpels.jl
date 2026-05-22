### A Pluto.jl notebook ###
# v0.20.10

using Markdown
using InteractiveUtils

# ╔═╡ 5eee68ba-3721-42e0-bc3e-79f032bea924
if true # true when developing Scalpels locally.  False when using registered version
import Pkg;
Pkg.activate(joinpath(@__DIR__,".."))
end

# ╔═╡ 83c89fd0-3cfa-11f1-3984-2f15dba17906
begin
using Revise
    using Scalpels
    using Scalpels.Simulation
using LinearAlgebra
    using Statistics, StatsBase
    using DataFrames, InlineStrings
using CSV, FITSIO
using Glob
    using Plots, ColorSchemes
end

# ╔═╡ a1000001-3cfa-11f1-0000-000000000001
md"""
# vSCALPELS on Real ESSP Data (DS1)

This notebook applies the vSCALPELS method to the ESSP DS1 dataset — real
CCF observations from four instruments (HARPS-N, EXPRES, NEID, HARPS).

Unlike the synthetic demo, RVs here are computed from order-by-order (OBO)
CCF data rather than provided directly.

## Contents
1. Configuration
2. Helper functions: reading and combining OBO CCF data
3. Load and combine data
4. Instrument offset correction
5. Inspect the data
6. Quality control and outlier detection
7. vSCALPELS cleaning (LOOCV)
8. Period search
"""

# ╔═╡ a1000002-3cfa-11f1-0000-000000000002
md"## 1. Configuration"

# ╔═╡ 5e41dfae-c359-4f3f-ab3e-90e0ba799f81
begin
dataset_num = 2
inst_process = "all"
end;

# ╔═╡ a1000003-3cfa-11f1-0000-000000000003
begin
    data_dir = joinpath(@__DIR__, "..", "data", "DS" * string(dataset_num), "CCFs")
    csv_path = joinpath(@__DIR__, "..", "data", "DS" * string(dataset_num), "DS" * string(dataset_num) *"_timeSeries.csv")
end;

# ╔═╡ d5a630d5-e1d7-4988-8825-bfb66dba432c
# Find CCF files present locally and match to CSV metadata.
if inst_process == "all"
ccf_files = sort(glob("DS*_ccfs_*.fits", data_dir))
else
ccf_files = sort(glob("DS*_ccfs_" * inst_process * ".fits", data_dir))
end;

# ╔═╡ a1000004-3cfa-11f1-0000-000000000004
md"""
## 2. Helper Functions

Each FITS file contains one observation with:
- `V_GRID`: velocity grid in km/s, shape `(Nv,)`
- `OBO_CCF` / `OBO_E_CCF`: order-by-order CCFs and uncertainties, shape `(Nv, No)` in Julia
- `OBO_CCF_RV` / `OBO_CCF_E_RV`: per-order RV and uncertainty, shape `(No,)`

NaN values mark masked or unavailable orders.
"""

# ╔═╡ cc1a480d-33bb-4ce0-b9c7-f38a683a102f
"""Read one FITS CCF file into a NamedTuple of OBO arrays."""
    function read_ccf_file(path::AbstractString)
        FITS(path) do f
#@info f
            v_grid    = read(f["V_GRID"])
orders    = read(f["ECHELLE_ORDERS"])
            obo_ccf   = read(f["OBO_CCF"])
            obo_e_ccf = read(f["OBO_E_CCF"])
            obo_rv    = Float64.(read(f["OBO_RV"]))
            obo_e_rv  = Float64.(read(f["OBO_E_RV"]))
            (; v_grid, obo_ccf, obo_e_ccf, obo_rv, obo_e_rv, orders)
        end
    end

# ╔═╡ b27aaf97-ea7c-46db-a4d9-7b2575af721b
"""
    Collapse OBO arrays to a single combined CCF vector.

    obo_ccf is (Nv, No) in Julia (FITS NAXIS1 = velocity, NAXIS2 = orders).
    Weights: NaN or missing entries are excluded for both ccf and input weights.
    """
    function combine_orders_ccf(ccf::AbstractMatrix, weights::AbstractVector)
        # CCF combination (Nv, No) → (Nv,) via weighted sum over orders (dim 2)
        bad_weight   =  ismissing.(weights) .| isnan.(weights)
bad_ccf =  isnan.(ccf) .| ismissing.(ccf) #.| .!(e_ccf .> 0)
        w_sum     = sum(ifelse.(bad_weight, 0.0, weights))
#@info size(ccf), size(weights) #, size(w_sum)
#@info sum(bad_ccf), sum(bad_weight)
        ccf_combined = vec(sum(ifelse.(bad_ccf,0.0,ccf) * ifelse.(bad_weight, 0.0, weights), dims=2)) / w_sum
#        ccf_combined = vec(sum(ccf * ifelse.(bad_weight, 0.0, weights), dims=2)) / w_sum
#@info size(ccf)
(;ccf_combined, weights)
end


# ╔═╡ 5f9e343f-8881-4f1c-a077-da1af6a7cbb1
"""
    Collapse OBO arrays to a single RV, and σ_rv.

    obo_ccf is (Nv, No) in Julia (FITS NAXIS1 = velocity, NAXIS2 = orders).
    Weights = 1/σ²; NaN or non-positive-error entries are excluded.
    """
    function combine_orders_rv(rv::AbstractVector, σrv::AbstractVector, weights::AbstractVector)
        bad_rv  = isnan.(weights) .| ismissing(weights) .| isnan.(rv) .| ismissing(rv)
w_sum     = sum(ifelse.(bad_rv, 0.0, weights))
        rv      = ( ifelse.(bad_rv, 0.0, rv)' * ifelse.(bad_rv, 0.0, weights) ) / w_sum
        σrv     = sqrt.(1.0 ./ (ifelse.(bad_rv, 0.0, 1.0 ./σrv.^2)' * ifelse.(bad_rv, 0.0, weights) ./ w_sum) )
        (; rv, σrv)
    end

# ╔═╡ a1000006-3cfa-11f1-0000-000000000006
md"## 3. Load and Combine Data"

# ╔═╡ ed40229b-8516-4d6c-b767-b7f318660368
# Load CSV time series for timestamps and instrument labels.
# Spec filenames (DS1.NNN_spec_INST.fits) → CCF stems (DS1.NNN)
function load_time_series(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)
    df.stem = map(fn -> match(r"DS\d\.\d+", fn).match,
                  df[!, "Standard File Name"])
    return df
end

# ╔═╡ 692094b7-0e37-41e2-95f0-80070757c0f9
df_ts = load_time_series(csv_path)

# ╔═╡ a1000008-3cfa-11f1-0000-000000000008
# Build a lookup: stem → (time, instrument)
stem_to_meta = Dict(
    row.stem => (t = row["Time [eMJD]"], inst = row.Instrument)
    for row in eachrow(df_ts)
);

# ╔═╡ c5e1f249-0a41-461a-b65e-43065e1a3a04
order_weights = CSV.read(joinpath(data_dir,"../..","order_weights.csv"),DataFrame);

# ╔═╡ 5ee08341-cb92-440d-80e3-910376b0d146
let
plt = plot()
for i in 2:length(names(order_weights))
scatter!(plt,order_weights.echelle, order_weights[!,i], label=names(order_weights)[i], lc=i)
end
plt
end


# ╔═╡ 65991dc3-c976-45be-bcd7-2ea2200f3df8
# ╠═╡ disabled = true
#=╠═╡
let
data = read_ccf_file(first(ccf_files))
weight_idx = map(i->searchsortedfirst(order_weights[!,"echelle"],data.orders[i], rev=true), 1:length(data.orders))
end
  ╠═╡ =#

# ╔═╡ d7d71d04-fbdb-4575-b420-9e2b5abf7641
max_obs_to_use = 400

# ╔═╡ 1901314d-cc55-40e5-b5a1-ec60e1cb64bb
# Load and combine each file. Build parallel arrays.
function load_obs_data(ccf_files, stem_to_meta, weights)
    rows = NamedTuple[]
    for path in ccf_files[1:min(max_obs_to_use,length(ccf_files))]
        base = splitext(basename(path))[1]
        stem = match(r"DS\d\.\d+", base).match
        meta = get(stem_to_meta, stem, nothing)
        meta === nothing && continue

        d   = read_ccf_file(path)

# Perform continuum normalization
nan_mask = isnan.(d.obo_ccf)
d.obo_ccf[nan_mask] .= 0.0
ccf_norm = estimate_continuum(d.v_grid,d.obo_ccf; quantile_level=0.9,
        line_half_width= 7.0,v_center= 0.0)
d.obo_ccf[nan_mask] .= NaN
d.obo_ccf ./= ccf_norm'
d.obo_e_ccf ./= ccf_norm'

#@info d.obo_ccf
# Computed weighted CCF
weight_idx = map(i->searchsortedfirst(order_weights[!,"echelle"],d.orders[i], rev=true), 1:length(d.orders))
weights = order_weights[weight_idx,meta.inst]
weights[any(isnan.(d.obo_ccf),dims=1)'] .= 0.0
#@info weights
(ccf_combo, weights2) = combine_orders_ccf(d.obo_ccf,weights )
(rv_combo, σrv_combo) = combine_orders_rv(d.obo_rv, d.obo_e_rv, weights2)
        push!(rows, (;
            stem         = stem,
            instrument   = meta.inst,
            t            = meta.t,
            rv           = rv_combo,
            σrv          = σrv_combo,
            ccf_combined = ccf_combo,
            v_grid       = d.v_grid,
        ))
end
    return rows
end

# ╔═╡ f5d7d732-f2df-4b05-9efd-c502ce2dfed7
obs_data = load_obs_data(ccf_files, stem_to_meta, order_weights);

# ╔═╡ a1000011-3cfa-11f1-0000-000000000011
# Assemble into matrices / vectors sorted by time.
begin
    perm       = sortperm([r.t for r in obs_data])
    t          = [obs_data[i].t          for i in perm]
    rvs        = [obs_data[i].rv         for i in perm]
    σrvs       = [obs_data[i].σrv        for i in perm]
    inst_labels = [obs_data[i].instrument for i in perm]
    v_grid     = obs_data[1].v_grid

    # CCF matrix: (Nv, Nobs)
    #ccfs = hcat([obs_data[i].ccf_combined for i in perm]...)
ccfs = stack(i->obs_data[i].ccf_combined, perm)
    num_obs    = length(t)
    num_vel    = length(v_grid)
    num_obs, num_vel
end

# ╔═╡ f99be7e2-5b21-4602-804e-0b92516e69d9
rvs

# ╔═╡ f1404c52-b5b6-4c58-af72-b4e274849c20
σrvs # TODO figure out why high

# ╔═╡ a1000012-3cfa-11f1-0000-000000000012
md"""
Loaded **$(num_obs)** observations over **$(round(last(t) - first(t), digits=1))** days
from instruments: $(join(unique(inst_labels), ", ")).
"""

# ╔═╡ a1000013-3cfa-11f1-0000-000000000013
md"""
## 4. Instrument Offset Correction

The four ESSP instruments have a mutual velocity offset of ~650 m/s.
We correct by subtracting each instrument's weighted-mean RV and re-centering
the full dataset to zero mean.
"""

# ╔═╡ 8145427c-8955-4614-b46a-56408e0c39d9
begin
instrument_offset = Dict{AbstractString,Float64}()
instrument_offset["neid"] = -79.72803738588587
instrument_offset["expres"] = -88.58011579897897
instrument_offset["harps"] = 566.2583461658813
instrument_offset["harpsn"] = 460.05051489822205
instrument_offset
end;

# ╔═╡ 19bc1a71-168c-4eb4-8073-5f4c7dd7f0e0
function correct_instrument_offsets(rvs, σrvs, inst_labels, offsets)
    rv    = copy(rvs)
    invar = 1.0 ./ σrvs.^2
    global_mean = 0.0 # sum(rv .* invar) / sum(invar)
    for inst in unique(inst_labels)
        mask      = inst_labels .== inst
        #inst_mean = sum(rv[mask] .* invar[mask]) / sum(invar[mask])
        inst_mean = offsets[inst]
rv[mask] .-= inst_mean - global_mean
    end
    return rv
end

# ╔═╡ b4cb977b-0214-4ad2-a99a-f4e60f44f8dc
rvs_corr = correct_instrument_offsets(rvs, σrvs, inst_labels, instrument_offset);

# ╔═╡ a1000015-3cfa-11f1-0000-000000000015
md"## 5. Inspect the Data"

# ╔═╡ a1000016-3cfa-11f1-0000-000000000016
# Per-instrument summary
let
    insts = unique(inst_labels)
    DataFrame(
        instrument = insts,
        n_obs      = [count(==(i), inst_labels) for i in insts],
        median_σrv = [round(median(σrvs[inst_labels .== i]), digits=3) for i in insts],
    )
end

# ╔═╡ a1000017-3cfa-11f1-0000-000000000017
# CCF heatmap
heatmap(
    v_grid,
    1:num_obs,
    ccfs',
    xlabel = "Velocity (km/s)",
    ylabel = "Observation (time-sorted)",
    title  = "CCF matrix — DS1",
    color  = :viridis,
)

# ╔═╡ a1000018-3cfa-11f1-0000-000000000018
# RVs vs time, coloured by instrument
let
    palette_map = Dict(u => i for (i, u) in enumerate(unique(inst_labels)))
    plt = plot(title="Offset-corrected RVs — DS1",
               xlabel="Time (eMJD)", ylabel="RV (m/s)", legend=:topright)
    for inst in unique(inst_labels)
        mask = inst_labels .== inst
        scatter!(plt, t[mask], rvs_corr[mask],
                 label=inst, #yerr=σrvs[mask],
ms=3, mc=palette_map[inst])
    end
    plt
end

# ╔═╡ a1000019-3cfa-11f1-0000-000000000019
md"""
## 6. Quality Control and Outlier Detection

`quality_control` flags observations that deviate by more than `threshold` MADs
from the median in the SVD coefficient space, and estimates the optimal number
of LOOCV shape vectors by comparing the LOOCV basis to the plain SVD basis.
"""

# ╔═╡ a1000020-3cfa-11f1-0000-000000000020
qc = quality_control(ccfs, rvs_corr, σrvs; max_scalpels_vectors=20, threshold=7);

# ╔═╡ a1000021-3cfa-11f1-0000-000000000021
md"""
- Observations passing QC: **$(sum(qc.obs_mask))** / $(num_obs)
- Estimated optimal number of shape vectors: **$(qc.kopt)**
"""

# ╔═╡ a1000022-3cfa-11f1-0000-000000000022
scatter(
    qc.madratio,
    xlabel = "Shape vector index",
    ylabel = "MAD ratio (LOOCV vs. SVD)",
    title  = "QC: LOOCV–SVD divergence",
    legend = false,
)

# ╔═╡ 6e5dba88-bafb-4369-bff5-5c3a85356066
let
plts = []
for inst in unique(inst_labels)
        mask      = inst_labels .== inst

plt = heatmap(v_grid, 1:sum(qc.obs_mask .& mask),
view(ccfs,:,qc.obs_mask .& mask)'.- mean(view(ccfs,:,qc.obs_mask .& mask)',dims=1),
color  = :viridis, title=inst)
push!(plts,plt)
end
plot(plts...,layout=(2,2))
end

# ╔═╡ a1000023-3cfa-11f1-0000-000000000023
# Flag outliers on the RV plot
let
    plt = plot(title="RVs with QC flags",
               xlabel="Time (eMJD)", ylabel="RV (m/s)")
    scatter!(plt, t[qc.obs_mask],  rvs_corr[qc.obs_mask],
             label="Good", ms=3)
    scatter!(plt, t[.!qc.obs_mask], rvs_corr[.!qc.obs_mask],
             label="Flagged", ms=5, marker=:x)
    plt
end

# ╔═╡ a1000024-3cfa-11f1-0000-000000000024
md"""
## 7. vSCALPELS Cleaning (LOOCV)

Apply vSCALPELS to the quality-controlled dataset.  The number of shape
vectors is determined by `qc.kopt` (minimum 1).
"""

# ╔═╡ a1000025-3cfa-11f1-0000-000000000025
begin
keep_all = sum(qc.obs_mask) == length(qc.obs_mask)
    good       = keep_all ? (1:length(qc.obs_mask)) : qc.obs_mask
    ccfs_good  = keep_all ? ccfs : view(ccfs,      :, good)
    rvs_good   = keep_all ? rvs_corr : view(rvs_corr,good)
    σrvs_good  = keep_all ? σrvs : view(σrvs,good)
    t_good     = keep_all ? t : view(t,good)
    inst_good  = keep_all ? inst_labels : view(inst_labels,good)
    kvecs      = 6 # max(1, qc.kopt)
end;

# ╔═╡ a1000026-3cfa-11f1-0000-000000000026
out_loocv = vscalpels_loocv(ccfs_good, rvs_good, σrvs_good;
                             max_scalpels_vectors = kvecs, weighted_mean=false, resort=false);

# ╔═╡ c9d13fc6-30ee-42c5-bb4d-ad3a1a118ebc
out_loocv

# ╔═╡ a1000027-3cfa-11f1-0000-000000000027
md"""
**vSCALPELS result**
- Shape vectors selected: $(size(out_loocv.u_loocv, 2))
- RMS before cleaning: $(round(std(rvs_good .- mean(rvs_good)), digits=3)) m/s
- RMS after cleaning:  $(round(std(out_loocv.rv_clean), digits=3)) m/s
"""

# ╔═╡ a1000028-3cfa-11f1-0000-000000000028
let
    plt = plot(title="vSCALPELS cleaning — DS" * string(dataset_num),
               xlabel="Time (eMJD)", ylabel="RV (m/s)", legend=:topright)
    for inst in unique(inst_good)
        mask = inst_good .== inst
        scatter!(plt, t_good[mask], (rvs_good .- mean(rvs_good))[mask],
                 label="$(inst) raw", ms=3, alpha=0.5)
    end
    scatter!(plt, t_good, out_loocv.rv_clean,
             label="Cleaned", ms=3, mc=:red, alpha=0.5)
    plt
end

# ╔═╡ a1000029-3cfa-11f1-0000-000000000029
md"""
## 8. Period Search

Search for periodic signals over a grid of trial periods spanning 2–200 days,
using `vscalpels_recover_loocv` at each trial period.
"""

# ╔═╡ d4b6ccb7-060f-4bc9-bd3d-207b13f0e3b3
run_period_search = true

# ╔═╡ a1000030-3cfa-11f1-0000-000000000030
period_list = make_period_list(t_good, 2.0, 200.0, oversample_factor=2);

# ╔═╡ b3434d0e-e573-44e4-996a-6dc5f69121e5


# ╔═╡ a1000031-3cfa-11f1-0000-000000000031
if run_period_search
out_search = map(
    P -> vscalpels_recover_loocv(
        t_good, rvs_good, σrvs_good, ccfs_good, [P];
        max_scalpels_vectors = kvecs
    ),
    period_list
);
end

# ╔═╡ a1000032-3cfa-11f1-0000-000000000032
if run_period_search
    amps = [o.amp[1]         for o in out_search]
    lls  = [-0.5 * o.χ²      for o in out_search]
    rms  = [std(o.rvresid)   for o in out_search]
    lp   = log10.(period_list)

    plt1 = plot(lp, amps, xlabel="log₁₀(P / days)", ylabel="K (m/s)",
                title="Recovered amplitude", legend=false)
    plt2 = plot(lp, lls,  xlabel="log₁₀(P / days)", ylabel="-χ²/2",
                title="Log-likelihood", legend=false)
    plt3 = plot(lp, rms,  xlabel="log₁₀(P / days)", ylabel="RMS (m/s)",
                title="RMS of residuals", legend=false)
    plot(plt1, plt2, plt3, layout=(3, 1), size=(700, 700))
end

# ╔═╡ b1000001-3cfa-11f1-0000-000000000001
md"""
## 9. Per-instrument Analysis

We apply vSCALPELS to each instrument's data independently, then compare:

1. **MAD ratios** from `quality_control` per instrument — indicates how many
   LOOCV vectors are well-determined by each instrument's data.
2. **Leading basis vectors** (SVD-ordered) — the velocity-space shape patterns
   captured by each instrument. Agreement across instruments validates the
   joint analysis.
"""

# ╔═╡ b1000002-3cfa-11f1-0000-000000000002
# Number of basis vectors to extract per instrument for comparison.
n_compare = min(12,kvecs);

# ╔═╡ 382fabed-38d4-4f32-a71f-d481e3179764
# Per-instrument: QC, basis vectors (SVD-ordered), and vSCALPELS.
function per_instrument_analysis(inst_labels, ccfs, rvs_corr, σrvs, t;
                                  n_compare::Integer = 4)
    results = Dict{String, NamedTuple}()
    for inst in sort(unique(inst_labels))
        mask   = inst_labels .== inst
        ccfs_i = ccfs[:, mask]
        rvs_i  = rvs_corr[mask]
        σrvs_i = σrvs[mask]
        t_i    = t[mask]
        nobs_i = sum(mask)

        kmax_i = min(n_compare, nobs_i - 2)
        kmax_i < 1 && continue

        qc_i = quality_control(ccfs_i, rvs_i, σrvs_i;
                                max_scalpels_vectors = min(10, kmax_i),
                                threshold = 7)

        invar_i        = 1.0 ./ σrvs_i.^2
        rvs_i_centered = rvs_i .- sum(rvs_i .* invar_i) / sum(invar_i)
        basis_i = calc_basis_scores_scalpels(
            rvs_i_centered, ccfs_i;
            σ_rvs = σrvs_i, num_basis = kmax_i, sort_by_responce = false,
            assume_centered = true,
        ).basis

        good_i  = qc_i.obs_mask
good_i   = 1:length(rvs_i)
        kv_i    = max(1, min(qc_i.kopt, sum(good_i) - 2))
        loocv_i = vscalpels_loocv(
            ccfs_i[:, good_i], rvs_i[good_i], σrvs_i[good_i];
            max_scalpels_vectors = kv_i,
        )

        results[inst] = (;
            qc         = qc_i,
            basis      = basis_i,
            loocv      = loocv_i,
            rvs        = rvs_i,
            σrvs       = σrvs_i,
            t          = t_i,
            good       = good_i,
            nobs       = nobs_i,
            kmax       = kmax_i,
            rms_before = std(rvs_i[good_i] .- mean(rvs_i[good_i])),
            rms_after  = std(loocv_i.rv_clean),
        )
    end
    return results
end

# ╔═╡ 6c12f943-c01c-4693-847b-d1f452df744d
inst_results = per_instrument_analysis(inst_labels, ccfs, rvs_corr, σrvs, t;
                                        n_compare);

# ╔═╡ b1000004-3cfa-11f1-0000-000000000004
md"""
### 9a. MAD Ratio Comparison

The MAD ratio measures how much the LOOCV basis vectors diverge from the
plain SVD vectors as more vectors are added. A rising ratio signals
overfitting; the optimal cut is where the ratio first rises sharply.

Comparing curves across instruments (and to the joint curve) helps choose
a robust `max_scalpels_vectors` for the joint analysis.
Dashed vertical line marks joint `qc.kopt`.
"""

# ╔═╡ b1000005-3cfa-11f1-0000-000000000005
let
    plt = plot(
        xlabel = "Shape vector index",
        ylabel = "MAD ratio (LOOCV vs. SVD)",
        title  = "Per-instrument QC: LOOCV–SVD divergence",
        legend = :topleft,
    )
    for inst in sort(collect(keys(inst_results)))
        mr = inst_results[inst].qc.madratio
        plot!(plt, 1:length(mr), mr; label=inst, lw=2, marker=:circle, ms=3)
    end
    plot!(plt, 1:length(qc.madratio), qc.madratio;
          label="Joint", lw=2, ls=:dash, color=:black, marker=:diamond, ms=3)
    vline!(plt, [qc.kopt]; lw=1, ls=:dot, color=:black,
           label="joint kopt = $(qc.kopt)")
    plt
end

# ╔═╡ b1000006-3cfa-11f1-0000-000000000006
md"""
### 9b. Leading Basis Vectors (Velocity-space Shapes)

The rows of the `basis` matrix are the dominant CCF shape patterns in
velocity space, sorted by singular value for a consistent cross-instrument
comparison. Signs are aligned to the corresponding joint basis vector.

Instruments that recover similar shapes support a joint analysis; large
disagreements may indicate instrument-specific systematics.
"""

# ╔═╡ b1000007-3cfa-11f1-0000-000000000007
# Joint basis vectors (SVD-ordered) used as sign-alignment reference.
joint_basis = let
    invar_g = 1.0 ./ σrvs_good.^2
    rv_cent = rvs_good .- sum(rvs_good .* invar_g) / sum(invar_g)
    calc_basis_scores_scalpels(
        rv_cent, ccfs_good;
        σ_rvs = σrvs_good, num_basis = n_compare,
        sort_by_responce = false, assume_centered = true,
    ).basis   # (n_compare, Nv)
end;

# ╔═╡ b1000008-3cfa-11f1-0000-000000000008
let
    insts  = sort(collect(keys(inst_results)))
    n_cols = length(insts) + 1   # one column per instrument + joint
    n_rows = n_compare

    plts = []
    for k in 1:n_rows
        for (j, inst) in enumerate(insts)
            res = inst_results[inst]
            if k > res.kmax
                push!(plts, plot(framestyle=:none,
                                 title=(k == 1 ? inst : "")))
                continue
            end
            bvec = copy(res.basis[k, :])
            bvec .*= sign(dot(bvec, joint_basis[k, :]))   # align sign
            p = plot(v_grid, bvec;
                     legend = false, lw = 1.5,
                     title  = (k == 1 ? inst : ""),
                     ylabel = (j == 1 ? "Basis $k" : ""),
                     xlabel = (k == n_rows ? "v (km/s)" : ""),
                     xticks = (k == n_rows ? :auto : :none))
            hline!(p, [0.0]; lw=0.5, color=:gray)
            push!(plts, p)
        end
        # Joint column
        p = plot(v_grid, joint_basis[k, :];
                 legend = false, lw = 1.5, color = :black,
                 title  = (k == 1 ? "Joint" : ""),
                 ylabel = "",
                 xlabel = (k == n_rows ? "v (km/s)" : ""),
                 xticks = (k == n_rows ? :auto : :none))
        hline!(p, [0.0]; lw=0.5, color=:gray)
        push!(plts, p)
    end

    plot(plts...; layout=(n_rows, n_cols),
         size=(200 * n_cols, 160 * n_rows))
end

# ╔═╡ b1000009-3cfa-11f1-0000-000000000009
md"### 9c. Per-instrument vSCALPELS Summary"

# ╔═╡ b1000010-3cfa-11f1-0000-000000000010
let
    insts = sort(collect(keys(inst_results)))
    DataFrame(
        instrument = insts,
        n_obs      = [inst_results[i].nobs                        for i in insts],
        n_good     = [sum(inst_results[i].good)                   for i in insts],
        kopt       = [inst_results[i].qc.kopt                     for i in insts],
        kvecs_used = [size(inst_results[i].loocv.u_loocv, 2)      for i in insts],
        rms_before = [round(inst_results[i].rms_before, digits=3) for i in insts],
        rms_after  = [round(inst_results[i].rms_after,  digits=3) for i in insts],
    )
end

# ╔═╡ b1000011-3cfa-11f1-0000-000000000011
let
    insts = sort(collect(keys(inst_results)))
    plts  = []
    for inst in insts
        res      = inst_results[inst]
        good     = res.good
        t_i      = res.t[good]
        rv_raw   = res.rvs[good] .- mean(res.rvs[good])
        rv_clean = res.loocv.rv_clean
        p = scatter(t_i, rv_raw;   label="raw",     ms=3, alpha=0.6,
                    title=inst, ylabel="RV (m/s)")
        scatter!(p, t_i, rv_clean; label="cleaned", ms=3)
        push!(plts, p)
    end
    plot(plts...; layout=(length(insts), 1),
         xlabel="Time (eMJD)", size=(700, 250 * length(insts)))
end

# ╔═╡ Cell order:
# ╠═5eee68ba-3721-42e0-bc3e-79f032bea924
# ╠═83c89fd0-3cfa-11f1-3984-2f15dba17906
# ╟─a1000001-3cfa-11f1-0000-000000000001
# ╟─a1000002-3cfa-11f1-0000-000000000002
# ╠═5e41dfae-c359-4f3f-ab3e-90e0ba799f81
# ╠═a1000003-3cfa-11f1-0000-000000000003
# ╠═d5a630d5-e1d7-4988-8825-bfb66dba432c
# ╟─a1000004-3cfa-11f1-0000-000000000004
# ╟─cc1a480d-33bb-4ce0-b9c7-f38a683a102f
# ╠═b27aaf97-ea7c-46db-a4d9-7b2575af721b
# ╠═5f9e343f-8881-4f1c-a077-da1af6a7cbb1
# ╟─a1000006-3cfa-11f1-0000-000000000006
# ╟─ed40229b-8516-4d6c-b767-b7f318660368
# ╠═692094b7-0e37-41e2-95f0-80070757c0f9
# ╠═a1000008-3cfa-11f1-0000-000000000008
# ╠═c5e1f249-0a41-461a-b65e-43065e1a3a04
# ╟─5ee08341-cb92-440d-80e3-910376b0d146
# ╠═65991dc3-c976-45be-bcd7-2ea2200f3df8
# ╠═d7d71d04-fbdb-4575-b420-9e2b5abf7641
# ╠═1901314d-cc55-40e5-b5a1-ec60e1cb64bb
# ╠═f5d7d732-f2df-4b05-9efd-c502ce2dfed7
# ╠═a1000011-3cfa-11f1-0000-000000000011
# ╠═f99be7e2-5b21-4602-804e-0b92516e69d9
# ╠═f1404c52-b5b6-4c58-af72-b4e274849c20
# ╟─a1000012-3cfa-11f1-0000-000000000012
# ╟─a1000013-3cfa-11f1-0000-000000000013
# ╠═8145427c-8955-4614-b46a-56408e0c39d9
# ╠═19bc1a71-168c-4eb4-8073-5f4c7dd7f0e0
# ╠═b4cb977b-0214-4ad2-a99a-f4e60f44f8dc
# ╟─a1000015-3cfa-11f1-0000-000000000015
# ╠═a1000016-3cfa-11f1-0000-000000000016
# ╠═a1000017-3cfa-11f1-0000-000000000017
# ╠═a1000018-3cfa-11f1-0000-000000000018
# ╟─a1000019-3cfa-11f1-0000-000000000019
# ╠═a1000020-3cfa-11f1-0000-000000000020
# ╟─a1000021-3cfa-11f1-0000-000000000021
# ╠═a1000022-3cfa-11f1-0000-000000000022
# ╟─6e5dba88-bafb-4369-bff5-5c3a85356066
# ╠═a1000023-3cfa-11f1-0000-000000000023
# ╟─a1000024-3cfa-11f1-0000-000000000024
# ╠═a1000025-3cfa-11f1-0000-000000000025
# ╠═a1000026-3cfa-11f1-0000-000000000026
# ╠═c9d13fc6-30ee-42c5-bb4d-ad3a1a118ebc
# ╟─a1000027-3cfa-11f1-0000-000000000027
# ╠═a1000028-3cfa-11f1-0000-000000000028
# ╟─a1000029-3cfa-11f1-0000-000000000029
# ╠═d4b6ccb7-060f-4bc9-bd3d-207b13f0e3b3
# ╠═a1000030-3cfa-11f1-0000-000000000030
# ╠═b3434d0e-e573-44e4-996a-6dc5f69121e5
# ╠═a1000031-3cfa-11f1-0000-000000000031
# ╠═a1000032-3cfa-11f1-0000-000000000032
# ╟─b1000001-3cfa-11f1-0000-000000000001
# ╠═b1000002-3cfa-11f1-0000-000000000002
# ╠═382fabed-38d4-4f32-a71f-d481e3179764
# ╠═6c12f943-c01c-4693-847b-d1f452df744d
# ╟─b1000004-3cfa-11f1-0000-000000000004
# ╠═b1000005-3cfa-11f1-0000-000000000005
# ╟─b1000006-3cfa-11f1-0000-000000000006
# ╠═b1000007-3cfa-11f1-0000-000000000007
# ╠═b1000008-3cfa-11f1-0000-000000000008
# ╟─b1000009-3cfa-11f1-0000-000000000009
# ╠═b1000010-3cfa-11f1-0000-000000000010
# ╠═b1000011-3cfa-11f1-0000-000000000011
