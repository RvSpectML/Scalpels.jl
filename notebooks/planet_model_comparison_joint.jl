### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000001
if true  # true when developing Scalpels locally
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 00000002-0000-0000-0000-000000000002
begin
using Revise
using Scalpels
using LinearAlgebra
using Statistics, StatsBase
using DataFrames
using CSV, FITSIO
using Glob
using Plots
end

# ╔═╡ 10000001-0000-0000-0000-000000000001
md"""
# Planet Model Comparison via SCALPELS (Per-Instrument)

Removes stellar activity and searches for 0–`max_num_pl` planets for each
instrument independently on a shared period grid. Models compared by AIC
and CCF reconstruction quality.

## Contents
1. Configuration
2. Helper functions
3. File discovery
4. Load data
5. Instrument offset correction
6. Split by instrument
7. Shared period grid
8. 0-planet model
9. Multi-planet models
10. Final summary
11. Save results
"""

# ╔═╡ 20000001-0000-0000-0000-000000000001
md"## 1. Configuration"

# ╔═╡ 20000002-0000-0000-0000-000000000002
# Data directory: first ARGS element when run as a script; default inside Pluto.
data_dir = if isdefined(Main, :PlutoRunner)
joinpath(@__DIR__, "..", "data", "DS3")
else
isempty(ARGS) ? joinpath(@__DIR__, "..", "data", "DS1") : first(ARGS)
end

# ╔═╡ 20000003-0000-0000-0000-000000000003
begin
Pmin              = 2.0    # Minimum trial period (days)
Pmax              = 200.0  # Maximum trial period (days)
oversample_factor = 4.0    # Frequency grid oversampling factor
max_num_basis     = 4      # Maximum SCALPELS feature vectors to test
max_num_pl        = 3      # Maximum number of planets to model
run_analysis      = true   # Set false to skip the expensive period search
use_rms_weights   = true   # true → 1/std(rv_order)² weights; false → CSV order weights
end;

# ╔═╡ 20000004-0000-0000-0000-000000000004
# Known per-instrument RV offsets (m/s) for ESSP datasets.
# These are subtracted before analysis when use_fixed_offsets = true.
known_instrument_offsets = Dict{String, Float64}(
"neid"   => -79.728,
"expres" => -88.580,
"harps"  =>  566.258,
"harpsn" =>  460.051,
);

# ╔═╡ 20000005-0000-0000-0000-000000000005
# true  → subtract known_instrument_offsets per instrument
# false → subtract per-instrument inverse-variance weighted mean from data
use_fixed_offsets = false;

# ╔═╡ 30000001-0000-0000-0000-000000000001
md"## 2. Helper Functions"

# ╔═╡ 30000002-0000-0000-0000-000000000002
"""Read one FITS CCF file; returns a NamedTuple of OBO arrays."""
function read_ccf_file(path::AbstractString)
FITS(path) do f
v_grid    = read(f["V_GRID"])
orders    = read(f["ECHELLE_ORDERS"])
obo_ccf   = read(f["OBO_CCF"])
obo_e_ccf = read(f["OBO_E_CCF"])
obo_rv    = Float64.(read(f["OBO_RV"]))
obo_e_rv  = Float64.(read(f["OBO_E_RV"]))
(; v_grid, obo_ccf, obo_e_ccf, obo_rv, obo_e_rv, orders)
end
end

# ╔═╡ 30000003-0000-0000-0000-000000000003
"""
Collapse OBO CCF matrix `ccf` (Nv × No) to a single combined CCF vector.
NaN entries in either `ccf` or `weights` are excluded.
"""
function combine_orders_ccf(ccf::AbstractMatrix, weights::AbstractVector)
bad   = isnan.(ccf) .| ismissing.(ccf)
w     = ifelse.(isnan.(weights) .| ismissing.(weights), 0.0, Float64.(weights))
w_sum = sum(w)
ccf_combined = (ifelse.(bad, 0.0, Float64.(ccf)) * w) / w_sum
(; ccf_combined, weights = w)
end

# ╔═╡ 30000004-0000-0000-0000-000000000004
"""
Collapse per-order RVs and uncertainties to a single combined (rv, σrv).
Uses inverse-variance weighted mean, excluding NaN or zero-weight orders.
"""
function combine_orders_rv(rv::AbstractVector, σrv::AbstractVector,
                            weights::AbstractVector)
bad   = isnan.(rv) .| ismissing.(rv) .| isnan.(weights) .| ismissing.(weights)
w     = ifelse.(bad, 0.0, Float64.(weights))
w_sum = sum(w)
rv_c  = dot(ifelse.(bad, 0.0, Float64.(rv)), w) / w_sum
#inv_var_sum = dot(ifelse.(bad, 0.0,  Float64.(σrv).^2), w)/ w_sum^2
#σrv_c = 1/sqrt(inv_var_sum) 
σrv_c = 1/sqrt(sum(w)) 
(; rv = rv_c, σrv = σrv_c)
end

# ╔═╡ 30000005-0000-0000-0000-000000000005
"""Load the time-series CSV; add a `stem` column extracted from filenames."""
function load_time_series(csv_path::AbstractString)
df = CSV.read(csv_path, DataFrame)
df.stem = map(fn -> match(r"DS\d\.\d+", fn).match,
             df[!, "Standard File Name"])
return df
end

# ╔═╡ 30000006-0000-0000-0000-000000000006
"""
Load and combine all CCF FITS files, applying continuum normalisation and
order weighting. Returns a vector of NamedTuples sorted by the call-site.

When `rms_weights_by_inst` is provided (a Dict from `compute_obo_rv_rms`),
per-order inverse-variance weights (1/std(rv_order)²) are used for both
the CCF and RV combination. Otherwise the CSV `order_weights` table is used.
"""
function load_obs_data(ccf_files, stem_to_meta, order_weights;
                       max_obs::Integer = typemax(Int),
                       rms_weights_by_inst = nothing)
rows = NamedTuple[]
for path in ccf_files[1:min(max_obs, length(ccf_files))]
base = splitext(basename(path))[1]
m    = match(r"DS\d\.\d+", base)
m === nothing && continue
stem = m.match
meta = get(stem_to_meta, stem, nothing)
meta === nothing && continue

d = read_ccf_file(path)

# Continuum normalisation per order
nan_mask = isnan.(d.obo_ccf)
d.obo_ccf[nan_mask] .= 0.0
ccf_norm = estimate_continuum(d.v_grid, d.obo_ccf;
                             quantile_level=0.9,
                             line_half_width=7.0, v_center=0.0)
d.obo_ccf[nan_mask] .= NaN
d.obo_ccf   ./= ccf_norm'
d.obo_e_ccf ./= ccf_norm'

# Per-order weights
if rms_weights_by_inst !== nothing && haskey(rms_weights_by_inst, meta.inst)
    stats   = rms_weights_by_inst[meta.inst]
    rms_map = Dict(stats.orders[i] => stats.inv_var[i] for i in 1:length(stats.orders))
    w = [get(rms_map, d.orders[i], 0.0) for i in 1:length(d.orders)]
else
    weight_idx = map(i -> searchsortedfirst(order_weights[!, "echelle"],
                                           d.orders[i], rev=true),
                    1:length(d.orders))
    order_weights[ismissing.(order_weights[!, meta.inst]), meta.inst] .= 0.0
    w = Float64.(order_weights[weight_idx, meta.inst])
end
w[vec(any(isnan.(d.obo_ccf), dims=1))] .= 0.0

(ccf_combo, _)    = combine_orders_ccf(d.obo_ccf, w)
(rv_combo, σrv_c) = combine_orders_rv(d.obo_rv, d.obo_e_rv, w)

push!(rows, (;
stem,
instrument   = meta.inst,
t            = meta.t,
rv           = rv_combo,
σrv          = σrv_c,
ccf_combined = ccf_combo,
v_grid       = d.v_grid,
))
end
return rows
end

# ╔═╡ 30000007-0000-0000-0000-000000000007
"""
Correct per-instrument RV offsets.
If `fixed_offsets` is a Dict, subtract those values.
Otherwise subtract each instrument's inverse-variance weighted mean.
Then re-centre the full dataset to a global weighted mean of zero.
"""
function correct_instrument_offsets(rvs, σrvs, inst_labels;
                                    fixed_offsets = nothing)
rv    = copy(rvs)
invar = 1.0 ./ σrvs.^2
for inst in unique(inst_labels)
mask = inst_labels .== inst
off  = if fixed_offsets !== nothing && haskey(fixed_offsets, inst)
fixed_offsets[inst]
else
sum(rv[mask] .* invar[mask]) / sum(invar[mask])
end
rv[mask] .-= off
end
rv .-= sum(rv .* invar) / sum(invar)  # global re-centering
return rv
end

# ╔═╡ 35000001-0000-0000-0000-000000000001
"""
Compute per-order temporal std(RV) and inverse-variance weights per instrument.

Reads only `OBO_RV` and `ECHELLE_ORDERS` from each FITS file. Returns a Dict
mapping instrument → `(; orders, rms_rv, inv_var)` where `inv_var[i]` is
`1/std(rv_order[i])²` (0.0 for orders with fewer than 2 valid epochs or NaN std).
"""
function compute_obo_rv_rms(ccf_files, df_ts)
    stem_meta = Dict(
        match(r"DS\d\.\d+", row["Standard File Name"]).match => (inst = row.Instrument,)
        for row in eachrow(df_ts)
    )

    inst_rv_lists  = Dict{String, Vector{Vector{Float64}}}()
    inst_order_ref = Dict{String, Vector{Int}}()

    for path in ccf_files
        base = splitext(basename(path))[1]
        m    = match(r"DS\d\.\d+", base)
        m === nothing && continue
        stem = m.match
        meta = get(stem_meta, stem, nothing)
        meta === nothing && continue

        obo_rv, orders = FITS(path) do f
            Float64.(read(f["OBO_RV"])), read(f["ECHELLE_ORDERS"])
        end

        inst = meta.inst
        if !haskey(inst_rv_lists, inst)
            inst_rv_lists[inst]  = Vector{Float64}[]
            inst_order_ref[inst] = orders
        end
        push!(inst_rv_lists[inst], obo_rv)
    end

    return Dict(
        inst => begin
            rvmat  = stack(rvlist)
            rms_rv = [begin
                v = filter(!isnan, view(rvmat, i, :))
                length(v) >= 2 ? std(v) : NaN
            end for i in 1:size(rvmat, 1)]
            inv_var = [(isnan(r) || r <= 0.0) ? 0.0 : 1.0 / r^2 for r in rms_rv]
            (; orders = inst_order_ref[inst], rms_rv, inv_var)
        end
        for (inst, rvlist) in inst_rv_lists
    )
end

# ╔═╡ 40000001-0000-0000-0000-000000000001
md"## 3. File Discovery"

# ╔═╡ 40000002-0000-0000-0000-000000000002
begin
ccf_dir   = joinpath(data_dir, "CCFs")
csv_files = glob("DS*_timeSeries.csv", data_dir)
@assert !isempty(csv_files) "No *_timeSeries.csv found in $data_dir"
csv_path  = first(csv_files)
ccf_files = sort(glob("DS*_ccfs_*.fits", ccf_dir))
order_weights_path = joinpath(data_dir, "..", "order_weights.csv")
md"""
- CSV: $(basename(csv_path))
- CCF files found: **$(length(ccf_files))**
- Order weights: $(order_weights_path)
"""
end

# ╔═╡ 50000001-0000-0000-0000-000000000001
md"## 4. Load Data"

# ╔═╡ 50000002-0000-0000-0000-000000000002
df_ts = load_time_series(csv_path);

# ╔═╡ 50000007-0000-0000-0000-000000000007
# Compute per-order temporal std(RV) per instrument for inverse-variance weighting.
# Only executed when use_rms_weights = true; otherwise skipped (nothing).
obo_rv_stats = use_rms_weights ? compute_obo_rv_rms(ccf_files, df_ts) : nothing;

# ╔═╡ be979827-99c1-4c98-be9a-911de0cd72ca
obo_rv_stats

# ╔═╡ 50000003-0000-0000-0000-000000000003
stem_to_meta = Dict(
row.stem => (t = row["Time [eMJD]"], inst = row.Instrument)
for row in eachrow(df_ts)
);

# ╔═╡ 50000004-0000-0000-0000-000000000004
order_weights = CSV.read(order_weights_path, DataFrame);

# ╔═╡ 50000005-0000-0000-0000-000000000005
obs_data = load_obs_data(ccf_files, stem_to_meta, order_weights;
                         rms_weights_by_inst = obo_rv_stats);

# ╔═╡ 50000006-0000-0000-0000-000000000006
begin
perm        = sortperm([r.t for r in obs_data])
t           = [obs_data[i].t          for i in perm]
rvs         = [obs_data[i].rv         for i in perm]
σrvs        = [obs_data[i].σrv        for i in perm]
inst_labels = [obs_data[i].instrument for i in perm]
v_grid      = obs_data[1].v_grid
ccfs        = stack(i -> obs_data[i].ccf_combined, perm)
num_obs     = length(t)
num_vel     = length(v_grid)
md"""
Loaded **$(num_obs)** observations over
**$(round(last(t) - first(t), digits=1))** days
from: $(join(unique(inst_labels), ", ")).
"""
end

# ╔═╡ 60000001-0000-0000-0000-000000000001
md"## 5. Instrument Offset Correction"

# ╔═╡ 60000002-0000-0000-0000-000000000002
rvs_corr = correct_instrument_offsets(
rvs, σrvs, inst_labels;
fixed_offsets = use_fixed_offsets ? known_instrument_offsets : nothing
);

# ╔═╡ 60000003-0000-0000-0000-000000000003
let
insts = unique(inst_labels)
DataFrame(
instrument   = insts,
n_obs        = [count(==(i), inst_labels) for i in insts],
median_σrv   = [round(median(σrvs[inst_labels .== i]),       digits=3) for i in insts],
mean_rv_corr = [round(mean(rvs_corr[inst_labels .== i]),     digits=3) for i in insts],
rms_rv_corr  = [round(std(rvs_corr[inst_labels .== i]),      digits=3) for i in insts],
)
end

# ╔═╡ 65000001-0000-0000-0000-000000000001
md"## 6. Split by Instrument"

# ╔═╡ 256f80f3-72c2-4a35-9f18-dda9c429d850
begin
instruments = sort(unique(inst_labels))
inst_data = Dict(
inst => let
mask       = inst_labels .== inst
t_i        = t[mask]
rvs_i      = rvs_corr[mask]
σrvs_i     = σrvs[mask]
ccfs_i     = ccfs[:, mask]
sv_i       = svd(ccfs_i).S
sv_cumul_i = cumsum(sv_i.^2) ./ sum(sv_i.^2)
(; t = t_i, rvs = rvs_i, σrvs = σrvs_i, ccfs = ccfs_i, sv_cumul = sv_cumul_i)
end
for inst in instruments
)
end

# ╔═╡ af3ec2d0-bc04-4a70-b11f-190e472f992f
md"""
## Instruments: $(join(instruments, ", "))
"""

# ╔═╡ 78f26eee-0ad2-4c9c-b753-3b36fb24b3f7
DataFrame(inst=instruments, n_obs = map(inst->length(inst_data[inst].t),instruments), tspan = map(inst->-(reverse(extrema(inst_data[inst].t))...),instruments))

# ╔═╡ 70000001-0000-0000-0000-000000000001
md"## 7. Shared Period Grid"

# ╔═╡ 70000002-0000-0000-0000-000000000002
# Built from the full combined time baseline for densest frequency coverage.
period_list = make_period_list(t, Pmin, Pmax; oversample_factor);

# ╔═╡ 70000003-0000-0000-0000-000000000003
md"""
**$(length(period_list))** trial periods from **$(round(first(period_list), digits=1))** to **$(round(last(period_list), digits=1))** days.
"""

# ╔═╡ 80000001-0000-0000-0000-000000000001
md"""
## 8. 0-Planet Model

Sweep SCALPELS feature vectors from 0 to `max_num_basis` with no planet
model for each instrument independently. Select `k₀` minimising RMS of
the cleaned RVs.
"""

# ╔═╡ e268dc8d-f9bc-42c2-9bf9-10d800f73c40
results_0pl = Dict(
    inst => let d = inst_data[inst]
        invar       = 1.0 ./ d.σrvs.^2
        rv_centered = d.rvs .- mean(d.rvs, weights(invar))
        loocv_out   = loocv(rv_centered, d.ccfs;
                            σ_rvs = d.σrvs,
                            max_scalpels_vectors = max_num_basis)
        u, α = loocv_out.u_loocv, loocv_out.α_loocv
        idx_perm = 1:max_num_basis
		#=
		idx_perm, _, aic_list, _ = reorder_uloocv(
            u, α, rv_centered, d.σrvs;
            max_scalpels_vectors = max_num_basis
        )
		=#
        u_ord = u[:, idx_perm]
        α_ord = α[:, idx_perm]
        aic_sweep = aic_zero_planet_vs_num_basis_loocv(u_ord, α_ord, rv_centered, d.σrvs)
        rms_sweep = aic_sweep.rms
        k_0       = aic_sweep.num_basis[argmin(rms_sweep)] #  3 # aic_sweep.num_basis[argmin(aic_sweep.aic)]
		rv_clean  = if k_0 == 0
            			rv_centered
        			else
            			rv_shape = sum(view(u_ord, :, 1:k_0) .* view(α_ord, :, 1:k_0), dims=2)
            			vec(rv_centered .- rv_shape)
					end
        u_loocv   = k_0 > 0 ? u_ord[:, 1:k_0] : zeros(length(rv_centered), 0)
        α_loocv   = k_0 > 0 ? α_ord[:, 1:k_0] : zeros(length(rv_centered), 0)
        (; aic_sweep, k_0, rv_clean, rms_sweep, u_loocv, α_loocv)
    end
    for inst in instruments
  );

# ╔═╡ 80000004-0000-0000-0000-000000000004
md"### 0-Planet Summary"

# ╔═╡ 80000005-0000-0000-0000-000000000005
DataFrame(map(instruments) do inst
    r  = results_0pl[inst]
    d  = inst_data[inst]
    fv = r.k_0 > 0 ? d.sv_cumul[r.k_0] : 0.0
    (
        instrument   = inst,
        n_obs        = length(d.t),
        k_0          = r.k_0,
        rms_raw      = round(r.rms_sweep[1],         digits = 3),
        rms_clean    = round(r.rms_sweep[r.k_0 + 1], digits = 3),
        frac_var_ccf = round(fv,                     digits = 4),
    )
end)

# ╔═╡ 90000001-0000-0000-0000-000000000001
md"""
## 9. Multi-Planet Models (1 to $(max_num_pl) planets, per instrument)

For each instrument independently, using the shared period grid:
- **a**: Greedy sequential period search via `vscalpels_recover_loocv` at
  instrument-specific `k₀` feature vectors, fixing previously found periods.
- **b**: Optimise feature vector count for the joint model via AIC.
"""

# ╔═╡ 90000002-0000-0000-0000-000000000002
planet_results = if run_analysis
    k_search_joint = max(1, maximum(results_0pl[inst].k_0 for inst in instruments))
    inst_data_vec  = [
        (; bjd  = inst_data[inst].t,
           rvs  = inst_data[inst].rvs,
           σ_rv = inst_data[inst].σrvs,
           ccfs = inst_data[inst].ccfs)
        for inst in instruments
    ]
    search_planets_loocv_joint(
        inst_data_vec,
        period_list;
        max_num_pl       = max_num_pl,
        k_search         = k_search_joint,
        min_period_ratio = 1.1,
        max_num_basis    = max_num_basis,
        resort           = false,
        fixed_k          = true,
    )
else
    (; results = NamedTuple[])
end;

# ╔═╡ 90000003-0000-0000-0000-000000000003
begin
"""CCF fraction-of-variance explained at `k` feature vectors for `inst`."""
function ccf_frac_var(inst, k)
    d = inst_data[inst]
    k == 0 ? 0.0 : d.sv_cumul[clamp(k, 1, length(d.sv_cumul))]
end

"""Build a per-planet-count summary DataFrame across all instruments."""
function planet_summary_table(num_pl)
    DataFrame(map(enumerate(instruments)) do (i, inst)
        if !run_analysis || isempty(planet_results.results)
            return (; instrument = inst, periods_days = "", k_opt = missing,
                     rms_resid = missing, min_aic = missing, frac_var_ccf = missing)
        end
        fr = planet_results.results[num_pl].fit_results[i]
        (
            instrument   = inst,
            periods_days = join(round.(fr.periods, digits = 2), ", "),
            k_opt        = fr.k_opt,
            rms_resid    = round(std(fr.final_fit.rvresid),  digits = 3),
            min_aic      = round(minimum(fr.sweep.aic),      digits = 1),
            frac_var_ccf = round(ccf_frac_var(inst, fr.k_opt), digits = 4),
        )
    end)
end
end

# ╔═╡ 91000001-0000-0000-0000-000000000001
md"### 1-Planet Results"

# ╔═╡ 91000003-0000-0000-0000-000000000003
md"#### 1-Planet Summary Table"

# ╔═╡ 91000004-0000-0000-0000-000000000004
run_analysis ? planet_summary_table(1) : md"*(analysis skipped)*"

# ╔═╡ 92000001-0000-0000-0000-000000000001
max_num_pl >= 2 ? md"### 2-Planet Results" : md""

# ╔═╡ 83eb6ec5-c232-4b0b-b6aa-c008231c7334
run_analysis ? planet_results.results[2].period_list : missing

# ╔═╡ 92000003-0000-0000-0000-000000000003
md"#### 2-Planet Summary Table"

# ╔═╡ 92000004-0000-0000-0000-000000000004
run_analysis && max_num_pl >= 2 ? planet_summary_table(2) : md"*(not computed)*"

# ╔═╡ 93000001-0000-0000-0000-000000000001
max_num_pl >= 3 ? md"### 3-Planet Results" : md""

# ╔═╡ 93000003-0000-0000-0000-000000000003
md"#### 3-Planet Summary Table"

# ╔═╡ 93000004-0000-0000-0000-000000000004
run_analysis && max_num_pl >= 3 ? planet_summary_table(3) : md"*(not computed)*"

# ╔═╡ 91a49e87-9e86-4a29-ba7a-6ae905d5015c
let i = findfirst(==("expres"), instruments)
    run_analysis ? planet_results.results[1].fit_results[i].sweep.aic : missing
end

# ╔═╡ f1408a3b-ddfa-4173-a605-3e49a6dff428
let i = findfirst(==("expres"), instruments)
    run_analysis ? planet_results.results[2].fit_results[i].sweep.aic : missing
end

# ╔═╡ 106ed2f6-6fa5-48b5-b5b2-1b55944a9068
let i = findfirst(==("expres"), instruments)
    run_analysis && max_num_pl >= 3 ? planet_results.results[3].fit_results[i].sweep.aic : missing
end

# ╔═╡ a0000001-0000-0000-0000-000000000001
md"""
## 10. Final Summary

Best planet count per instrument selected by minimum AIC across the
1–`max_num_pl` planet models.
"""

# ╔═╡ a0000002-0000-0000-0000-000000000002
final_summary = DataFrame(map(instruments) do inst
    r0 = results_0pl[inst]
    d  = inst_data[inst]

    best_num_pl  = 0
    best_periods = ""
    best_rms     = round(r0.rms_sweep[r0.k_0 + 1], digits = 3)

    if run_analysis && !isempty(planet_results.results)
        i    = findfirst(==(inst), instruments)
        aics = [minimum(planet_results.results[n].fit_results[i].sweep.aic)
                for n in 1:max_num_pl]
        best_num_pl  = argmin(aics)
        fr           = planet_results.results[best_num_pl].fit_results[i]
        best_periods = join(round.(fr.periods, digits = 2), ", ")
        best_rms     = round(std(fr.final_fit.rvresid), digits = 3)
    end

    (
        instrument   = inst,
        n_obs        = length(d.t),
        k_0          = r0.k_0,
        rms_0pl      = round(r0.rms_sweep[r0.k_0 + 1], digits = 3),
        best_num_pl,
        best_periods,
        best_rms,
    )
end);

# ╔═╡ 6b414786-1bef-49d5-b611-809b50f2d778
planet_results.results[1].fit_results[1].final_fit

# ╔═╡ ed7b8dfc-68b5-4f4a-b973-20e930b0f718
md"""
### Final Summary Table
"""

# ╔═╡ cc45a010-7689-4c0e-9b19-15567ee159c4
final_summary

# ╔═╡ f388e23a-739a-434c-a8b3-23f3137b5a02
md"""
| Column | Description |
|--------|-------------|
| `k_0` | Optimal SCALPELS feature vectors for 0-planet model |
| `rms_0pl` | RMS of cleaned RVs at 0-planet model (m/s) |
| `best_num_pl` | Recommended planet count (minimum AIC across 1–$(max_num_pl) planets) |
| `best_periods` | Orbital periods of the recommended model (days) |
| `best_rms` | RMS of residuals at the recommended model (m/s) |
"""

# ╔═╡ b0000001-0000-0000-0000-000000000001
md"## 11. Save Results"

# ╔═╡ b0000002-0000-0000-0000-000000000002
begin
	results_dir = joinpath(data_dir, "results")
	mkpath(results_dir)
	results_dir
end

# ╔═╡ eb5eebd0-0cdb-4032-a1c5-0e762d939435
let
	for n_pl in 1:3
	df_out_pl = DataFrame()
	for inst in 1:length(instruments)	
		df_tmp = DataFrame(
	"K [m/s]" => planet_results.results[n_pl].fit_results[inst].final_fit.amp,
	"P [d]"=>planet_results.results[n_pl].fit_results[inst].final_fit.periods,
	"t0 [eMJD]" => planet_results.results[n_pl].fit_results[inst].final_fit.phase,
	"e" => zeros(n_pl),
	"w [deg]" => zeros(n_pl),
	"Kx [m/s]" => planet_results.results[n_pl].fit_results[inst].final_fit.Kx,
	"Ky [m/s]"=>planet_results.results[n_pl].fit_results[inst].final_fit.Ky,
	"σKx [m/s]" => planet_results.results[n_pl].fit_results[inst].final_fit.dKx,
	"σKy [m/s]"=>planet_results.results[n_pl].fit_results[inst].final_fit.dKy,
	"Instrument" => fill(instruments[inst],n_pl))
		append!(df_out_pl, df_tmp)
	end
	sort!(df_out_pl,Symbol("P [d]"))
	fn = last(split(data_dir,"/")) * "_PSU_Scalpels" * string(n_pl) * "pl_planetFit.csv"
	CSV.write(joinpath(results_dir, fn), df_out_pl)
	@info "Saved `$fn`."
	end
end  

# ╔═╡ 65d6b218-8a99-4df5-a2a8-0db3531e3b2f
 let
  dfs = map(instruments) do inst
      d     = inst_data[inst]
      r     = results_0pl[inst]
      invar = 1.0 ./ d.σrvs.^2
      rv_raw   = d.rvs .- mean(d.rvs, weights(invar))
      rv_clean = r.rv_clean .- mean(r.rv_clean, weights(invar))
      df = DataFrame(
          instrument = fill(inst, length(d.t)),
          t          = d.t,
          RV_C       = rv_clean,
		  #rv_obs     = rv_raw,
          #rv_clean   = rv_clean,
          #rv_orbit   = zeros(length(d.t)),
          #rv_resid   = rv_clean,
		  sigma_rv   = d.σrvs,
		  RV_A       = d.rvs - rv_clean
      )
	  #df[!,"Standardd File Name"] = TODO
      for k in 1:r.k_0
          df[!, "Ind. $k"] = r.u_loocv[:, k]
		  #df[!, "shape_score_$k"] = r.u_loocv[:, k]
          #df[!, "rv_score_$k"]    = r.α_loocv[:, k]
      end
      df
  end
  df = sort(vcat(dfs...; cols = :union), [:instrument, :t])
  rename!(df, "t" => "Time [eMJD]")
  rename!(df, "sigma_rv" => "eRV_C")
  fn = last(split(data_dir,"/")) * "_PSU_Scalpels0pl_results.csv"
  CSV.write(joinpath(results_dir, fn), df)
  @info "Saved $fn ($(nrow(df)) rows, $(ncol(df)) columns)."
  end

# ╔═╡ f795600d-f744-45e2-b805-b21ace067d6d
 # Multi-planet results — one file per planet count, instrument column
  if run_analysis
  for num_pl in 1:max_num_pl
      dfs = map(enumerate(instruments)) do (i, inst)
          fr  = planet_results.results[num_pl].fit_results[i]
          f   = fr.final_fit
          d   = inst_data[inst]
          df  = DataFrame(
			  instrument = fill(inst, length(d.t)),
              t       = d.t,
              #rv_obs     = d.rvs .- mean(d.rvs),
			  #rv_clean   = vec(f.rvclean),
              #rv_orbit   = vec(f.rvorbit),
              #rv_resid   = vec(f.rvresid),
			  RV_C       = vec(f.rvresid) .+ vec(f.rvorbit),
              sigma_rv   = d.σrvs,
			  RV_A       = d.rvs - vec(f.rvresid) .+ vec(f.rvorbit)
          )
		  #df[!,"Standardd File Name"] = TODO
          for k in 1:size(f.u_loocv, 2)
              df[!, "Ind. $k"] = f.u_loocv[:, k]
			  #df[!, "shape_score_$k"] = f.u_loocv[:, k]
              #df[!, "rv_score_$k"]    = f.α_loocv[:, k]
          end
          df
      end
      df = sort(vcat(dfs...; cols = :union), [:instrument, :t])
	  rename!(df, "t" => "Time [eMJD]")
	  rename!(df, "sigma_rv" => "eRV_C")
	  fn = last(split(data_dir,"/")) * "_PSU_Scalpels$(num_pl)pl_results.csv"
	  CSV.write(joinpath(results_dir, fn), df)
  	@info "Saved " fn
  end
	wrote_results_files = true
  end

# ╔═╡ c8c22be8-6a0f-4ee7-b6a6-a5e81d05a7a8
final_summary

# ╔═╡ 9bbd555e-73e5-4f08-868c-ab0cd3fe9996
let
	wrote_results_files
	num_pl_to_use_for_DS = [ 1, 1, 1, 2, 2, 1, 2, 2, 0 ]
	DSid = parse(Int,last(split(data_dir,"/"))[3])
	num_pl_to_use_for_DS[DSid]
	fn_in = last(split(data_dir,"/")) * "_PSU_Scalpels" * string(num_pl_to_use_for_DS[DSid]) * "pl_results.csv"
	fn_out = last(split(data_dir,"/")) * "_PSU_Scalpels_results.csv"
	cp(joinpath(results_dir,fn_in),joinpath(results_dir,fn_out), force=true)
	if num_pl_to_use_for_DS[DSid] >= 1
		fn_in = last(split(data_dir,"/")) * "_PSU_Scalpels" * string(num_pl_to_use_for_DS[DSid]) * "pl_planetFit.csv"
		fn_out = last(split(data_dir,"/")) * "_PSU_Scalpels_planetFit.csv"
		cp(joinpath(results_dir,fn_in),joinpath(results_dir,fn_out), force=true)
	end
	num_pl_to_use_for_DS[DSid]
end

# ╔═╡ 0748f263-2e29-4742-9e5a-5ea566e47919
CSV.write(joinpath(data_dir,"summary.csv"),final_summary)

# ╔═╡ 30000008-0000-0000-0000-000000000008
"""Save `fig` to `path` when running as a script; returns `fig` in all cases."""
function maybe_savefig(fig, path::AbstractString)
    if !isdefined(Main, :PlutoRunner)
        mkpath(dirname(path))
        savefig(fig, path)
    end
    fig
end

# ╔═╡ 80000003-0000-0000-0000-000000000003
let
ks = 0:max_num_basis
plts = map(instruments) do inst
    r = results_0pl[inst]
    p = plot(ks, r.rms_sweep;
            marker = :circle, ms = 3, lw = 1.5, legend = :topright,
            xlabel = "Number of feature vectors", ylabel = "RMS (m/s)",
            title  = "$inst — 0-planet RMS vs. feature vectors")
    vline!(p, [r.k_0]; ls = :dash, lw = 1.5, label = "k₀ = $(r.k_0)")
    p
end
fig0 = plot(plts...; layout = (length(instruments), 1),
    size = (700, 260 * length(instruments)))
maybe_savefig(fig0, joinpath(data_dir, "results", "plot_0pl_rms.png"))
end

# ╔═╡ fb225623-1b50-4679-8eaf-397421b0ecfc
 # 80000003b-0000-0000-0000-000000000000
  let
  ninst = length(instruments)
  plts  = map(instruments) do inst
      r  = results_0pl[inst]
      ks = collect(r.aic_sweep.num_basis)

      p_aic = plot(ks, r.aic_sweep.aic;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "AIC",
                   title  = "$inst — AIC", legend = false)
      vline!(p_aic, [r.k_0]; ls = :dash, lw = 1.5, lc = :red,
             label = "k₀ = $(r.k_0)")

      p_rms = plot(ks, r.aic_sweep.rms;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "RMS (m/s)",
                   title  = "$inst — RMS", legend = false)
      vline!(p_rms, [r.k_0]; ls = :dash, lw = 1.5, lc = :red,
             label = "k₀ = $(r.k_0)")

      (p_aic, p_rms)
  end
  fig = plot(Iterators.flatten(plts)...;
       layout = (ninst, 2),
       size   = (800, 220 * ninst))
  maybe_savefig(fig, joinpath(data_dir, "results", "plot_0pl_aic_rms.png"))
  end

# ╔═╡ 91000002-0000-0000-0000-000000000002
# Periodograms for the 1st-planet search: combined χ² + per-instrument panels
if run_analysis
res1 = planet_results.results[1]
lp   = log10.(res1.period_list)
local best_P1 = res1.fit_results[1].periods[end]  # shared across instruments

p_combined = plot(lp, -res1.chi2_total;
    lw = 0.8, legend = :topright,
    xlabel = "log₁₀(P / days)", ylabel = "−χ² (total)",
    title  = "Joint period search — 1 planet (all instruments)")
vline!(p_combined, [log10(best_P1)];
    ls = :dash, lw = 1.5, label = "P = $(round(best_P1, digits=2)) d")

plts_inst = map(enumerate(instruments)) do (i, inst)
    chi2_i = [out.χ² for out in res1.inst_search_outs[i]]
    p = plot(lp, -chi2_i;
        lw = 0.8, legend = :topright,
        xlabel = "log₁₀(P / days)", ylabel = "−χ²",
        title  = "$inst — period search (1 planet)")
    vline!(p, [log10(best_P1)];
        ls = :dash, lw = 1.5,
        label = "P = $(round(best_P1, digits=2)) d")
    p
end

fig1a = plot(p_combined, plts_inst...;
    layout = (1 + length(instruments), 1),
    size   = (700, 260 * (1 + length(instruments))))
maybe_savefig(fig1a, joinpath(data_dir, "results", "plot_1pl_periodogram.png"))
end

# ╔═╡ 839a8642-b6d2-4fd6-9778-bd06e932b6b9
# Amplitude periodogram — 1-planet search, all instruments
let
  if run_analysis
  res1  = planet_results.results[1]
  lp    = log10.(res1.period_list)
  best_P1 = res1.fit_results[1].periods[1]
  plt1b = plot(;
      xlabel = "log₁₀(Period / days)", ylabel = "K (m/s)",
      title  = "1-planet amplitude periodogram",
      legend = :topright)
  for (i, inst) in enumerate(instruments)
      amp = [out.amp[1] for out in res1.inst_search_outs[i]]
      plot!(plt1b, lp, amp; lw = 0.8, label = inst, lc = i)
      vline!(plt1b, [log10(best_P1)]; ls = :dash, lw = 1.5, lc = i, label = "")
  end
  maybe_savefig(plt1b, joinpath(data_dir, "results", "plot_1pl_amplitude.png"))
  plt1b
  end
end

# ╔═╡ 728e1bb4-8ecd-43ff-a36a-14b03853542e
# 91000005-0000-0000-0000-000000000000
let  # AIC and RMS vs. feature vectors — 1-planet model
  if run_analysis
  ninst = length(instruments)
  plts  = map(enumerate(instruments)) do (i, inst)
      fr  = planet_results.results[1].fit_results[i]
      sw  = fr.sweep
      ks  = collect(sw.num_basis)

      p_aic = plot(ks, sw.aic;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "AIC",
                   title  = "$inst — 1-planet AIC", legend = false)
      vline!(p_aic, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red)

      p_rms = plot(ks, sw.rms;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "RMS (m/s)",
                   title  = "$inst — 1-planet RMS", legend = false)
      vline!(p_rms, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red,
             label = "k_opt = $(fr.k_opt)")

      (p_aic, p_rms)
  end
  fig1c = plot(Iterators.flatten(plts)...;
       layout = (ninst, 2),
       size   = (800, 220 * ninst))
  maybe_savefig(fig1c, joinpath(data_dir, "results", "plot_1pl_aic_rms.png"))
  end
end

# ╔═╡ 92000002-0000-0000-0000-000000000002
# Periodograms for the 2nd-planet search: combined χ² + per-instrument panels
if run_analysis && max_num_pl >= 2
res2    = planet_results.results[2]
lp2     = log10.(res2.period_list)
local best_P1 = res2.fit_results[1].periods[1]
local best_P2 = res2.fit_results[1].periods[2]
p1_str  = "fixing P₁ = $(round(best_P1, digits=1)) d"

p2_combined = plot(lp2, -res2.chi2_total;
    lw = 0.8, legend = :topright,
    xlabel = "log₁₀(P / days)", ylabel = "−χ² (total)",
    title  = "Joint period search — 2nd planet ($p1_str)")
vline!(p2_combined, [log10(best_P2)];
    ls = :dash, lw = 1.5, label = "P₂ = $(round(best_P2, digits=2)) d")

plts2_inst = map(enumerate(instruments)) do (i, inst)
    chi2_i = [out.χ² for out in res2.inst_search_outs[i]]
    p = scatter(lp2, -chi2_i;
        ms = 1.0, lw = 0.8, legend = :topright,
        xlabel = "log₁₀(P / days)", ylabel = "−χ²",
        title  = "$inst — period search (2nd planet)")
    vline!(p, [log10(best_P2)];
        ls = :dash, lw = 1.5,
        label = "P₂ = $(round(best_P2, digits=2)) d")
    p
end

fig2a = plot(p2_combined, plts2_inst...;
    layout = (1 + length(instruments), 1),
    size   = (700, 260 * (1 + length(instruments))))
maybe_savefig(fig2a, joinpath(data_dir, "results", "plot_2pl_periodogram.png"))
end

# ╔═╡ f5b4b8a6-5f34-47a5-99f9-5fc15beeb5dc
# Amplitude periodogram — 2nd-planet search, all instruments
let
  if run_analysis && max_num_pl >= 2
  res2   = planet_results.results[2]
  lp2    = log10.(res2.period_list)
  best_P2 = res2.fit_results[1].periods[2]
  plt2b = plot(;
      xlabel = "log₁₀(Period / days)", ylabel = "K (m/s)",
      title  = "2nd-planet amplitude periodogram (shared P₁ fixed)",
      legend = :topright)
  for (i, inst) in enumerate(instruments)
      amp = [out.amp[2] for out in res2.inst_search_outs[i]]
      plot!(plt2b, lp2, amp; lw = 0.8, label = inst, lc = i)
      vline!(plt2b, [log10(best_P2)]; ls = :dash, lw = 1.5, lc = i, label = "")
	  ylims!(plt2b,0,3)
  end
  maybe_savefig(plt2b, joinpath(data_dir, "results", "plot_2pl_amplitude.png"))
  plt2b
  end
end

# ╔═╡ 27432d4a-66da-4aef-b6e5-848d0473a7c6
# 92000005-0000-0000-0000-000000000000
let
  # AIC and RMS vs. feature vectors — 2-planet model
  if run_analysis && max_num_pl >= 2
  ninst = length(instruments)
  plts  = map(enumerate(instruments)) do (i, inst)
      fr  = planet_results.results[2].fit_results[i]
      sw  = fr.sweep
      ks  = collect(sw.num_basis)

      p_aic = plot(ks, sw.aic;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "AIC",
                   title  = "$inst — 2-planet AIC", legend = false)
      vline!(p_aic, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red)

      p_rms = plot(ks, sw.rms;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "RMS (m/s)",
                   title  = "$inst — 2-planet RMS", legend = false)
      vline!(p_rms, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red,
             label = "k_opt = $(fr.k_opt)")

      (p_aic, p_rms)
  end
  fig2c = plot(Iterators.flatten(plts)...;
       layout = (ninst, 2),
       size   = (800, 220 * ninst))
  maybe_savefig(fig2c, joinpath(data_dir, "results", "plot_2pl_aic_rms.png"))
  end
end

# ╔═╡ 93000002-0000-0000-0000-000000000002
# Periodograms for the 3rd-planet search: combined χ² + per-instrument panels
if run_analysis && max_num_pl >= 3
res3     = planet_results.results[3]
lp3      = log10.(res3.period_list)
best_P3  = res3.fit_results[1].periods[3]
fixed_str = "fixing P₁=$(round(res3.fit_results[1].periods[1], digits=1)), P₂=$(round(res3.fit_results[1].periods[2], digits=1)) d"

p3_combined = plot(lp3, -res3.chi2_total;
    lw = 0.8, legend = :topright,
    xlabel = "log₁₀(P / days)", ylabel = "−χ² (total)",
    title  = "Joint period search — 3rd planet ($fixed_str)")
vline!(p3_combined, [log10(best_P3)];
    ls = :dash, lw = 1.5, label = "P₃ = $(round(best_P3, digits=2)) d")

plts3_inst = map(enumerate(instruments)) do (i, inst)
    chi2_i = [out.χ² for out in res3.inst_search_outs[i]]
    p = scatter(lp3, -chi2_i;
        ms = 1.0, lw = 0.8, legend = :topright,
        xlabel = "log₁₀(P / days)", ylabel = "−χ²",
        title  = "$inst — period search (3rd planet)")
    vline!(p, [log10(best_P3)];
        ls = :dash, lw = 1.5,
        label = "P₃ = $(round(best_P3, digits=2)) d")
    p
end

fig3a = plot(p3_combined, plts3_inst...;
    layout = (1 + length(instruments), 1),
    size   = (700, 260 * (1 + length(instruments))))
maybe_savefig(fig3a, joinpath(data_dir, "results", "plot_3pl_periodogram.png"))
end

# ╔═╡ 80582cde-c823-4e5a-bc88-80348ddcaee6
# Amplitude periodogram — 3rd-planet search, all instruments
let
  if run_analysis && max_num_pl >= 3
  res3    = planet_results.results[3]
  lp3     = log10.(res3.period_list)
  best_P3 = res3.fit_results[1].periods[3]
  plt3b = plot(;
      xlabel = "log₁₀(Period / days)", ylabel = "K (m/s)",
      title  = "3rd-planet amplitude periodogram (shared P₁, P₂ fixed)",
      legend = :topright)
  for (i, inst) in enumerate(instruments)
      amp = [out.amp[3] for out in res3.inst_search_outs[i]]
      plot!(plt3b, lp3, amp; lw = 0.8, label = inst, lc = i)
      vline!(plt3b, [log10(best_P3)]; ls = :dash, lw = 1.5, lc = i, label = "")
	  ylims!(plt3b, 0, 3)
  end
  maybe_savefig(plt3b, joinpath(data_dir, "results", "plot_3pl_amplitude.png"))
  plt3b
  end
end

# ╔═╡ 37b0e6f9-8cd4-4f88-abed-8f35e9aa33df
# 93000005-0000-0000-0000-000000000000
let
  # AIC and RMS vs. feature vectors — 3-planet model
  if run_analysis && max_num_pl >= 3
  ninst = length(instruments)
  plts  = map(enumerate(instruments)) do (i, inst)
      fr  = planet_results.results[3].fit_results[i]
      sw  = fr.sweep
      ks  = collect(sw.num_basis)

      p_aic = plot(ks, sw.aic;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "AIC",
                   title  = "$inst — 3-planet AIC", legend = false)
      vline!(p_aic, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red)

      p_rms = plot(ks, sw.rms;
                   marker = :circle, ms = 3, lw = 1.5,
                   xlabel = "Number of feature vectors", ylabel = "RMS (m/s)",
                   title  = "$inst — 3-planet RMS", legend = false)
      vline!(p_rms, [fr.k_opt]; ls = :dash, lw = 1.5, lc = :red,
             label = "k_opt = $(fr.k_opt)")

      (p_aic, p_rms)
  end
  fig3c = plot(Iterators.flatten(plts)...;
       layout = (ninst, 2),
       size   = (800, 220 * ninst))
  maybe_savefig(fig3c, joinpath(data_dir, "results", "plot_3pl_aic_rms.png"))
  end
end

# ╔═╡ Cell order:
# ╠═00000001-0000-0000-0000-000000000001
# ╠═00000002-0000-0000-0000-000000000002
# ╟─10000001-0000-0000-0000-000000000001
# ╟─20000001-0000-0000-0000-000000000001
# ╠═20000002-0000-0000-0000-000000000002
# ╠═20000003-0000-0000-0000-000000000003
# ╠═20000004-0000-0000-0000-000000000004
# ╠═20000005-0000-0000-0000-000000000005
# ╟─30000001-0000-0000-0000-000000000001
# ╟─30000002-0000-0000-0000-000000000002
# ╟─30000003-0000-0000-0000-000000000003
# ╟─30000004-0000-0000-0000-000000000004
# ╟─30000005-0000-0000-0000-000000000005
# ╟─30000006-0000-0000-0000-000000000006
# ╠═be979827-99c1-4c98-be9a-911de0cd72ca
# ╟─30000007-0000-0000-0000-000000000007
# ╟─35000001-0000-0000-0000-000000000001
# ╟─40000001-0000-0000-0000-000000000001
# ╟─40000002-0000-0000-0000-000000000002
# ╟─50000001-0000-0000-0000-000000000001
# ╠═50000002-0000-0000-0000-000000000002
# ╠═50000007-0000-0000-0000-000000000007
# ╠═50000003-0000-0000-0000-000000000003
# ╠═50000004-0000-0000-0000-000000000004
# ╠═50000005-0000-0000-0000-000000000005
# ╠═50000006-0000-0000-0000-000000000006
# ╟─60000001-0000-0000-0000-000000000001
# ╠═60000002-0000-0000-0000-000000000002
# ╟─60000003-0000-0000-0000-000000000003
# ╟─65000001-0000-0000-0000-000000000001
# ╠═256f80f3-72c2-4a35-9f18-dda9c429d850
# ╟─af3ec2d0-bc04-4a70-b11f-190e472f992f
# ╟─78f26eee-0ad2-4c9c-b753-3b36fb24b3f7
# ╟─70000001-0000-0000-0000-000000000001
# ╠═70000002-0000-0000-0000-000000000002
# ╟─70000003-0000-0000-0000-000000000003
# ╟─80000001-0000-0000-0000-000000000001
# ╠═e268dc8d-f9bc-42c2-9bf9-10d800f73c40
# ╠═80000003-0000-0000-0000-000000000003
# ╟─fb225623-1b50-4679-8eaf-397421b0ecfc
# ╟─80000004-0000-0000-0000-000000000004
# ╠═80000005-0000-0000-0000-000000000005
# ╟─90000001-0000-0000-0000-000000000001
# ╠═90000002-0000-0000-0000-000000000002
# ╠═90000003-0000-0000-0000-000000000003
# ╟─91000001-0000-0000-0000-000000000001
# ╟─91000002-0000-0000-0000-000000000002
# ╟─839a8642-b6d2-4fd6-9778-bd06e932b6b9
# ╟─91000003-0000-0000-0000-000000000003
# ╠═91000004-0000-0000-0000-000000000004
# ╟─728e1bb4-8ecd-43ff-a36a-14b03853542e
# ╟─92000001-0000-0000-0000-000000000001
# ╟─92000002-0000-0000-0000-000000000002
# ╟─f5b4b8a6-5f34-47a5-99f9-5fc15beeb5dc
# ╠═83eb6ec5-c232-4b0b-b6aa-c008231c7334
# ╟─92000003-0000-0000-0000-000000000003
# ╠═92000004-0000-0000-0000-000000000004
# ╟─27432d4a-66da-4aef-b6e5-848d0473a7c6
# ╟─93000001-0000-0000-0000-000000000001
# ╟─93000002-0000-0000-0000-000000000002
# ╠═80582cde-c823-4e5a-bc88-80348ddcaee6
# ╟─93000003-0000-0000-0000-000000000003
# ╠═93000004-0000-0000-0000-000000000004
# ╟─37b0e6f9-8cd4-4f88-abed-8f35e9aa33df
# ╠═91a49e87-9e86-4a29-ba7a-6ae905d5015c
# ╠═f1408a3b-ddfa-4173-a605-3e49a6dff428
# ╠═106ed2f6-6fa5-48b5-b5b2-1b55944a9068
# ╟─a0000001-0000-0000-0000-000000000001
# ╠═a0000002-0000-0000-0000-000000000002
# ╠═6b414786-1bef-49d5-b611-809b50f2d778
# ╠═eb5eebd0-0cdb-4032-a1c5-0e762d939435
# ╟─ed7b8dfc-68b5-4f4a-b973-20e930b0f718
# ╠═cc45a010-7689-4c0e-9b19-15567ee159c4
# ╟─f388e23a-739a-434c-a8b3-23f3137b5a02
# ╟─b0000001-0000-0000-0000-000000000001
# ╠═b0000002-0000-0000-0000-000000000002
# ╠═65d6b218-8a99-4df5-a2a8-0db3531e3b2f
# ╠═f795600d-f744-45e2-b805-b21ace067d6d
# ╠═c8c22be8-6a0f-4ee7-b6a6-a5e81d05a7a8
# ╠═9bbd555e-73e5-4f08-868c-ab0cd3fe9996
# ╠═0748f263-2e29-4742-9e5a-5ea566e47919
# ╟─30000008-0000-0000-0000-000000000008
