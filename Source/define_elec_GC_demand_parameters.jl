# ==============================================================================
# define_elec_GC_demand_parameters.jl — Electricity GC demand agent parameters
# ==============================================================================
#
# PURPOSE:
#   Sets parameters and timeseries for the electricity GC demand agent: peak
#   load, load profile column name, and quadratic utility parameters A_GC, B_GC.
#   Builds the 3D load profile LOAD_GC[jh, jd, jy] from the full-year time
#   series so that demand is elastic but bounded by peak * profile (inverse
#   demand interpretation in the objective).
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — data = merge(General, Electricity_GC_Demand[agent]).
#
# ==============================================================================

function define_elec_GC_demand_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]
    times  = mod.ext[:timeseries]

    n_ts = data["nTimesteps"]
    n_rd = data["nReprDays"]
    n_yr = data["nYears"]
    JY   = mod.ext[:sets][:JY]
    JD   = mod.ext[:sets][:JD]
    # Scenario labels are 1..nYears (file keys). Fallback if Main.years missing.
    _years    = isdefined(Main, :years) ? Main.years : Dict(1 => 1)

    params[:PeakLoad]    = data["PeakLoad"]            # Peak GC demand (MW_GC); scales the normalized profile to absolute units
    params[:Load_Column] = String(data["Load_Column"])  # Timeseries column with the normalized GC demand shape (0–1)

    # Quadratic utility for GC demand — same functional form as the electricity
    # consumer: U(d) = A_GC·d − ½·B_GC·d², giving inverse demand p(d) = A_GC − B_GC·d.
    # A_GC = intercept (€/MWh_GC): maximum willingness to pay for the first GC unit.
    # B_GC = slope (€/MWh_GC²): rate at which willingness to pay declines with volume.
    params[:A_GC]        = data["A_GC"]
    params[:B_GC]        = data["B_GC"]
    col = params[:Load_Column]

    # Build 3D load profile LOAD_GC[jh, jd, jy] (normalized, 0–1).
    # The timeseries CSV stores representative-day data sequentially in its
    # first (nReprDays * nTimesteps) rows, so we index directly:
    # row = (jd-1)*n_ts + jh.
    times[:LOAD_GC] = Array{Float64}(undef, n_ts, n_rd, n_yr)
    for jy in JY
        yr = _years[jy]
        for jd in JD, jh in 1:n_ts
            row = (jd - 1) * n_ts + jh
            times[:LOAD_GC][jh, jd, jy] = ts[yr][!, Symbol(col)][row]
        end
    end
    return mod
end
