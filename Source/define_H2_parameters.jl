# ==============================================================================
# define_H2_parameters.jl — Electrolytic H₂ producer parameters
# ==============================================================================
#
# PURPOSE:
#   Fills mod.ext[:parameters] for the hydrogen-sector agent (electrolyzer).
#   IEA nameplate is electrical input (Capacity_Electrolyzer, MW_e). Implied H₂
#   output is Capacity_Electrolyzer / SpecificConsumption. CAPEX in yaml is
#   €/MW_e-year; code stores FixedCost_per_MW_H2 = F_elec × SC to charge on
#   the JuMP variable cap_H2_y (MW_H2).
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — Same as elsewhere; ts/repr_days not used for
#     this agent but kept for a uniform interface.
#
# ==============================================================================

# IEA / project nameplate is electrical input (MW_e). The JuMP capacity variable
# `cap_H2_y` is H₂ output (MW_H2), so the annuity charged on it is
#   F_H2 = F_elec × SpecificConsumption
# with F_elec in €/MW_e-year (IEA USD/kWe). Every ADMM, planner, contract, and
# merged path must use FixedCost_per_MW_H2 against cap_H2_y.
function electrolyzer_h2_annuity(params::Dict)
    f_h2 = get(params, :FixedCost_per_MW_H2, nothing)
    f_h2 !== nothing && return Float64(f_h2)
    f_e = Float64(get(params, :FixedCost_per_MW_Electrolyzer, 0.0))
    sc  = Float64(get(params, :SpecificConsumption, 1.0))
    return f_e * sc
end

function derive_electrolyzer_capacities!(params::Dict, data::AbstractDict)
    sc = Float64(data["SpecificConsumption"])
    sc > 0 || error("SpecificConsumption must be positive")
    cap_e = Float64(data["Capacity_Electrolyzer"])
    cap_h2 = cap_e / sc
    if haskey(data, "Capacity_H2_Output")
        yaml_h2 = Float64(data["Capacity_H2_Output"])
        if abs(yaml_h2 - cap_h2) / max(cap_h2, 1e-9) > 0.02
            error("Capacity_H2_Output ($yaml_h2 MW_H2) must equal " *
                  "Capacity_Electrolyzer / SpecificConsumption " *
                  "($cap_e / $sc = $(round(cap_h2; digits=3)) MW_H2)")
        end
    end
    f_e = Float64(get(data, "FixedCost_per_MW_Electrolyzer", 0.0))
    params[:SpecificConsumption] = sc
    params[:η_elec_H2] = 1.0 / sc
    params[:Capacity_Electrolyzer] = cap_e
    params[:Capacity_H2_Output] = cap_h2
    params[:FixedCost_per_MW_Electrolyzer] = f_e          # €/MW_e-year (IEA)
    params[:FixedCost_per_MW_H2] = f_e * sc               # €/MW_H2-year (charged on cap_H2_y)
    return params
end

function define_H2_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]

    params[:Type] = String(get(data, "Type", ""))
    derive_electrolyzer_capacities!(params, data)
    params[:OperationalCost] = get(data, "OperationalCost", 0.0)

    return mod
end
