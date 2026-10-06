# ==============================================================================
# define_offtaker_parameters.jl — Offtaker agent parameters
# ==============================================================================
#
# PURPOSE:
#   Copies all key-value pairs from the agent's data block (Hydrogen_Offtaker in
#   data.yaml) into mod.ext[:parameters], so that build_offtaker_agent! and
#   solve_offtaker_agent! can read Type, Capacity, Capacity_H2_In, Capacity_EP_Out,
#   Alpha, ProcessingCost, MarginalCost, gamma_NH3, ImportCost, etc. Also sets
#   gamma_GC (default 0.42) for the 42% H₂ GC mandate shared by green and grey
#   offtakers.
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — data = merge(General, Hydrogen_Offtaker[agent]).
#
# ==============================================================================

function define_offtaker_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]

    # Copy every key-value pair from the agent's data.yaml block into params.
    # Why copy all keys generically instead of listing them explicitly?
    # → Flexible: any new parameter added to data.yaml (e.g. a ramp rate or
    #   storage capacity) is automatically available in build_offtaker_agent!
    #   and solve_offtaker_agent! without modifying this function.
    # Symbol(k) converts the YAML string key to a Julia Symbol so we can
    # access values as params[:Capacity], params[:Alpha], etc.
    for (k, v) in data
        params[Symbol(k)] = v
    end

    # Regulatory green-certificate mandate: at least 42% of end-product output
    # must be backed by hydrogen Guarantees of Origin (H₂ GCs). This reflects
    # EU renewable-energy targets for hard-to-abate sectors. Both green and grey
    # offtakers are subject to this constraint. Defaults to 0.42 if not overridden.
    params[:gamma_GC] = get(data, "gamma_GC", 0.42)

    # Green ammonia nameplate is product output (MW_EP / t NH₃), not H₂ feed.
    # H₂ intake is implied: Capacity_H2_In = Capacity_EP_Out / Alpha.
    if String(get(data, "Type", "")) == "GreenOfftaker"
        derive_green_offtaker_capacities!(params, data)
    end

    # --- Grey ammonia: gas-price-dependent marginal cost, one value per scenario ---
    # Grey ammonia is SMR-based, so its variable cost is dominated by natural gas
    # and tracks the same gas price that drives the conventional generator:
    #
    #   MC[jy] = GasIntensity × gas_price × gas_multiplier[jy]
    #            + CO2Intensity × CO2_price
    #            + VariableOM
    #
    # GasIntensity ≈ 1.72 MWh_th per MWh_EP (32 GJ_LHV/t NH₃ ÷ 5.167 MWh/t) is
    # numerically the same as a 58%-efficient CCGT's 1/0.58, which is why grey
    # ammonia and gas-fired power move together under a gas shock.
    #
    # Stored as :MarginalCostByYear. The scalar :MarginalCost is kept as the
    # base-scenario value so legacy/reporting paths that expect a scalar still
    # read something sensible; every optimisation path uses the vector.
    n_yr = Int(get(data, "nYears", 1))
    if String(get(data, "Type", "")) == "GreyOfftaker"
        gas_mult = scenario_gas_multipliers(data, n_yr)
        fuel = get(data, "Fuel", Dict{String,Any}())

        if haskey(data, "GasIntensity")
            gas_int = Float64(data["GasIntensity"])
            co2_int = Float64(get(data, "CO2Intensity", 0.0))
            vom     = Float64(get(data, "VariableOM", 0.0))
            gas_p   = Float64(get(fuel, "GasPrice", 0.0))
            co2_p   = Float64(get(fuel, "CO2Price", 0.0))
            mc = [gas_int * gas_p * gas_mult[jy] + co2_int * co2_p + vom for jy in 1:n_yr]
        else
            # Legacy: a flat, gas-invariant marginal cost from data.yaml.
            mc = fill(Float64(get(data, "MarginalCost", 0.0)), n_yr)
        end

        params[:MarginalCostByYear] = mc
        params[:MarginalCost] = mc[1]
    end

    return mod
end

function derive_green_offtaker_capacities!(params::Dict, data::AbstractDict)
    alpha = Float64(get(data, "Alpha", 1.0))
    alpha > 0 || error("GreenOfftaker Alpha must be positive")
    cap_ep = Float64(data["Capacity_EP_Out"])
    cap_h2 = cap_ep / alpha
    if haskey(data, "Capacity_H2_In")
        yaml_h2 = Float64(data["Capacity_H2_In"])
        if abs(yaml_h2 - cap_h2) / max(cap_h2, 1e-9) > 0.02
            error("Capacity_H2_In ($yaml_h2 MW_H2) must equal " *
                  "Capacity_EP_Out / Alpha " *
                  "($cap_ep / $alpha = $(round(cap_h2; digits=3)) MW_H2)")
        end
    end
    params[:Capacity_EP_Out] = cap_ep
    params[:Capacity_H2_In] = cap_h2
    params[:Alpha] = alpha
    return params
end
