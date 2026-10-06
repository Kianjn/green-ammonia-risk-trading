# ==============================================================================
# define_scenarios.jl — Builds the scenario grid shared by every entry point
# ==============================================================================
#
# PURPOSE:
#   The model's uncertainty set is the cross product of a weather year and a
#   natural-gas price level. This file turns the `Scenarios` block of data.yaml
#   into the flat scenario index jy = 1..nYears that the rest of the model uses,
#   together with the two lookups every downstream file needs:
#
#     years[jy]            -> weather-year FILE LABEL for scenario jy.
#                             Several jy share one label (same weather, different
#                             gas price), so Input/timeseries_<label>.csv and
#                             Input/output_<label>/ are read once per label and
#                             reused. define_common_parameters! consumes this via
#                             Main.years when building W[jd,jy] and AF[jh,jd,jy].
#
#     gas_multiplier[jy]   -> scaling applied to Fuel.GasPrice in scenario jy.
#                             define_power_parameters! and define_offtaker_parameters!
#                             use it to derive the conventional stage costs and the
#                             grey ammonia marginal cost, so one gas price moves
#                             power and ammonia together (see docs/TECHNICAL.md, Uncertainty set).
#
# ORDERING:
#   Gas varies fastest, so scenarios group by weather year:
#     jy = (weather_index - 1) * n_gas + gas_index
#   With 5 weather years and 3 gas levels: jy 1-3 are weather year 1 at
#   1.0/1.1/1.2 x gas, jy 4-6 are weather year 2, ... jy 13-15 weather year 5.
#
# FALLBACK:
#   If data.yaml has no `Scenarios` block, the grid degenerates to weather years
#   1..nScenarioYears (or General.nYears) at a single gas multiplier of 1.0,
#   which reproduces the pre-scenario-grid behaviour.
#
# ==============================================================================

function build_scenario_grid(data::Dict)
    # Scale solar/wind CAPEX before any agent is built so every entry point
    # (ADMM, contracts, social planner, merged green) sees the same costs.
    apply_vres_fixed_cost_scale!(data)

    sc  = get(data, "Scenarios", Dict{String,Any}())
    gen = get(data, "General", Dict{String,Any}())

    weather = Int.(get(sc, "weather_years", Int[]))
    if isempty(weather)
        n_fallback = if haskey(data, "ADMM") && haskey(data["ADMM"], "nScenarioYears")
            Int(data["ADMM"]["nScenarioYears"])
        else
            Int(get(gen, "nYears", 1))
        end
        weather = collect(1:n_fallback)
    end

    gas_mult = Float64.(get(sc, "gas_price_multipliers", Float64[]))
    isempty(gas_mult) && (gas_mult = [1.0])

    n_weather = length(weather)
    n_gas     = length(gas_mult)
    n_years   = n_weather * n_gas

    years          = Dict{Int,Int}()
    gas_multiplier = zeros(Float64, n_years)
    weather_index  = zeros(Int, n_years)
    gas_index      = zeros(Int, n_years)

    for iw in 1:n_weather, ig in 1:n_gas
        jy = (iw - 1) * n_gas + ig
        years[jy]          = weather[iw]
        gas_multiplier[jy] = gas_mult[ig]
        weather_index[jy]  = iw
        gas_index[jy]      = ig
    end

    return (
        n_years        = n_years,
        years          = years,
        gas_multiplier = gas_multiplier,
        weather_index  = weather_index,
        gas_index      = gas_index,
        weather_years  = weather,
        gas_levels     = gas_mult,
    )
end

# ------------------------------------------------------------------ fuel ---
#
# Scenario-adjusted commodity price (EUR/MWh_th) and combustion emission factor
# (tCO2/MWh_th) for one fuel. Only gas carries the scenario multiplier: the
# 2×/3× scenarios are a gas-market shock, so coal and biomass are unchanged.

function fuel_price_ef(fuel::AbstractDict, name, gas_mult::Float64)
    f = lowercase(String(name))
    if f == "gas"
        return Float64(get(fuel, "GasPrice", 0.0)) * gas_mult,
               Float64(get(fuel, "GasEmissionFactor", 0.0))
    elseif f == "coal"
        return Float64(get(fuel, "CoalPrice", 0.0)),
               Float64(get(fuel, "CoalEmissionFactor", 0.0))
    elseif f == "biomass"
        return Float64(get(fuel, "BiomassPrice", 0.0)),
               Float64(get(fuel, "BiomassEmissionFactor", 0.0))
    end
    error("Unknown fuel \"$name\"; expected one of Gas, Coal, Biomass (Data/data.yaml → Fuel)")
end

# Short-run marginal cost of a thermal technology in one scenario (EUR/MWh_e):
#
#   SRMC = fuel_price/eta + (emission_factor/eta) * CO2_price + vom
#
# `tech` is one entry of Power.<gen>.StageTechnologies or PeakTechnology and
# must carry `fuel`, `efficiency` and optionally `vom`.
function thermal_srmc(tech::AbstractDict, fuel::AbstractDict, gas_mult::Float64)
    price, ef = fuel_price_ef(fuel, get(tech, "fuel", ""), gas_mult)
    eta = Float64(get(tech, "efficiency", 0.0))
    eta > 0 || error("Technology $(get(tech, "name", "?")) needs a positive efficiency")
    return price / eta + (ef / eta) * Float64(get(fuel, "CO2Price", 0.0)) +
           Float64(get(tech, "vom", 0.0))
end

# Per-scenario gas multiplier vector, defaulting to no shock. Any file that
# derives a fuel-linked cost calls this so the fallback is identical everywhere.
function scenario_gas_multipliers(data::AbstractDict, n_years::Int)
    mult = Float64.(get(data, "GasPriceMultiplier", ones(n_years)))
    length(mult) == n_years ||
        error("GasPriceMultiplier has $(length(mult)) entries but the run has $n_years scenarios")
    return mult
end

# Multiply Gen_VRES_* FixedCost_per_MW by General.VRES_FixedCost_Scale (default 1.0).
# Idempotent: the unscaled yaml value is stored on first call and reused, so a
# second build_scenario_grid on the same dict cannot double-scale.
function apply_vres_fixed_cost_scale!(data::Dict)
    gen = get(data, "General", Dict{String,Any}())
    scale = Float64(get(gen, "VRES_FixedCost_Scale", 1.0))
    power = get(data, "Power", nothing)
    power isa AbstractDict || return nothing

    applied = String[]
    for id in sort!(collect(keys(power)); by = string)
        block = power[id]
        block isa AbstractDict || continue
        String(get(block, "Type", "")) == "VRES" || continue
        haskey(block, "FixedCost_per_MW") || continue
        orig = Float64(get(block, "FixedCost_per_MW_unscaled", block["FixedCost_per_MW"]))
        block["FixedCost_per_MW_unscaled"] = orig
        block["FixedCost_per_MW"] = orig * scale
        push!(applied, "$(id) $(orig) → $(orig * scale) €/MW-year")
    end
    if !isempty(applied) && abs(scale - 1.0) > 1e-12
        println("VRES_FixedCost_Scale = $scale applied: " * join(applied, "; "))
    end
    return nothing
end

# Human-readable one-line summary, printed by the entry points at start-up so a
# run's log always records which uncertainty set it optimised over.
function describe_scenario_grid(scen)
    println("Scenario grid: $(length(scen.weather_years)) weather year(s) " *
            "$(scen.weather_years) x $(length(scen.gas_levels)) gas level(s) " *
            "$(scen.gas_levels) = $(scen.n_years) scenarios (equal probability " *
            "$(round(1 / scen.n_years, digits = 4)) each)")
    return nothing
end

# Print γ/β at start-up and warn if a risk-averse run has a tail narrower than
# one scenario (CVaR then collapses onto the single worst outcome).
function describe_risk_parameters(data::Dict, n_years::Int)
    admm = get(data, "ADMM", Dict{String,Any}())
    gamma = Float64(get(admm, "gamma", 1.0))
    beta  = Float64(get(admm, "beta", 0.95))
    println("Risk parameters: gamma = $gamma, beta = $beta " *
            "(tail = worst $(round(100 * (1 - beta); digits = 1))% of scenarios)")
    if gamma < 1.0 - 1e-12 && (1 - beta) < 1 / n_years - 1e-9
        @warn "ADMM.beta = $beta requests a tail narrower than one of $n_years " *
              "equiprobable scenarios. CVaR will collapse onto the single worst " *
              "scenario. For a tail of k scenarios use beta = 1 - k/$n_years " *
              "(e.g. $(round(1 - 3 / n_years; digits = 3)) for k = 3)."
    end
    return nothing
end
