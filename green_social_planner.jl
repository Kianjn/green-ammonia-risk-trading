# ==============================================================================
# GREEN SOCIAL PLANNER — ME-style ADMM with merged VRES + H₂ + offtaker chain
# ==============================================================================
#
# Same ADMM loop as market_exposure.jl. Solar, wind, Prod_H2_Green, and
# Offtaker_Green are replaced by one merged agent (Green_Coalition).
#
# HOW TO RUN:  julia green_social_planner.jl
# OUTPUT:      green_social_planner_results/
#
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)

using JuMP
using Gurobi
using DataFrames
using CSV
using YAML
using ProgressBars
using Printf
using TimerOutputs
using Statistics

const GUROBI_ENV = Gurobi.Env()
const home_dir = @__DIR__
const PLANNER_KEY = "Green"
const results_dir = joinpath(home_dir, "green_social_planner_results")

include(joinpath(home_dir, "Source", "define_scenarios.jl"))
include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))
include(joinpath(home_dir, "Source", "merged_agent_setup.jl"))
include(joinpath(home_dir, "Source", "define_merged_agent.jl"))
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_merged_agent.jl"))
include(joinpath(home_dir, "Source", "solve_merged_agent.jl"))
include(joinpath(home_dir, "Source", "define_results.jl"))
include(joinpath(home_dir, "Source", "ADMM.jl"))
include(joinpath(home_dir, "Source", "ADMM_subroutine.jl"))
include(joinpath(home_dir, "Source", "solve_power_agent.jl"))
include(joinpath(home_dir, "Source", "solve_H2_agent.jl"))
include(joinpath(home_dir, "Source", "solve_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "solve_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "solve_EP_demand_agent.jl"))
include(joinpath(home_dir, "Source", "update_rho.jl"))
include(joinpath(home_dir, "Source", "compute_agent_objective.jl"))
include(joinpath(home_dir, "Source", "save_results.jl"))

# ── Data ──────────────────────────────────────────────────────────────────────

data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))
ts = Dict()
repr_days = Dict()
gen  = data["General"]
scen = build_scenario_grid(data)
n_years = scen.n_years
years   = scen.years
run_general = merge(gen, Dict(
    "nYears"             => n_years,
    "Fuel"               => get(data, "Fuel", Dict{String,Any}()),
    "GasPriceMultiplier" => scen.gas_multiplier,
))
data_run = copy(data)
data_run["General"] = run_general
describe_scenario_grid(scen)
describe_risk_parameters(data, n_years)

for y in unique(values(years))
    ts[y] = CSV.read(joinpath(home_dir, "Input", "timeseries_$(y).csv"), DataFrame)
    repr_days[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "decision_variables_short.csv"), delim=",", DataFrame)
end

isdir(results_dir) || mkdir(results_dir)

# ── Agents ────────────────────────────────────────────────────────────────────

mdict = Dict{String, Model}()
agents, merged_id, merged_type, member_ids =
    setup_merged_agents!(data_run, PLANNER_KEY, mdict, Gurobi.Optimizer)

for m in values(mdict)
    set_silent(m)
end

elec_market = Dict{String, Any}()
H2_market = Dict{String, Any}()
elec_GC_market = Dict{String, Any}()
H2_GC_market = Dict{String, Any}()
EP_market = Dict{String, Any}()
elec_market["nAgents"] = 0
H2_market["nAgents"] = 0
elec_GC_market["nAgents"] = 0
H2_GC_market["nAgents"] = 0
EP_market["nAgents"] = 0

define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts, repr_days)

member_data = merged_member_data(data, PLANNER_KEY)
define_merged_agent!(merged_id, mdict[merged_id], merged_type, data_run, ts, repr_days, agents, member_data)

for m in agents[:power]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Power"][m], data["ADMM"]), ts, repr_days, agents)
    define_power_parameters!(m, mdict[m], merge(run_general, data["Power"][m]), ts, repr_days)
end
for m in agents[:H2]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m], data["ADMM"]), ts, repr_days, agents)
    define_H2_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m]), ts, repr_days)
end
for m in agents[:offtaker]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m], data["ADMM"]), ts, repr_days, agents)
    define_offtaker_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m]), ts, repr_days)
end
for m in agents[:elec_GC_demand]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m], data["ADMM"]), ts, repr_days, agents)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m]), ts, repr_days)
end

agents[:cap_agents] = [m for m in agents[:all] if haskey(mdict[m].ext[:parameters], :z_cap)]
elec_market["nAgents"] = length(agents[:elec_market])
H2_market["nAgents"] = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])
EP_market["nAgents"] = length(agents[:EP_market])

# ── Build models ────────────────────────────────────────────────────────────

build_merged_agent!(merged_id, mdict[merged_id])
for m in agents[:power]
    build_power_agent!(m, mdict[m], elec_market, elec_GC_market)
end
for m in agents[:H2]
    build_H2_agent!(m, mdict[m], H2_market, H2_GC_market)
end
for m in agents[:offtaker]
    build_offtaker_agent!(m, mdict[m], EP_market, H2_market, H2_GC_market)
end
for m in agents[:elec_GC_demand]
    build_elec_GC_demand_agent!(m, mdict[m], elec_GC_market)
end

# ── SP primal + capacity warm-start (risk-neutral only; Alessio otherwise) ─

n_cap_warmstart = 0
sp_cap_file = joinpath(home_dir, "social_planner_results", "SP_Capacities.csv")
sp_primal_file = joinpath(home_dir, "social_planner_results", "SP_Primal_Quantities.csv")
n_ts = run_general["nTimesteps"]
n_rd = run_general["nReprDays"]
n_yr = run_general["nYears"]
rn_admm = admm_is_risk_neutral(data)

if rn_admm && isfile(sp_primal_file)
    try
        sp_primal_df = CSV.read(sp_primal_file, DataFrame)
        merged_operational_warmstart!(mdict[merged_id], sp_primal_df, member_ids, n_ts, n_rd, n_yr)
    catch e
        @warn "Could not apply SP primal warm-start to merged agent: $e"
    end
end

if rn_admm && isfile(sp_cap_file)
    try
        sp_cap_df = CSV.read(sp_cap_file, DataFrame)
        merged_cap_warmstart!(mdict[merged_id], sp_cap_df, member_ids)
        global n_cap_warmstart += 1
        for m in agents[:cap_agents]
            m == merged_id && continue
            mod = mdict[m]
            agent_type = String(get(mod.ext[:parameters], :Type, ""))
            if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
                row = sp_cap_df[sp_cap_df.AgentID .== m, :]
                cap_val = _sp_cap_scalar(row)
                if cap_val !== nothing
                    vres_cap_inv_warmstart!(mod, cap_val)
                    global n_cap_warmstart += 1
                end
            end
        end
    catch e
        @warn "Could not load SP capacities ($sp_cap_file): $e"
    end
end

# ── ADMM ─────────────────────────────────────────────────────────────────────

results = Dict()
ADMM = Dict()
TO = TimerOutput()

sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
define_results!(merge(run_general, data["ADMM"]), results, ADMM, agents,
    elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market;
    sp_prices_file=sp_prices_file,
    sp_primal_file=rn_admm ? sp_primal_file : "",
    sp_cap_file=rn_admm ? sp_cap_file : "",
    use_primal_warmstart=rn_admm)

results["Cap_Merged"] = get!(results, "Cap_Merged", Dict(m => [] for m in agents[:merged]))
results["Inv_Merged"] = get!(results, "Inv_Merged", Dict(m => [] for m in agents[:merged]))
results["Merged_z_flow"] = get!(results, "Merged_z_flow", Dict(m => [] for m in agents[:merged]))
results["Cap_Merged_slots"] = get!(results, "Cap_Merged_slots", Dict{String, Vector{String}}())
results["Cap_Merged_floors"] = get!(results, "Cap_Merged_floors", Dict{String, Vector{Float64}}())
if haskey(ADMM, "Capacity") && !isempty(ADMM["Capacity"]["z"][merged_id])
    z0 = ADMM["Capacity"]["z"][merged_id][end]
    λ0 = ADMM["Capacity"]["λ"][merged_id][end]
    nslot = length(mdict[merged_id].ext[:parameters][:z_cap])
    mdict[merged_id].ext[:parameters][:z_cap] = z0 isa AbstractVector && length(z0) == nslot ? Float64.(z0) : fill(_cap_scalar(z0), nslot)
    mdict[merged_id].ext[:parameters][:λ_cap] = λ0 isa AbstractVector && length(λ0) == nslot ? Float64.(λ0) : fill(_cap_scalar(λ0), nslot)
    results["Cap_Merged_slots"][merged_id] = String.(mdict[merged_id].ext[:parameters][:cap_slots])
    results["Cap_Merged_floors"][merged_id] = merged_cap_floors(mdict[merged_id])
end

@info "Green social planner" merged_agent=merged_id members=member_ids
ws = results["warmstart"]
parts = String[]
ws["λ"] && push!(parts, "λ from SP prices")
ws["primal"] && push!(parts, "primal quantities from SP")
n_cap_warmstart > 0 && push!(parts, "capacity seeds for $n_cap_warmstart agents")
!isempty(parts) && @info "ADMM warm-start: $(join(parts, ", "))"

ADMM!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, mdict, agents, data_run, TO)
ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

Base.invokelatest(save_results, mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, ADMM, results, agents;
    results_dir=results_dir, case_label="green_social_planner")
YAML.write_file(joinpath(results_dir, "TimerOutput.yaml"), TO)
