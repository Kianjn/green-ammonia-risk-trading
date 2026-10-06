# ==============================================================================
# MARKET EXPOSURE WITH BILATERAL CONTRACTS (PPA + HPA) — ADMM EQUILIBRIUM
# By Kian Jafarinejad - PhD Researcher at TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
#
# PURPOSE:
#   Entry point for the "ME + contracts" institution: the decentralised
#   market-exposure equilibrium of market_exposure.jl, with the three green
#   agents allowed to hedge each other through standard bilateral contracts:
#
#     PPA  VRES  ──►  electrolyzer      fixed price K_ppa (€/MWh_e), agreed
#                                        capacity C_ppa (MW), pay-as-produced
#     HPA  electrolyzer ──► green offtaker   fixed price K_hpa (€/MWh_H2), agreed
#                                        capacity C_hpa (MW_H2), baseload
#
#   Each contract is a private agreement between two firms: the buyer pays the
#   seller K on the contracted volume, both keep dispatching against the spot
#   pools, and the resulting cash flow enters BOTH parties' per-scenario loss
#   and hence both CVaR terms. No third party (no government CfD, no strike
#   subsidy) is involved. Contract capacity is bounded by the parties' plant
#   sizes so a contract is a hedge on real output, not a speculative position.
#
#   The contract price K is an equilibrium outcome: it is the dual of the
#   contract-volume clearing condition (seller's offered MW = buyers' demanded
#   MW), updated by ADMM alongside the five spot prices. The contract volume is
#   likewise chosen by both sides. Results therefore sit between the two
#   benchmarks already available:
#
#     green_social_planner.jl / green_h2_social_planner.jl   complete risk trading
#                                                            inside the green chain
#     me_contracts.jl (this file)                            partial risk trading via
#                                                            fixed-price contracts
#     market_exposure.jl                                     no risk trading
#
#   In the risk-neutral run (gamma = 1) contracts are fairly priced and carry
#   no value, so dispatch, capacities and spot prices reproduce market_exposure.
#
# HOW TO RUN:
#   From the project root:  julia me_contracts.jl
#   Results are written to me_contracts_results/.
#
# FLOW (same as market_exposure.jl, with contract steps marked ★):
#   1. Environment and packages
#   2. Load Source/*.jl
#   3. Load Data/data.yaml and Input (timeseries, representative days)
#   4. Initialize agents dict and JuMP models (mdict)
#   5. Market parameter dicts (five spot markets ★ + PPA/HPA contract markets)
#   6. Agent parameters (★ + contract flags / placeholders, then link parties)
#   7. Build models (★ contract-aware builders add the scalar contract volumes)
#   8. Results/ADMM state (★ + contract prices K, volumes, link residuals)
#   9. Run ADMM_contracts! (★ spot markets + contract links)
#  10. Save CSVs (★ + PPAs.csv, HPAs.csv, cash flows, contract history)
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)

# ------------------------------------------------------------------------------
# SECTION 2: PACKAGE LOADING
# ------------------------------------------------------------------------------

using JuMP
using Gurobi
using DataFrames
using CSV
using YAML
using DataStructures
using ProgressBars
using Printf
using TimerOutputs
using ArgParse
using Statistics
using Base.Threads: @spawn
using Base: split

# Single shared Gurobi environment (one license token, faster model creation).
const GUROBI_ENV = Gurobi.Env()

# ------------------------------------------------------------------------------
# SECTION 3: DIRECTORY SETUP
# ------------------------------------------------------------------------------

const home_dir = @__DIR__
const results_dir = joinpath(home_dir, "me_contracts_results")
const case_label = "me_contracts"

# ------------------------------------------------------------------------------
# SECTION 4: FUNCTION LOADING (SOURCE FILES)
# ------------------------------------------------------------------------------

# Parameter definition (identical to market_exposure.jl).
include(joinpath(home_dir, "Source", "define_scenarios.jl"))
include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))

# Spot-market definitions (identical to market_exposure.jl).
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))

# ★ Contracts: settlement helpers, contract-market dicts, per-agent flags/links.
include(joinpath(home_dir, "Source", "contract_settlement.jl"))
include(joinpath(home_dir, "Source", "define_contract_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_contract_parameters.jl"))

# Model building: base builders plus ★ contract-aware wrappers that add the
# scalar contract-volume variables (C_ppa, C_ppa_buy, C_hpa, C_hpa_buy).
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent_contracts.jl"))

# ADMM and solving: base solves (used by agents without contracts) plus the
# ★ contract-aware solves, loop, subroutine, ρ update and result writer.
include(joinpath(home_dir, "Source", "define_results.jl"))
include(joinpath(home_dir, "Source", "define_results_contracts.jl"))
include(joinpath(home_dir, "Source", "ADMM.jl"))
include(joinpath(home_dir, "Source", "ADMM_subroutine.jl"))
include(joinpath(home_dir, "Source", "ADMM_contracts.jl"))
include(joinpath(home_dir, "Source", "ADMM_subroutine_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_power_agent.jl"))
include(joinpath(home_dir, "Source", "solve_H2_agent.jl"))
include(joinpath(home_dir, "Source", "solve_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "solve_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "solve_EP_demand_agent.jl"))
include(joinpath(home_dir, "Source", "solve_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_offtaker_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "update_rho.jl"))
include(joinpath(home_dir, "Source", "update_rho_contracts.jl"))
include(joinpath(home_dir, "Source", "compute_agent_objective.jl"))
include(joinpath(home_dir, "Source", "save_results.jl"))
include(joinpath(home_dir, "Source", "save_results_contracts.jl"))

# ------------------------------------------------------------------------------
# SECTION 5: DATA LOADING
# ------------------------------------------------------------------------------

data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))

ts = Dict()
order_matrix = Dict()
repr_days = Dict()

# Scenario grid (weather years × gas-price levels), see docs/TECHNICAL.md (Uncertainty set).
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
    order_matrix[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "ordering_variable.csv"), delim=",", DataFrame)
    repr_days[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "decision_variables_short.csv"), delim=",", DataFrame)
end

# ------------------------------------------------------------------------------
# SECTION 6: RESULTS FOLDER
# ------------------------------------------------------------------------------

isdir(results_dir) || mkdir(results_dir)

# ------------------------------------------------------------------------------
# SECTION 7: AGENT INITIALIZATION
# ------------------------------------------------------------------------------

agents = Dict()
agents[:power]          = [id for id in keys(data["Power"])]
agents[:H2]             = [id for id in keys(data["Hydrogen"])]
agents[:offtaker]       = [id for id in keys(data["Hydrogen_Offtaker"])]
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ? [id for id in keys(data["Electricity_GC_Demand"])] : []
agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand])

# Spot-market participant lists (filled by define_common_parameters!).
agents[:elec_market]    = []
agents[:H2_market]      = []
agents[:elec_GC_market] = []
agents[:H2_GC_market]   = []
agents[:EP_market]      = []

# ★ Contract-party lists (filled by define_contract_parameters!):
#   :ppa_vres / :ppa_buyers  — PPA sellers (VRES) / buyers (electrolyzers)
#   :hpa_h2   / :hpa_buyers  — HPA sellers (electrolyzers) / buyers (green offtakers)
agents[:ppa_market] = String[]
agents[:hpa_market] = String[]
agents[:ppa_vres]   = String[]
agents[:ppa_buyers] = String[]
agents[:hpa_h2]     = String[]
agents[:hpa_buyers] = String[]

mdict = Dict(i => Model(Gurobi.Optimizer) for i in agents[:all])
for m in values(mdict)
    configure_gurobi_agent!(m)
end

# ------------------------------------------------------------------------------
# SECTION 9: MARKET PARAMETER DEFINITION
# ------------------------------------------------------------------------------

elec_market    = Dict{String,Any}()
H2_market      = Dict{String,Any}()
elec_GC_market = Dict{String,Any}()
H2_GC_market   = Dict{String,Any}()
EP_market      = Dict{String,Any}()
ppa_market     = Dict{String,Any}()   # ★
hpa_market     = Dict{String,Any}()   # ★

elec_market["nAgents"]    = 0
H2_market["nAgents"]      = 0
elec_GC_market["nAgents"] = 0
H2_GC_market["nAgents"]   = 0
EP_market["nAgents"]      = 0

define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts, repr_days)

# ★ PPA / HPA contract markets: initial K rule ("spot" = fair value) and ρ.
define_contract_market_parameters!(ppa_market, hpa_market, data, agents)

# ------------------------------------------------------------------------------
# SECTION 10: AGENT PARAMETER DEFINITION
# ------------------------------------------------------------------------------

for m in agents[:power]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Power"][m], data["ADMM"]), ts, repr_days, agents)
    define_power_parameters!(m, mdict[m], merge(run_general, data["Power"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], data, agents)          # ★
end

for m in agents[:H2]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m], data["ADMM"]), ts, repr_days, agents)
    define_H2_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], data, agents)          # ★
end

for m in agents[:offtaker]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m], data["ADMM"]), ts, repr_days, agents)
    define_offtaker_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], data, agents)          # ★
end

for m in agents[:elec_GC_demand]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m], data["ADMM"]), ts, repr_days, agents)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m]), ts, repr_days)
end

# Agents with endogenous capacity (VRES, electrolyzer, green offtaker).
agents[:cap_agents] = [m for m in agents[:all] if haskey(mdict[m].ext[:parameters], :z_cap)]

elec_market["nAgents"]    = length(agents[:elec_market])
H2_market["nAgents"]      = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"]   = length(agents[:H2_GC_market])
EP_market["nAgents"]      = length(agents[:EP_market])

# ★ Wire the contract links now that every party is known: party lists in the
# market dicts, consensus denominators, expected MWh per MW-year of each link,
# and the sellers' availability profiles handed to the PPA buyers.
link_contract_parties!(mdict, agents, ppa_market, hpa_market)
@info "Contract links: $(length(ppa_market["ppa_vres"])) PPA (sellers $(join(ppa_market["ppa_vres"], ", ")) → " *
      "buyers $(join(ppa_market["ppa_buyers"], ", "))), " *
      "$(length(hpa_market["hpa_h2"])) HPA (sellers $(join(hpa_market["hpa_h2"], ", ")) → " *
      "buyers $(join(hpa_market["hpa_buyers"], ", ")))"

# ------------------------------------------------------------------------------
# SECTION 11: BUILD OPTIMIZATION MODELS
# ------------------------------------------------------------------------------

for m in agents[:power]
    build_power_agent_contracts!(m, mdict[m], elec_market, elec_GC_market, ppa_market)   # ★
end

for m in agents[:H2]
    build_H2_agent_contracts!(m, mdict[m], H2_market, H2_GC_market, ppa_market)         # ★
end

for m in agents[:offtaker]
    build_offtaker_agent_contracts!(m, mdict[m], EP_market, H2_market, H2_GC_market, hpa_market)   # ★
end

for m in agents[:elec_GC_demand]
    build_elec_GC_demand_agent!(m, mdict[m], elec_GC_market)
end

# ------------------------------------------------------------------------------
# SECTION 11b: CAPACITY WARM-START FROM SP (risk-neutral only, as in ME)
# ------------------------------------------------------------------------------

n_cap_warmstart = 0
sp_cap_file = joinpath(home_dir, "social_planner_results", "SP_Capacities.csv")
rn_admm = admm_is_risk_neutral(data)
if rn_admm && isfile(sp_cap_file)
    try
        sp_cap_df = CSV.read(sp_cap_file, DataFrame)
        for m in agents[:cap_agents]
            mod = mdict[m]
            agent_type = String(get(mod.ext[:parameters], :Type, ""))
            cap_var = nothing
            if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
                cap_var = mod.ext[:variables][:cap_VRES]
            elseif agent_type == "GreenProducer" && haskey(mod.ext[:variables], :cap_H2_y)
                cap_var = mod.ext[:variables][:cap_H2_y]
            elseif agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
                cap_var = mod.ext[:variables][:cap_EP_y]
            end
            if cap_var !== nothing
                row = sp_cap_df[sp_cap_df.AgentID .== m, :]
                cap_val = _sp_cap_scalar(row)
                cap_val === nothing || set_start_value(cap_var, cap_val)
                global n_cap_warmstart += 1
            end
        end
    catch e
        @warn "Could not load SP capacities ($sp_cap_file): $e"
    end
end

# ------------------------------------------------------------------------------
# SECTION 12: RUN ADMM
# ------------------------------------------------------------------------------

results = Dict()
ADMM = Dict()
TO = TimerOutput()

# Warm start exactly as in market_exposure.jl: SP λ always; SP primal and
# capacity state only when risk-neutral (RA must find its own dispatch).
# ★ Contract prices K start at the fair value implied by the λ warm start
#   (or at the numeric `initial_price` in data.yaml); contract volumes start at 0.
sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
sp_primal_file = joinpath(home_dir, "social_planner_results", "SP_Primal_Quantities.csv")
define_results_contracts!(merge(run_general, data["ADMM"]), results, ADMM, agents,
    elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market,
    ppa_market, hpa_market, mdict;
    sp_prices_file = sp_prices_file,
    sp_primal_file = rn_admm ? sp_primal_file : "",
    sp_cap_file    = rn_admm ? sp_cap_file : "",
    use_primal_warmstart = rn_admm)

ws = results["warmstart"]
parts = String[]
ws["λ"] && push!(parts, "λ from SP prices")
ws["primal"] && push!(parts, "primal quantities from SP")
n_cap_warmstart > 0 && push!(parts, "capacity seeds for $n_cap_warmstart agents")
for v in ppa_market["ppa_vres"]
    push!(parts, @sprintf("K_ppa[%s] = %.2f €/MWh", v, results["K_ppa"][v][1]))
end
for h in hpa_market["hpa_h2"]
    push!(parts, @sprintf("K_hpa[%s] = %.2f €/MWh", h, results["K_hpa"][h][1]))
end
isempty(parts) || @info "ADMM warm-start: $(join(parts, ", "))"

# ★ Spot markets + contract links.
ADMM_contracts!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market,
                ppa_market, hpa_market, mdict, agents, data_run, TO)

ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ------------------------------------------------------------------------------
# SECTION 13: SAVE RESULTS
# ------------------------------------------------------------------------------

Base.invokelatest(save_results_contracts!, mdict, elec_market, H2_market, elec_GC_market, H2_GC_market,
    ppa_market, hpa_market, ADMM, results, agents;
    results_dir = results_dir, case_label = case_label)

YAML.write_file(joinpath(results_dir, "TimerOutput.yaml"), TO)
