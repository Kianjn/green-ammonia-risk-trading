# ==============================================================================
# MARKET EXPOSURE SCRIPT: MULTI-AGENT ENERGY MARKET SIMULATION USING ADMM
# By Kian Jafarinejad - PhD Researcher at TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
#
# PURPOSE:
#   Entry point for the market-exposure multi-agent energy system optimization.
#   Loads configuration and time series, builds per-agent JuMP models, runs the
#   ADMM coordination loop, and writes results to the "market_exposure_results"
#   folder.
#
# HOW TO RUN:
#   From the project root:  julia market_exposure.jl
#
# FLOW:
#   1. Environment and packages
#   2. Load Source/*.jl (parameter definitions, model builders, ADMM, solve, save)
#   3. Load Data/data.yaml and Input (timeseries, representative days)
#   4. Initialize agents dict and JuMP models (mdict)
#   5. Define market parameter dicts and agent parameters (incl. market lists)
#   6. Build optimization models for each agent
#   7. Initialize results/ADMM state; run ADMM! loop
#   8. Save CSVs and timer profile
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

using Pkg
# Activate the project environment in the same directory as this script so that
# Project.toml / Manifest.toml dictate package versions.
Pkg.activate(@__DIR__)

# ------------------------------------------------------------------------------
# SECTION 2: PACKAGE LOADING
# ------------------------------------------------------------------------------

using JuMP
# JuMP: algebraic modeling for optimization; we use it to build each agent's
# objective, variables, and constraints.

using Gurobi
# Gurobi: solver used for all agent subproblems (QP). Must be installed and
# licensed.

using DataFrames
# Tabular data; used when reading CSVs (timeseries, representative days) and
# when writing result DataFrames to CSV.

using CSV
# Read/write CSV files (inputs and output CSVs in market_exposure_results).

using YAML
# Parse data.yaml: General, ADMM, market blocks, and agent blocks (Power,
# Hydrogen, Hydrogen_Offtaker, Electricity_GC_Demand, etc.).

using DataStructures
# Optional extra collections; available if needed elsewhere.

using ProgressBars
# Progress bar over ADMM iterations (used in ADMM.jl).

using Printf
# Formatted strings (e.g. @sprintf); used in ADMM for progress or diagnostics
# if enabled.

using TimerOutputs
# Time the main sections (e.g. imbalances, residuals, price updates) inside
# the ADMM loop.

using ArgParse
# Parse command-line arguments; available for future CLI options.

using Statistics
# mean, etc., used in ADMM diagnostics (price and imbalance means).

using Base.Threads: @spawn
# For potential parallel agent solves; currently we use a single-threaded loop.

using Base: split
# String utility; available if needed.

# Single shared Gurobi environment for the entire process. WHY: each Gurobi
# Env consumes a license token; sharing one Env across all agent models avoids
# acquiring multiple licenses and significantly reduces solver-startup overhead.
const GUROBI_ENV = Gurobi.Env()

# ------------------------------------------------------------------------------
# SECTION 3: DIRECTORY SETUP
# ------------------------------------------------------------------------------

# Root directory of the project; all paths (Data/, Input/, Source/, results) are
# built from this.
const home_dir = @__DIR__
const results_dir = joinpath(home_dir, "market_exposure_results")

# ------------------------------------------------------------------------------
# SECTION 4: FUNCTION LOADING (SOURCE FILES)
# ------------------------------------------------------------------------------

# Parameter definition: attach to each agent model sets, weights, ADMM arrays,
# and market participation flags; fill agent-specific parameters and timeseries.
include(joinpath(home_dir, "Source", "define_scenarios.jl"))
include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))

# Market definitions: initial prices, rho, and for EP_market the fixed demand
# profile D_EP.
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))

# Model building: create JuMP variables, constraints, and objective expressions
# for each agent type (no solve yet).
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))

# ADMM and solving: result buffers, main loop, per-agent subroutine, solve
# wrappers, rho update, and CSV export.
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

# ------------------------------------------------------------------------------
# SECTION 5: DATA LOADING
# ------------------------------------------------------------------------------

# Load the single configuration file: General (nTimesteps, nReprDays, nYears,
# base_year), ADMM (max_iter, epsilon, rho_initial), per-market blocks
# (initial_price, rho_initial; EP_market also Demand_Column, Total_Demand), and
# per-agent blocks under Power, Hydrogen, Hydrogen_Offtaker, Electricity_GC_Demand.
data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))

# Time series: keyed by scenario label (1..nYears). Each value is a DataFrame with
# columns such as SOLAR, LOAD_E, LOAD_H, LOAD_EP (normalized 0–1 profiles).
ts = Dict()

# Ordering matrix: loaded for completeness — it is used by the upstream
# representative-day selection algorithm (outside this script) but is NOT
# directly used in the optimization itself.
order_matrix = Dict() #Can I remove it?

# Representative days: keyed by year. Each value is a DataFrame with columns
# periods (day index 1–365), weights (frequency), selected_periods.
repr_days = Dict()

# Build the scenario grid: every combination of a weather year and a gas-price
# level. years[jy] gives the weather-year FILE LABEL for scenario jy (several jy
# share a label because they differ only in gas price), and the gas multiplier
# rides along in run_general so the cost-defining files can derive fuel-linked
# marginal costs per scenario. See docs/TECHNICAL.md (Uncertainty set).
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

# Time series and representative days, loaded once per distinct weather label.
# Input files follow:
#   Input/timeseries_<label>.csv
#   Input/output_<label>/ordering_variable.csv
#   Input/output_<label>/decision_variables_short.csv
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

# List of agent IDs that belong to the power sector (VRES, conventional, consumer).
agents[:power] = [id for id in keys(data["Power"])]

# List of hydrogen-sector agent IDs (e.g. electrolyzer).
agents[:H2] = [id for id in keys(data["Hydrogen"])]

# List of offtaker agent IDs (green, grey, importer).
agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"])]

# Electricity GC demand agents; empty if the block is missing in data.yaml.
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ? [id for id in keys(data["Electricity_GC_Demand"])] : []

# Union of all agents: used to create one JuMP model per agent and to iterate
# in the ADMM subroutine.
agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand])

# These lists are filled by define_common_parameters! when each agent's type is
# known; they indicate which agents participate in which market (for imbalance
# sums and nAgents).
agents[:elec_market] = []
agents[:H2_market] = []
agents[:elec_GC_market] = []
agents[:H2_GC_market] = []
agents[:EP_market] = []

# One JuMP model per agent; Gurobi is the solver. Keys are agent IDs (strings).
mdict = Dict(i => Model(Gurobi.Optimizer) for i in agents[:all])

# Suppress Gurobi output and apply numerically robust QP defaults (NumericFocus,
# homogeneous barrier, aggressive scaling). See configure_gurobi_agent!.
for m in values(mdict)
    configure_gurobi_agent!(m)
end

# ------------------------------------------------------------------------------
# SECTION 9: MARKET PARAMETER DEFINITION
# ------------------------------------------------------------------------------

elec_market = Dict{String,Any}()
H2_market = Dict{String,Any}()
elec_GC_market = Dict{String,Any}()
H2_GC_market = Dict{String,Any}()
EP_market = Dict{String,Any}()

# Placeholder nAgents counts — all zero at this point because
# define_common_parameters! has not yet been called and agents[:elec_market]
# etc. are still empty lists. These are set here so the Dict keys exist;
# they are overwritten with the correct counts after all agents are defined
# (see Section 10 below).
elec_market["nAgents"] = length(agents[:elec_market])
H2_market["nAgents"] = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])
EP_market["nAgents"] = length(agents[:offtaker])

# Fill market dicts with initial_price, rho_initial, and for EP_market also
# Demand_Column, Total_Demand, and the 3D demand array D_EP.
define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts, repr_days)

# ------------------------------------------------------------------------------
# SECTION 10: AGENT PARAMETER DEFINITION
# ------------------------------------------------------------------------------

for m in agents[:power]
    # Common: sets (JY, JD, JH), weights W, P, γ, β, market flags, ADMM arrays.
    define_common_parameters!(m, mdict[m], merge(run_general, data["Power"][m], data["ADMM"]), ts, repr_days, agents)
    # Power-specific: capacity, profile column, costs, or consumer utility/load.
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

# Agents with endogenous capacity (VRES, H2 producer, GreenOfftaker) for investment consensus.
agents[:cap_agents] = [m for m in agents[:all] if haskey(mdict[m].ext[:parameters], :z_cap)]

# Set market participant counts used in ADMM (consensus denominator n+1).
elec_market["nAgents"]    = length(agents[:elec_market])
H2_market["nAgents"]     = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"]  = length(agents[:H2_GC_market])
EP_market["nAgents"]     = length(agents[:EP_market])

# ------------------------------------------------------------------------------
# SECTION 11: BUILD OPTIMIZATION MODELS
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# SECTION 11b: CAPACITY WARM-START FROM SP (risk-neutral only)
# ------------------------------------------------------------------------------
# RA ME must not inherit the matching-β planner dispatch: that hedge adds green,
# which is the opposite of the private weather-tail response. Alessio starts
# capacity as a free QP variable. Keep SP λ as a price hint.
n_cap_warmstart = 0
sp_cap_file = joinpath(home_dir, "social_planner_results", "SP_Capacities.csv")
rn_admm = admm_is_risk_neutral(data)
if rn_admm && isfile(sp_cap_file)
    try
        sp_cap_df = CSV.read(sp_cap_file, DataFrame)
        jy_set = collect(mdict[agents[:all][1]].ext[:sets][:JY])
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
                if cap_val !== nothing
                    set_start_value(cap_var, cap_val)
                end
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

# Allocate result buffers (per-agent quantity lists, price history, ADMM ρ,
# imbalances, residuals, tolerances) and set initial prices.
# If the social planner has been run, warm-start ADMM λ from its hourly prices;
# otherwise fall back to uniform scalar initial_price from data.yaml.
sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
sp_primal_file = joinpath(home_dir, "social_planner_results", "SP_Primal_Quantities.csv")
# Primal warm-start only when γ=1 (ME = SP). Under RA, g_bar = SP generation
# would pin cap via g ≤ AF·cap to the planner's extra green.
define_results!(merge(run_general, data["ADMM"]), results, ADMM, agents, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market;
    sp_prices_file=sp_prices_file,
    sp_primal_file=rn_admm ? sp_primal_file : "",
    sp_cap_file=rn_admm ? sp_cap_file : "",
    use_primal_warmstart=rn_admm)

# Single consolidated warm-start message
ws = results["warmstart"]
parts = String[]
ws["λ"] && push!(parts, "λ from SP prices")
ws["primal"] && push!(parts, "primal quantities from SP")
n_cap_warmstart > 0 && push!(parts, "capacity seeds for $n_cap_warmstart agents")
if !isempty(parts)
    @info "ADMM warm-start: $(join(parts, ", "))"
end

# Run the coordination loop: each iteration solves all agents, aggregates
# imbalances, updates prices and ρ, and checks convergence.
ADMM!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, mdict, agents, data_run, TO)

# Total ADMM wall-clock time in minutes for human-readable reporting.
# TimerOutputs.tottime returns nanoseconds; multiply by 10^-9 to get
# seconds, then divide by 60 to convert to minutes.
ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ------------------------------------------------------------------------------
# SECTION 13: SAVE RESULTS
# ------------------------------------------------------------------------------

Base.invokelatest(save_results, mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, ADMM, results, agents;
    results_dir=results_dir)

YAML.write_file(joinpath(results_dir, "TimerOutput.yaml"), TO)