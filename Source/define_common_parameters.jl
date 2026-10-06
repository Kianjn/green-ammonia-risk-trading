# ==============================================================================
# define_common_parameters.jl — Common setup for every agent model
# ==============================================================================
#
# PURPOSE:
#   Called once per agent during initialization. It (1) creates the standard
#   index sets (years, representative days, hours) and stores them on the
#   JuMP model; (2) builds the weight matrix W and scenario probabilities P;
#   (3) sets risk-related parameters γ and β if present in data; (4) determines
#   which markets this agent participates in from its Type and pushes the agent
#   ID into the corresponding lists (agents[:elec_market], etc.); (5) allocates
#   ADMM placeholder arrays (λ, g_bar) and scalar ρ for each market so that
#   ADMM_subroutine can overwrite them each iteration without reallocating.
#
# ARGUMENTS:
#   m         — Agent ID (string).
#   mod       — The JuMP model for this agent (we write into mod.ext).
#   data      — Merged dict: General + ADMM + agent-specific block from data.yaml.
#   ts        — Dict keyed by year; values are DataFrames of hourly time series.
#   repr_days — Dict keyed by year; values are DataFrames with :periods, :weights.
#   agents    — Dict of agent lists; we push! m into the relevant market lists.
#
# ==============================================================================

function define_common_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict, agents::Dict)
    # --- Storage on the JuMP model ---
    # Six separate dictionaries keep the model's data cleanly separated by role:
    #   :sets        — Index ranges (JY, JD, JH) that define the model dimensions.
    #   :parameters  — Scalar/array constants (capacities, costs, ADMM prices, etc.).
    #   :timeseries  — 3D hourly profiles built from CSV data (AF, LOAD_E, …).
    #   :variables   — JuMP decision variables (filled later by build_*_agent!).
    #   :constraints — JuMP constraints (filled later by build_*_agent!).
    #   :expressions — JuMP expressions like objective terms (filled by build_*_agent!).
    # Separating them avoids name collisions and lets downstream code (build_*,
    # solve_*, ADMM_subroutine) access exactly the namespace it needs.
    mod.ext[:sets]        = Dict{Symbol,Any}()
    mod.ext[:parameters]  = Dict{Symbol,Any}()
    mod.ext[:timeseries]  = Dict{Symbol,Any}()
    mod.ext[:variables]   = Dict{Symbol,Any}()
    mod.ext[:constraints] = Dict{Symbol,Any}()
    mod.ext[:expressions] = Dict{Symbol,Any}()

    # --- Dimensions from config ---
    n_years     = data["nYears"]      # Scenarios = weather years x gas levels (15 by default).
    n_repr_days = data["nReprDays"]   # Number of representative days (8).
    n_timesteps = data["nTimesteps"]  # Hours per representative day (24).

    # --- Index sets ---
    JY = 1:n_years
    JD = 1:n_repr_days
    JH = 1:n_timesteps

    mod.ext[:sets][:JY] = JY
    mod.ext[:sets][:JD] = JD
    mod.ext[:sets][:JH] = JH

    # --- Representative-day weights ---
    # W[jd, jy] = how many real calendar days representative day jd stands for in
    # scenario jy. This weight scales per-representative-day objective values up
    # to a full-year total (e.g. sum over jh,jd,jy of W[jd,jy] * cost[jh,jd,jy]).
    #
    # Main.years maps the flat scenario index jy to a WEATHER FILE LABEL (the key
    # for timeseries_<label>.csv). It is NOT the identity: with the default 5x3
    # grid, jy = 1..15 maps onto labels 1..5, so three consecutive jy share one
    # label and differ only in gas price (see define_scenarios.jl).
    # Fallback Dict(1 => 1) is used only if Main.years is not defined.
    _years = isdefined(Main, :years) ? Main.years : Dict(1 => 1)
    # Build the 2D weight matrix: for each (jd, jy), look up the scenario label
    # from _years, then pull the weight of that representative day from repr_days.
    W = [repr_days[_years[jy]][!, :weights][jd] for jd in JD, jy in JY]
    mod.ext[:parameters][:W] = W

    # --- Scenario probabilities and risk parameters ---
    # P[jy] = probability of year-scenario jy occurring. With nYears=1, P=[1.0].
    # For multi-scenario runs (nYears>1), we assume a uniform prior unless
    # overridden in data.yaml and use P in CVaR risk terms for green agents.
    P = ones(n_years) ./ n_years
    mod.ext[:parameters][:P] = P
    # γ (gamma): per-agent risk weight in the objective. γ=1 gives a strictly
    #   risk-neutral agent; γ<1 increases the weight on the CVaR(loss) term for
    #   agents that implement risk aversion (currently only VRES, electrolyzer,
    #   and GreenOfftaker read and use γ explicitly).
    # β (beta): CVaR confidence level τ (e.g. 0.95 = worst 5% of scenarios).
    # All agents receive γ, β here for convenience; build_* files decide
    # whether to actually use them in the objective.
    mod.ext[:parameters][:γ] = get(data, "gamma", 1.0)
    mod.ext[:parameters][:β] = get(data, "beta", 0.95)

    # --- Agent type and market participation ---
    # Type comes from the agent block in data.yaml (e.g. "VRES", "Consumer").
    agent_type = String(get(data, "Type", ""))
    # Store on the model so build_*_agent! functions can branch on it.
    mod.ext[:parameters][:Type] = agent_type

    # Which markets this agent trades in. Used both to push m into the right
    # agent-list and to set boolean flags on the model so ADMM_subroutine and
    # solve_* know which price/quantity arrays to read/write.
    #
    # Economic rationale for each mapping:
    #   elec:    Generators (VRES, Conv) sell electricity; Consumer buys; GreenProducer
    #            buys electricity to run the electrolyzer → all participate in elec.
    #   H2:      GreenProducer sells H2; GreenOfftaker buys H2 → bilateral H2 market.
    #   elec_GC: VRES produces electricity GCs; GreenProducer can trade GCs; GC_Demand
    #            buys GCs to satisfy renewable obligations.
    #   H2_GC:   GreenProducer creates H2 GCs; GreenOfftaker & GreyOfftaker need GCs
    #            to certify their end product as (partially) green.
    #   EP:      GreenOfftaker, GreyOfftaker, and EPImporter supply end product to
    #            meet the fixed EP demand; they compete on cost in the EP market.
    in_elec    = agent_type in ("VRES", "Conventional", "Consumer", "GreenProducer")
    in_H2      = agent_type in ("GreenProducer", "GreenOfftaker")
    in_elec_GC = agent_type in ("VRES", "GreenProducer", "GC_Demand")
    in_H2_GC   = agent_type in ("GreenProducer", "GreenOfftaker", "GreyOfftaker")
    in_EP      = agent_type in ("GreenOfftaker", "GreyOfftaker", "EPImporter")

    if in_elec    push!(agents[:elec_market], m) end
    if in_H2      push!(agents[:H2_market], m) end
    if in_elec_GC push!(agents[:elec_GC_market], m) end
    if in_H2_GC   push!(agents[:H2_GC_market], m) end
    if in_EP      push!(agents[:EP_market], m) end

    mod.ext[:parameters][:in_elec_market]    = in_elec
    mod.ext[:parameters][:in_H2_market]      = in_H2
    mod.ext[:parameters][:in_elec_GC_market] = in_elec_GC
    mod.ext[:parameters][:in_H2_GC_market]   = in_H2_GC
    mod.ext[:parameters][:in_EP_market]      = in_EP

    # --- ADMM placeholders (overwritten each iteration in ADMM_subroutine) ---
    # Pre-allocated so ADMM_subroutine can overwrite each iteration without
    # reallocating memory. Zeros are the initial values before the first ADMM
    # iteration; they will be replaced by actual prices / consensus terms.
    #   λ_*     = Lagrange multiplier (price) array [jh, jd, jy] for market *.
    #   g_bar_* = Consensus / penalty centre for market * (mean of all agents'
    #             net positions; used in the augmented-Lagrangian penalty term).
    #   ρ_*     = Scalar penalty weight for market * (updated by update_rho!).
    # All markets get the same 3D shape so we can broadcast element-wise in
    # price and consensus updates inside ADMM.jl.
    shp = (n_timesteps, n_repr_days, n_years)

    # Electricity market ADMM arrays
    mod.ext[:parameters][:λ_elec]     = zeros(shp)
    mod.ext[:parameters][:g_bar_elec] = zeros(shp)
    mod.ext[:parameters][:ρ_elec]     = 1.0

    # Hydrogen market ADMM arrays
    mod.ext[:parameters][:λ_H2]      = zeros(shp)
    mod.ext[:parameters][:g_bar_H2]   = zeros(shp)
    mod.ext[:parameters][:ρ_H2]       = 1.0

    # Electricity GC market ADMM arrays
    mod.ext[:parameters][:λ_elec_GC]     = zeros(shp)
    mod.ext[:parameters][:g_bar_elec_GC]  = zeros(shp)
    mod.ext[:parameters][:ρ_elec_GC]     = 1.0

    # Hydrogen GC market ADMM arrays
    mod.ext[:parameters][:λ_H2_GC]     = zeros(shp)
    mod.ext[:parameters][:g_bar_H2_GC]  = zeros(shp)
    mod.ext[:parameters][:ρ_H2_GC]      = 1.0

    # End-product market ADMM arrays
    mod.ext[:parameters][:λ_EP]     = zeros(shp)
    mod.ext[:parameters][:g_bar_EP]  = zeros(shp)
    mod.ext[:parameters][:ρ_EP]      = 1.0

    # Investment consensus: agents with endogenous capacity (VRES, H2 producer,
    # GreenOfftaker) participate in a PER-AGENT ADMM EQUALITY SPLIT for
    # capacity. For each such agent m and year y we maintain three local
    # parameters that ADMM_subroutine refreshes every iteration:
    #
    #   :z_cap[y]   — auxiliary capacity target derived from flow consensus
    #                 (this is the analogue of g_bar for the capacity split)
    #   :λ_cap[y]   — Lagrange multiplier for the equality x_cap = z_cap
    #   :ρ_cap     — scalar penalty weight for this agent
    #
    # The agent objective then adds, per year:
    #   λ_cap[y] · (x_cap[y] - z_cap[y]) + (ρ_cap/2) · (x_cap[y] - z_cap[y])²
    #
    # WHY the equality split (vs. a pure quadratic penalty against a derived
    # target):
    #   * The dual λ_cap provides the missing first-order force that drives
    #     x_cap towards z_cap exactly in the limit. With only a quadratic
    #     penalty, once ρ_cap saturates the CAPEX gradient dominates and
    #     leaves a persistent residual; convergence stalls regardless of how
    #     smart the ρ controller is.
    #   * Per-agent ρ_cap lets the controller specialise per agent type
    #     (VRES vs electrolyzer vs green offtaker) instead of compromising
    #     on a single global value. See update_rho.jl and docs/TECHNICAL.md (ADMM).
    #
    # All three parameters are initialised here as zeros / `rho_cap_initial`;
    # ADMM_subroutine overwrites them each iteration from ADMM["Capacity"][m].
    rho_cap = get(data, "rho_cap_initial", 0.1)
    if agent_type in ("VRES", "GreenProducer", "GreenOfftaker")
        mod.ext[:parameters][:z_cap]  = 0.0
        mod.ext[:parameters][:λ_cap]  = 0.0
        mod.ext[:parameters][:ρ_cap]  = rho_cap
    end

    return mod, agents
end
