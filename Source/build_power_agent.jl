# ==============================================================================
# build_power_agent.jl — JuMP model for power-sector agents
# ==============================================================================
#
# PURPOSE:
#   Constructs variables, constraints, and objective for one power-sector agent
#   (VRES, Conventional, or Consumer) in the ADMM formulation, and adds the
#   corresponding block to the social planner model.
#
#   - VRES: variable renewable generator with endogenous yearly capacity
#     scalar cap_VRES and one-shot investment inv_VRES (same capacity in all
#     weather scenarios). Supplies electricity and
#     electricity GCs 1:1. Net positions: g_net_elec = +g, g_net_elec_GC = +g.
#     ADMM objective = generation cost − elec revenue − GC revenue
#     + ADMM penalties + fixed annualised CAPEX on cap_VRES + optional CVaR
#     term (γ·β_VRES). Capacity constraint: g ≤ AF × cap_VRES (non-anticipative).
#
#   - Conventional: dispatchable generator with fixed Capacity; either a flat
#     single-technology plant (constant SRMC from Fuel block) or a legacy 3-stage
#     increasing-cost stack when StageTechnologies is set. Net position:
#     g_net_elec = +g. ADMM objective = generation cost − elec revenue + ADMM penalties.
#
#   - Consumer: elastic electricity demand with quadratic utility. Net position:
#     g_net_elec = −d. ADMM objective = expenditure − utility + ADMM penalty.
#
#   No solve is performed here; solve_power_agent! re-sets the objective with
#   current λ/ρ/g_bar (and risk variables) and calls optimize!.
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model (parameters and sets already set by define_*_parameters!).
#   elec_market, elec_GC_market — Not read here; nAgents etc. used in ADMM.
#
# ==============================================================================

function build_power_agent!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict)
    # ── Index sets ────────────────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W   = mod.ext[:parameters][:W]    # W[jd,jy] = weight (number of real days
                                      #   represented by this representative day)

    # ── ADMM parameters for the ELECTRICITY market ────────────────────────
    λ_elec     = mod.ext[:parameters][:λ_elec]       # λ = ADMM price (Lagrange
                                      # multiplier for the electricity market-
                                      # clearing constraint, updated each ADMM
                                      # iteration)
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]   # ḡ = consensus target
                                      # (average of all agents' net positions
                                      # in the electricity market; agents are
                                      # driven toward this shared schedule)
    ρ_elec     = mod.ext[:parameters][:ρ_elec]       # ρ = ADMM penalty weight
                                      # (quadratic coefficient that drives each
                                      # agent's position toward the consensus ḡ;
                                      # larger ρ → faster convergence but harder
                                      # sub-problems)

    # ── ADMM parameters for the ELECTRICITY-GC market ─────────────────────
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]        # λ_GC = Lagrange
                                          # multiplier for the elec-GC market
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]   # ḡ_GC = consensus
                                               # target in the elec-GC market
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]            # ρ_GC = penalty
                                                  # weight for elec-GC market
    agent_type = mod.ext[:parameters][:Type]

    if agent_type == "VRES"
        cap_initial = mod.ext[:parameters][:Capacity]   # initial installed capacity (MW in first model year)
        AF  = mod.ext[:timeseries][:AF]          # AF[jh,jd,jy] = hour-specific
                                                 #   availability factor (0–1);
                                                 #   reflects wind/solar resource
        MC  = mod.ext[:parameters][:MarginalCost]  # marginal cost (€/MWh)
        # Annualised fixed investment cost per MW of installed capacity (€/MW-year).
        # Default 0.0 preserves original behaviour if not provided.
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        # Risk parameters (CVaR skeleton; γ = 0 ⇒ risk-neutral by default).
        gamma = get(mod.ext[:parameters], :γ, 1.0)
        beta_conf = get(mod.ext[:parameters], :β, 0.95)   # confidence level β
        P = mod.ext[:parameters][:P]
        n_years = length(JY)   # apply the annualised charge once per model year

        # ── Capacity and investment (single decision, all weather scenarios) ───
        # cap_VRES = installed VRES capacity (MW), chosen once before weather is known.
        # inv_VRES = new capacity investment (MW) in this decision year.
        cap_VRES = mod.ext[:variables][:cap_VRES] = @variable(mod, lower_bound = 0, base_name = "cap_VRES")
        inv_VRES = mod.ext[:variables][:inv_VRES] = @variable(mod, lower_bound = 0, base_name = "inv_VRES")
        mod.ext[:constraints][:cap_VRES_init] = @constraint(mod, cap_VRES == cap_initial + inv_VRES)

        # Generation variable g ≥ 0 (MWh produced in each hour/day/year).
        g = mod.ext[:variables][:g] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gen")

        # Net positions — VRES output counts as positive supply in BOTH the
        # electricity market AND the electricity-GC market, because all VRES
        # output is inherently renewable-certified (each MWh of wind/solar
        # automatically generates one green certificate).
        mod.ext[:expressions][:g_net_elec]    = @expression(mod, g)   # g_net_elec    = +g
        mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g)   # g_net_elec_GC = +g

        # ── Risk variables (agent-level CVaR) ───────────────────────────────
        # α_VRES: VaR proxy; CVaR_VRES: Conditional Value-at-Risk of loss;
        # u_VRES[jy]: shortfall per scenario year.
        # α and CVaR are unbounded: losses can be negative (profits). Forcing
        # them ≥ 0 would zero-out CVaR in profitable years and invert γ<1.
        alpha_VRES = mod.ext[:variables][:alpha_VRES] = @variable(mod, base_name = "alpha_VRES_$(m)")
        cvar_VRES  = mod.ext[:variables][:CVaR_VRES]  = @variable(mod, base_name = "CVaR_VRES_$(m)")
        u_VRES     = mod.ext[:variables][:u_VRES]     = @variable(mod, [jy in JY], lower_bound = 0, base_name = "u_VRES_$(m)")

        # Per-year economic loss (cost − revenue) excluding ADMM penalties; prices
        # λ are updated each ADMM iteration via mod.ext[:parameters][:λ_*].
        # loss_total = loss_VRES + F_cap·cap: CVaR must use FULL loss so that with
        # nYears=1, changing γ has no effect (SP/ME equivalence).
        loss_VRES = Dict{Int,JuMP.AffExpr}()
        loss_total = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_VRES[jy] = @expression(mod,
                sum(W[jd, jy] * (MC * g[jh, jd, jy]
                    - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                    - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD)
            )
            loss_total[jy] = @expression(mod, loss_VRES[jy] + F_cap * cap_VRES)
        end
        mod.ext[:expressions][:loss_VRES] = loss_VRES

        # CVaR shortfall constraints: u_VRES[jy] ≥ loss_total[jy] − α.
        mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
           u_VRES[jy] >= loss_total[jy] - alpha_VRES
        )

        # CVaR definition: CVaR_VRES ≥ α_VRES + (1/(1−β)) * Σ P[jy]*u_VRES[jy].
        one_minus_beta = max(1e-6, 1.0 - beta_conf)
        mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
            cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

        # Objective:
        #   min  Σ_{h,d,y} W[d,y]·( MC·g − λ_elec·g − λ_GC·g )       ← (1)
        #      + Σ_{h,d,y} (ρ_elec/2)·W[d,y]·(g − ḡ_elec)²           ← (2)
        #      + Σ_{h,d,y} (ρ_GC /2)·W[d,y]·(g − ḡ_GC)²              ← (3)
        #
        # (1) Production cost minus revenue from the electricity market
        #     minus revenue from the elec-GC market.  The agent earns λ_elec
        #     and λ_GC per MWh of generation g.
        # (2) ADMM augmented-Lagrangian penalty pushing g toward the
        #     electricity-market consensus ḡ_elec.
        # (3) ADMM augmented-Lagrangian penalty pushing g toward the
        #     elec-GC-market consensus ḡ_GC.
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * g[jh, jd, jy]
                - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec_GC/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            # Fixed annualised investment cost once (same capacity in every scenario).
            + F_cap * cap_VRES
        )

        # Capacity constraint: generation limited by AF × single installed capacity.
        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], g[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES)

    elseif agent_type == "Conventional"
        cap = mod.ext[:parameters][:Capacity]        # installed capacity (MW)
        MC  = mod.ext[:parameters][:MarginalCost]    # fallback marginal cost (€/MWh)
        MC_y = get(mod.ext[:parameters], :MarginalCostByYear, nothing)

        g = mod.ext[:variables][:g] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gen")
        mod.ext[:expressions][:g_net_elec] = @expression(mod, g)

        if MC_y === nothing && haskey(mod.ext[:parameters], :ConvStageBaseCost)
        # Legacy 3-stage stacked technologies.
        stage_cap = get(mod.ext[:parameters], :ConvStageCap, [cap, 0.0, 0.0])
        stage_base = mod.ext[:parameters][:ConvStageBaseCost]
        stage_slope = get(mod.ext[:parameters], :ConvStageSlope, zeros(3, length(JY)))

        g_stage = mod.ext[:variables][:g_stage] = @variable(mod, [s in 1:3, jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gen_stage")

        mod.ext[:constraints][:stage_balance] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
            g[jh, jd, jy] == sum(g_stage[s, jh, jd, jy] for s in 1:3))
        mod.ext[:constraints][:stage_caps] = @constraint(mod, [s in 1:3, jh in JH, jd in JD, jy in JY],
            g_stage[s, jh, jd, jy] <= stage_cap[s])

        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (
                sum(stage_base[s, jy] * g_stage[s, jh, jd, jy] + 0.5 * stage_slope[s, jy] * g_stage[s, jh, jd, jy]^2 for s in 1:3)
                - λ_elec[jh, jd, jy] * g[jh, jd, jy]
            ) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )
        else
        # Flat single-technology plant: constant SRMC per scenario (merit order via λ_elec).
        if MC_y === nothing
            MC_y = fill(MC, length(JY))
        end
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC_y[jy] * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )
        end

        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], g[jh, jd, jy] <= cap)

    elseif agent_type == "Consumer"
        peak = mod.ext[:parameters][:PeakLoad]   # peak demand (MW)
        A_E  = mod.ext[:parameters][:A_E]         # intercept of inverse demand
        B_E  = mod.ext[:parameters][:B_E]         # slope of inverse demand
        load = mod.ext[:timeseries][:LOAD_E]      # LOAD_E[jh,jd,jy] = normalized
                                                  #   hourly load profile (0–1)

        # Demand variable d ≥ 0 (MWh consumed).
        d = mod.ext[:variables][:d] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "demand")

        # Net position: demand is NEGATIVE supply in the electricity market.
        mod.ext[:expressions][:g_net_elec] = @expression(mod, -d)   # g_net_elec = −d

        # Objective:
        #   min  Σ W·( λ_elec·d  −  U(d) )           ← (1)
        #      + Σ (ρ_elec/2)·W·(−d − ḡ_elec)²       ← (2)
        #
        # U(d) = A_E·d − (B_E/2)·d²  is the quadratic consumer utility
        #   (area under the inverse demand curve).
        # (1) The agent minimizes expenditure (λ·d) minus utility U(d),
        #     which is equivalent to maximizing consumer surplus U(d) − λ·d.
        # (2) ADMM penalty on the net position (−d) toward consensus ḡ_elec.
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

        # Load constraint: demand bounded by peak load × hourly load profile,
        # representing the physical maximum consumption in each hour.
        mod.ext[:constraints][:load] = @constraint(mod, [jh in JH, jd in JD, jy in JY], d[jh, jd, jy] <= peak * load[jh, jd, jy])
    end

    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add power agent block to shared planner model (no ADMM terms).
#
# Unlike the ADMM build above, the planner optimizes ALL agents jointly in a
# single model.  Therefore there are NO ADMM penalty terms: no λ (prices emerge
# as dual variables of market-clearing constraints added elsewhere), no ρ
# (penalty weight), and no ḡ (consensus target).  Each agent contributes only
# its physical constraints and its welfare expression (utility for consumers,
# negative cost for generators) to the planner's objective.
#
# Returns: the agent's contribution to total social welfare as a JuMP expression.
# Side-effect: stores the agent's decision variables in var_dict so that the
# caller can build market-clearing constraints across all agents.
# ------------------------------------------------------------------------------

function add_power_agent_to_planner!(planner::Model, id::String, mod::Model,
                                     var_dict::Dict, W::AbstractArray)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    _p(m, k) = get(m.ext[:parameters], k, nothing)
    _ts(m, k) = m.ext[:timeseries][k]
    agent_type = String(_p(mod, :Type))

    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    if agent_type == "Consumer"
        A_E = _p(mod, :A_E) !== nothing ? _p(mod, :A_E) : 500.0
        B_E = _p(mod, :B_E) !== nothing ? _p(mod, :B_E) : 0.5
        D_bar = _ts(mod, :LOAD_E) .* _p(mod, :PeakLoad)

        d_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_E_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], d_E[jh, jd, jy] <= D_bar[jh, jd, jy])

        # Per-year welfare = consumer utility U(d) = A_E·d − (B_E/2)·d².
        # No expenditure term: market payments are transfers that cancel
        # in the aggregate planner objective. No per-agent CVaR: a single
        # social CVaR is applied in build_social_planner! to the aggregate.
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                sum(W_dict[jy][jd] * ((A_E * d_E[jh, jd, jy]) - 0.5 * B_E * d_E[jh, jd, jy]^2)
                    for jh in JH, jd in JD)
            )
        end
        var_dict[:power_d_E][id] = d_E
        return welfare_per_year

    elseif agent_type == "VRES"
        AF = _ts(mod, :AF)
        C = _p(mod, :MarginalCost)
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        cap_initial = _p(mod, :Capacity)

        cap_VRES = @variable(planner, lower_bound=0, base_name="cap_VRES_$(id)")
        inv_VRES = @variable(planner, lower_bound=0, base_name="inv_VRES_$(id)")
        @constraint(planner, cap_VRES == cap_initial + inv_VRES)

        q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], q_E[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES)

        # Per-scenario operational welfare; fixed CAPEX −F_cap·cap is included in
        # each scenario's loss for CVaR (same cap every jy). Planner objective
        # uses P-weighted sum in build_social_planner!.
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -(sum(W_dict[jy][jd] * (C * q_E[jh, jd, jy]) for jh in JH, jd in JD)
                  + F_cap * cap_VRES)
            )
        end
        var_dict[:power_q_E][id] = q_E
        var_dict[:power_cap_VRES][id] = cap_VRES
        var_dict[:power_inv_VRES][id] = inv_VRES
        return welfare_per_year

    else  # Conventional
        cap = _p(mod, :Capacity)
        C = _p(mod, :MarginalCost)
        MC_y = get(mod.ext[:parameters], :MarginalCostByYear, nothing)

        q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")

        if MC_y === nothing && haskey(mod.ext[:parameters], :ConvStageBaseCost)
        stage_cap = get(mod.ext[:parameters], :ConvStageCap, [cap, 0.0, 0.0])
        stage_base = mod.ext[:parameters][:ConvStageBaseCost]
        stage_slope = get(mod.ext[:parameters], :ConvStageSlope, zeros(3, length(JY)))

        q_stage = @variable(planner, [s in 1:3, jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_stage_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], q_E[jh, jd, jy] == sum(q_stage[s, jh, jd, jy] for s in 1:3))
        @constraint(planner, [s in 1:3, jh in JH, jd in JD, jy in JY], q_stage[s, jh, jd, jy] <= stage_cap[s])

        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -sum(W_dict[jy][jd] * (
                    sum(stage_base[s, jy] * q_stage[s, jh, jd, jy] + 0.5 * stage_slope[s, jy] * q_stage[s, jh, jd, jy]^2 for s in 1:3)
                ) for jh in JH, jd in JD)
            )
        end
        var_dict[:power_q_E_stage][id] = q_stage
        else
        if MC_y === nothing
            MC_y = fill(C, length(JY))
        end
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -sum(W_dict[jy][jd] * (MC_y[jy] * q_E[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        end

        @constraint(planner, [jh in JH, jd in JD, jy in JY], q_E[jh, jd, jy] <= cap)

        var_dict[:power_q_E][id] = q_E
        return welfare_per_year
    end
end
