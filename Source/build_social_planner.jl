# ==============================================================================
# build_social_planner.jl — Build centralized social planner from Source blocks
# ==============================================================================
#
# PURPOSE:
#   Constructs the social planner optimization problem by calling add_*_to_planner!
#   from each build_* module. Uses the same parameter definitions (define_*_parameters!)
#   and agent types as the market-exposure ADMM case. No duplication of objectives
#   or constraints: they all live in the build_* files.
#
#   Changes to objectives or constraints in build_* automatically propagate to
#   both market_exposure and social_planner.
#
# ARGUMENTS:
#   mdict — Dict of agent models with ext[:parameters], ext[:sets], ext[:timeseries].
#   agents — Dict of agent lists (power, H2, offtaker, elec_GC_demand, EP_demand).
#   elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market — Market dicts.
#
# RETURNS:
#   planner — JuMP model (convex QCP with epigraph formulation; ready to optimize).
#   planner_state — Dict with:
#     :var_dict, :agent_welfare (total per agent), :agent_welfare_per_year,
#     :social_welfare (aggregate per year), :sw_aux (epigraph proxy variables),
#     :gamma, :beta, :demand_var_keys (var_dict keys for quadratic-utility
#       demand variables; retained for diagnostics/extensions),
#     :elec_balance, :elec_GC_balance, :H2_balance, :H2_GC_balance, :EP_balance,
#     :power_consumers, :power_vres, :power_conv, :H2_producers, :H2_consumers,
#     :offtaker_green, :offtaker_grey, :offtaker_import, :JH, :JD, :JY, :W.
#
# ==============================================================================

# Pair α, u, CVaR with current sw_aux starts so
#   u[y] ≥ −sw_aux[y] − α,   CVaR ≥ α + (1/(1−β)) E[u]
# hold at x⁰. Uses the discrete Rockafellar–Uryasev minimizer over candidate α.
# Call after any primal copy (RN warm-start) — RN CVaR auxiliaries are not tight
# and can violate the target-β linking constraint.
function seed_social_cvar_starts!(sw_aux, u_social, alpha_social, cvar_social, JY, P, beta_conf;
                                  sw_fallback::Float64 = -1.0e11)
    n = length(JY)
    sw = Vector{Float64}(undef, n)
    for (i, jy) in enumerate(JY)
        s = start_value(sw_aux[jy])
        sw[i] = (s === nothing || !isfinite(Float64(s))) ? sw_fallback : Float64(s)
        set_start_value(sw_aux[jy], sw[i])
    end
    loss = -sw
    one_m_b = max(1e-6, 1.0 - Float64(beta_conf))
    best_cvar = Inf
    best_alpha = loss[1]
    best_u = zeros(n)
    for a in unique(loss)
        u = max.(loss .- a, 0.0)
        c = a + sum(Float64(P[JY[i]]) * u[i] for i in 1:n) / one_m_b
        if c < best_cvar
            best_cvar = c
            best_alpha = a
            best_u = u
        end
    end
    set_start_value(alpha_social, best_alpha)
    set_start_value(cvar_social, best_cvar)
    for (i, jy) in enumerate(JY)
        set_start_value(u_social[jy], best_u[i])
    end
    return nothing
end

function seed_social_cvar_starts!(planner_state::Dict)
    seed_social_cvar_starts!(
        planner_state[:sw_aux], planner_state[:u_social],
        planner_state[:alpha_social], planner_state[:cvar_social],
        planner_state[:JY], planner_state[:P], planner_state[:beta])
end

# data: full data.yaml dict; gamma and beta are read from data["ADMM"] so that
#       changing them there updates both social planner and ADMM simultaneously.
# env: optional Gurobi environment for license reuse (used only when the
# optimizer factory is Gurobi). For non-Gurobi solvers this is ignored.
# optimizer_factory: JuMP optimizer constructor/factory (e.g., Gurobi.Optimizer,
# Ipopt.Optimizer). This lets social_planner.jl switch SP solver without
# touching ADMM solvers.
function build_social_planner!(mdict::Dict, agents::Dict, elec_market::Dict, H2_market::Dict,
                               elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
                               data::Dict;
                               env::Union{Gurobi.Env, Nothing} = nothing,
                               optimizer_factory = Gurobi.Optimizer)
    if isempty(agents[:all])
        error("No agents defined in data.yaml; cannot build social planner problem.")
    end

    planner = if optimizer_factory === Gurobi.Optimizer
        env !== nothing ? Model(() -> Gurobi.Optimizer(env)) : Model(Gurobi.Optimizer)
    else
        Model(optimizer_factory)
    end
    set_silent(planner)

    m0 = mdict[agents[:all][1]]
    JH = collect(m0.ext[:sets][:JH])
    JD = collect(m0.ext[:sets][:JD])
    JY = collect(m0.ext[:sets][:JY])
    W = m0.ext[:parameters][:W]
    P = m0.ext[:parameters][:P]
    # W_dict: nested Dict {year => {day => weight}} that mirrors the W matrix.
    # WHY: JuMP @expression macros are more readable with W_dict[jy][jd] than
    # W[jd, jy], and the nested structure naturally matches the (jy, jd) loop
    # nesting used in the objective and constraint definitions below.
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    # var_dict: stores all agent decision variables indexed by agent ID and
    # organized by market role (e.g. :power_q_E for electricity supply).
    # WHY: market-clearing constraints need to sum over all participants in a
    # given market. Collecting variables here lets us iterate over the relevant
    # subset without revisiting each agent's internal model.
    var_dict = Dict{Symbol, Dict{String, Any}}(
        :power_d_E => Dict{String, Any}(),
        :power_q_E => Dict{String, Any}(),
        :power_q_E_stage => Dict{String, Any}(),
        :power_cap_VRES => Dict{String, Any}(),
        :power_inv_VRES => Dict{String, Any}(),
        :H2_e_buy => Dict{String, Any}(),
        :H2_gc_e_buy => Dict{String, Any}(),
        :H2_h_sell => Dict{String, Any}(),
        :H2_gc_h_sell => Dict{String, Any}(),
        :H2_cap_elec => Dict{String, Any}(),
        :H2_inv_elec => Dict{String, Any}(),
        :H2_d_H => Dict{String, Any}(),
        :offtaker_h_buy => Dict{String, Any}(),
        :offtaker_gc_h_buy => Dict{String, Any}(),
        :offtaker_ep_sell => Dict{String, Any}(),
        :offtaker_cap_EP_green => Dict{String, Any}(),
        :offtaker_inv_EP_green => Dict{String, Any}(),
        :offtaker_gc_h_buy_G => Dict{String, Any}(),
        :offtaker_ep_sell_import => Dict{String, Any}(),
        :elec_GC_demand_d_GC_E => Dict{String, Any}(),
        :EP_demand_d_EP => Dict{String, Any}(),
    )

    # agent_welfare_per_year: per-year welfare Dict from each add_*_to_planner!.
    # agent_welfare: total welfare per agent (sum over years); kept for
    # backward compatibility with save_social_planner_results!.
    agent_welfare_per_year = Dict{String, Dict{Int, Any}}()
    agent_welfare = Dict{String, Any}()

    # ── Classify agents by sub-type ──────────────────────────────────────
    _hasp(m, k) = haskey(m.ext[:parameters], k)
    power_consumers = String[]
    power_vres = String[]
    power_conv = String[]

    for id in agents[:power]
        m = mdict[id]
        atype = String(get(m.ext[:parameters], :Type, ""))
        if atype == "Consumer"
            push!(power_consumers, id)
        elseif atype == "VRES"
            push!(power_vres, id)
        else
            push!(power_conv, id)
        end
    end

    H2_producers = String[]
    H2_consumers = String[]

    for id in agents[:H2]
        m = mdict[id]
        if _hasp(m, :Capacity_Electrolyzer) || (_hasp(m, :E_bar) && _hasp(m, :H_bar))
            push!(H2_producers, id)
        elseif _hasp(m, :D_H_bar)
            push!(H2_consumers, id)
        end
    end

    offtaker_green = String[]
    offtaker_grey = String[]
    offtaker_import = String[]

    for id in agents[:offtaker]
        m = mdict[id]
        atype = String(get(m.ext[:parameters], :Type, ""))
        if atype == "GreenOfftaker"
            push!(offtaker_green, id)
        elseif atype == "EPImporter"
            push!(offtaker_import, id)
        else
            push!(offtaker_grey, id)
        end
    end

    # ── Add agents (each returns per-year welfare Dict) ──────────────────
    # Each add_*_to_planner! function creates the agent's decision variables
    # and physical constraints inside the shared `planner` model, stores
    # variables in `var_dict` for use in market-clearing constraints, and
    # returns a Dict{Int, Any} mapping year jy → JuMP expression for that
    # agent's welfare contribution in year jy (utility for consumers,
    # negative cost for producers). Market transfers (λ·q) cancel in the
    # aggregate and are therefore excluded from welfare expressions.
    #
    # agent_welfare_per_year[id][jy] is needed for the social CVaR, which
    # operates on the per-year aggregate. agent_welfare[id] = Σ_y wpy[jy]
    # is kept for backward compatibility with save_social_planner_results!.

    for id in agents[:power]
        wpy = add_power_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
    end

    for id in H2_producers
        wpy = add_H2_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
    end

    # H₂ consumers: inelastic demand with linear utility (no quadratic
    # term). D_H_bar[jh,jd,jy] = LOAD_H × D_H_bar is the maximum H₂
    # demand in each hour. Per-year welfare = Utility × total H₂ consumed.
    for id in H2_consumers
        m = mdict[id]
        p = m.ext[:parameters]
        ts = m.ext[:timeseries]
        D_H_bar = ts[:LOAD_H] .* p[:D_H_bar]
        utility_val = get(p, :Utility, 0.0)

        d_H = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_H_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], d_H[jh, jd, jy] <= D_H_bar[jh, jd, jy])

        # Per-year welfare = linear utility of H₂ consumption.
        # No expenditure: H₂ payments cancel in the planner aggregate.
        wpy = Dict{Int, Any}()
        for jy in JY
            wpy[jy] = @expression(planner,
                sum(W_dict[jy][jd] * (utility_val * d_H[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
        var_dict[:H2_d_H][id] = d_H
    end

    for id in agents[:offtaker]
        wpy = add_offtaker_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
    end

    for id in agents[:elec_GC_demand]
        wpy = add_elec_GC_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
    end

    for id in agents[:EP_demand]
        wpy = add_EP_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(P[jy] * wpy[jy] for jy in JY))
    end

    # ── Market-clearing constraints ──────────────────────────────────────
    # In the planner, prices are NOT explicit decision variables — they
    # emerge as shadow prices (dual values) of the market-clearing
    # constraints. Each constraint enforces hourly supply = demand for one
    # market. The haskey() guards handle agents that participate in a
    # market subset (e.g. only VRES supply elec-GCs; conventional does not).

    # Electricity market: VRES + Conventional supply = Consumer demand + H₂ producer intake.
    elec_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:power_q_E], id) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:power_d_E], id) ? var_dict[:power_d_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_e_buy], id) ? var_dict[:H2_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) == 0.0
    )

    # Electricity-GC market: VRES GC supply = H₂ producer GC purchases + GC demand.
    elec_GC_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum((haskey(var_dict[:power_q_E], id) && id in power_vres) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_gc_e_buy], id) ? var_dict[:H2_gc_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:elec_GC_demand_d_GC_E], id) ? var_dict[:elec_GC_demand_d_GC_E][id][jh, jd, jy] : 0.0 for id in agents[:elec_GC_demand]) == 0.0
    )

    # H₂ market: H₂ producer output = H₂ consumer demand + GreenOfftaker purchases.
    H2_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:H2_h_sell], id) ? var_dict[:H2_h_sell][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:H2_d_H], id) ? var_dict[:H2_d_H][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:offtaker_h_buy], id) ? var_dict[:offtaker_h_buy][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) == 0.0
    )

    # H₂-GC market: H₂ producer GC issuance = GreenOfftaker + GreyOfftaker GC purchases.
    H2_GC_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:H2_gc_h_sell], id) ? var_dict[:H2_gc_h_sell][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:offtaker_gc_h_buy], id) ? var_dict[:offtaker_gc_h_buy][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) -
        sum(haskey(var_dict[:offtaker_gc_h_buy_G], id) ? var_dict[:offtaker_gc_h_buy_G][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) == 0.0
    )

    # EP market: GreenOfftaker + EPImporter supply = fixed inelastic demand D_EP + elastic EP demand.
    D_EP = EP_market["D_EP"]
    EP_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:offtaker_ep_sell], id) ? var_dict[:offtaker_ep_sell][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) +
        sum(haskey(var_dict[:offtaker_ep_sell_import], id) ? var_dict[:offtaker_ep_sell_import][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) -
        D_EP[jh, jd, jy] -
        sum(haskey(var_dict[:EP_demand_d_EP], id) ? var_dict[:EP_demand_d_EP][id][jh, jd, jy] : 0.0 for id in agents[:EP_demand]) == 0.0
    )

    # ── Single social CVaR ───────────────────────────────────────────────
    # Instead of per-agent CVaR terms (which incorrectly produced 3 separate
    # risk-aversion terms), the social planner applies a SINGLE CVaR to the
    # aggregate social welfare. γ and β are read from data["ADMM"] so that
    # changing them there updates both social planner and ADMM simultaneously.
    # When γ=1 (risk-neutral), the CVaR term vanishes and the planner reduces
    # to the standard welfare maximization, matching the ADMM risk-neutral equilibrium.
    admm = get(data, "ADMM", Dict{String, Any}())
    gamma = get(admm, "gamma", 1.0)
    beta_conf = get(admm, "beta", 0.95)
    P = mdict[agents[:all][1]].ext[:parameters][:P]

    # ── Aggregate social welfare per year ─────────────────────────────────
    # social_welfare[jy] sums ALL agents' per-year welfare, including
    # quadratic consumer utility terms (A·d − B/2·d²).
    all_ids = collect(keys(agent_welfare_per_year))
    social_welfare = Dict{Int, Any}()
    for jy in JY
        social_welfare[jy] = @expression(planner,
            sum(agent_welfare_per_year[id][jy] for id in all_ids)
        )
    end

    # ── Epigraph variables for social welfare ────────────────────────────
    # Keep a single planner formulation for all gamma values: CVaR becomes
    # inactive automatically when gamma = 1 via the objective coefficient.
    sw_aux = @variable(planner, [jy in JY], base_name = "sw_aux")
    social_welfare_epigraph = @constraint(planner, social_welfare_epigraph[jy in JY],
        sw_aux[jy] <= social_welfare[jy]
    )

    alpha_social = @variable(planner, base_name = "alpha_social")
    cvar_social  = @variable(planner, base_name = "CVaR_social")
    u_social     = @variable(planner, [jy in JY], lower_bound = 0, base_name = "u_social")

    @constraint(planner, [jy in JY],
        u_social[jy] >= -sw_aux[jy] - alpha_social
    )
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    @constraint(planner,
        cvar_social >= alpha_social + (1 / one_minus_beta) * sum(P[jy] * u_social[jy] for jy in JY)
    )

    @objective(planner, Max,
        gamma * sum(P[jy] * sw_aux[jy] for jy in JY)
        - (1 - gamma) * cvar_social
    )

    # Cold start: sw_aux well below seed-CAPEX welfare (~−5e9 € if d=g=0) so the
    # epigraph sw_aux ≤ social_welfare holds. Then pair α, u, CVaR so the
    # Rockafellar–Uryasev shortfall is feasible. The old seeds (sw_aux = α = −1e15,
    # u = 0, CVaR = 0) made u ≥ −sw_aux − α infeasible (RHS = 2e15) and IPOPT
    # died in restoration once γ<1 put CVaR in the objective.
    for jy in JY
        set_start_value(sw_aux[jy], -1.0e11)
    end
    seed_social_cvar_starts!(sw_aux, u_social, alpha_social, cvar_social, JY, P, beta_conf)

    # ── Return planner model and state dictionary ─────────────────────────
    # planner_state carries all information that save_social_planner_results!
    # and post-processing scripts need:
    #   :var_dict              — agent decision variables by market role
    #   :agent_welfare         — total welfare per agent (Σ_y), for results
    #   :agent_welfare_per_year — per-year welfare Dict (for social CVaR)
    #   :social_welfare        — aggregate social welfare per year
    #   :*_balance             — market-clearing constraints (duals = prices)
    #   :power_*/H2_*/offtaker_* — agent ID lists by sub-type
    #   :JH, :JD, :JY, :W     — index sets and weights
    #   :gamma, :beta          — risk parameters (shared by all agents)
    #   :demand_var_keys       — var_dict keys for demand agents (diagnostics/extensions)
    planner_state = Dict(
        :var_dict => var_dict,
        :agent_welfare => agent_welfare,
        :agent_welfare_per_year => agent_welfare_per_year,
        :social_welfare => social_welfare,
        :elec_balance => elec_balance,
        :elec_GC_balance => elec_GC_balance,
        :H2_balance => H2_balance,
        :H2_GC_balance => H2_GC_balance,
        :EP_balance => EP_balance,
        :power_consumers => power_consumers,
        :power_vres => power_vres,
        :power_conv => power_conv,
        :H2_producers => H2_producers,
        :H2_consumers => H2_consumers,
        :offtaker_green => offtaker_green,
        :offtaker_grey => offtaker_grey,
        :offtaker_import => offtaker_import,
        :JH => JH,
        :JD => JD,
        :JY => JY,
        :W => W,
        :gamma => gamma,
        :beta => beta_conf,
        :P => P,
        :demand_var_keys => [:power_d_E, :elec_GC_demand_d_GC_E, :EP_demand_d_EP],
        :sw_aux => sw_aux,
        :social_welfare_epigraph => social_welfare_epigraph,
        :cvar_social => cvar_social,
        :alpha_social => alpha_social,
        :u_social => u_social,
    )

    return planner, planner_state
end
