# ==============================================================================
# save_social_planner_results.jl — Write social planner outputs to CSV
# ==============================================================================
#
# PURPOSE:
#   Saves Market_Prices.csv (dual of balance constraints) and Agent_Summary.csv
#   from the solved social planner model.
#
#   CONTEXT: This function is called after direct QCP solve in
#   social_planner.jl with solver duals available.
#
#   POST-PROCESSING: The social planner maximizes welfare (no prices in objective);
#   prices emerge as dual variables. To match the ADMM Agent_Summary format
#   (which reports cost − revenue per agent), we post-process: after extracting
#   equilibrium prices from the duals, we compute for each agent the ADMM-style
#   objective (cost − revenue) evaluated at the optimal quantities and those
#   prices. This ensures direct comparability with market_exposure_results.
#
# ARGUMENTS:
#   planner — Solved JuMP model (QCP with duals available).
#   planner_state — Dict from build_social_planner! (:var_dict, :agent_welfare,
#     :agent_welfare_per_year, :social_welfare, :sw_aux, :demand_var_keys,
#     :elec_balance, :elec_GC_balance, :H2_balance, :H2_GC_balance, :EP_balance,
#     :power_consumers, :power_vres, :power_conv, :H2_producers, :H2_consumers,
#     :offtaker_green, :offtaker_grey, :offtaker_import, :JH, :JD, :JY, :W,
#     :gamma, :beta).
#   agents — Dict of agent lists.
#   mdict — Dict of parameter-container models (one per agent); needed to read
#     cost/utility parameters for ADMM-style objective computation.
#   results_folder — Path to social_planner_results directory.
#
# ==============================================================================

import Printf: @sprintf, @printf

if !isdefined(@__MODULE__, :print_social_planner_run_summary!)
    include(joinpath(@__DIR__, "print_run_summary.jl"))
end
if !isdefined(@__MODULE__, :with_run_summary_log)
    include(joinpath(@__DIR__, "tee_run_log.jl"))
end
include(joinpath(@__DIR__, "compute_social_risk_metrics.jl"))

function save_social_planner_results!(planner::Model, planner_state::Dict, agents::Dict,
                                      mdict::Dict, results_folder::String)
    var_dict = planner_state[:var_dict]
    agent_welfare = planner_state[:agent_welfare]
    power_consumers = planner_state[:power_consumers]
    power_vres = planner_state[:power_vres]
    power_conv = planner_state[:power_conv]
    H2_producers = planner_state[:H2_producers]
    H2_consumers = planner_state[:H2_consumers]
    offtaker_green = planner_state[:offtaker_green]
    offtaker_grey = planner_state[:offtaker_grey]
    offtaker_import = planner_state[:offtaker_import]
    JH = planner_state[:JH]
    JD = planner_state[:JD]
    JY = planner_state[:JY]

    elec_balance = planner_state[:elec_balance]
    elec_GC_balance = planner_state[:elec_GC_balance]
    H2_balance = planner_state[:H2_balance]
    H2_GC_balance = planner_state[:H2_GC_balance]
    EP_balance = planner_state[:EP_balance]
    W = planner_state[:W]   # Representative-day weights W[jd, jy]
    # Uniform value helper for variables/expressions used below.
    _val(v::JuMP.VariableRef) = value(v)
    _val(x) = value(x)

    # ── Check solver status before extracting results ─────────────────
    status = termination_status(planner)
    # This function is called after direct QCP solve.
    # LOCALLY_SOLVED is accepted as a fallback for convex models.
    if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED && status != MOI.ALMOST_LOCALLY_SOLVED
        @warn "Social planner did NOT solve to optimality (status: $status). " *
              "Cannot extract duals or variable values."
        if status == MOI.INFEASIBLE
            @error "Model is INFEASIBLE. Check that demand ≤ total supply capacity " *
                   "for all markets (especially EP: D_EP = Total_Demand × LOAD_EP ≤ Σ EP capacities)."
        elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
            @error "Model status is INFEASIBLE_OR_UNBOUNDED. " *
                   "Check solver settings and convexity assumptions; " *
                   "if infeasible, check market-capacity feasibility (especially EP)."
        elseif status == MOI.DUAL_INFEASIBLE
            @error "Model appears unbounded (DUAL_INFEASIBLE). " *
                   "Check model scaling/convexity and solver settings."
        end
        return
    end

    # ── Market_Prices.csv — Equilibrium prices from dual variables ──────
    # By convex duality, the shadow price (dual value) of a market-clearing
    # constraint equals the equilibrium price in that market at that
    # timestep. Extracting duals is the standard way to recover prices from
    # a centralized welfare-maximization problem. Includes all 5 markets.
    #
    # IMPORTANT: The raw duals of per-timestep constraints carry TWO scaling
    # factors that must be removed to recover the true economic price:
    #
    #   (1) Representative-day weight W[jd,jy]: the objective sums W * welfare
    #       over days, so each dual is proportional to W.
    #
    #   (2) Effective probability weight μ[jy]: the objective scales per-year
    #       welfare by γ * P[jy] (plus the CVaR adjustment for tail scenarios).
    #       For a risk-neutral planner (γ=1): μ[jy] = P[jy].
    #       For a risk-adjusted planner (γ<1) the CVaR adds extra weight to
    #       tail scenarios: μ[jy] = γ*P[jy] + ξ[jy], where ξ[jy] is the dual
    #       of the shortfall constraint u_social[jy] ≥ -sw_aux[jy] - α_social.
    #
    #   True price = raw_dual / (W[jd,jy] * μ[jy])
    #
    # We obtain μ[jy] as the dual of the epigraph constraint
    # sw_aux[jy] ≤ social_welfare[jy], which at optimality equals
    # γ*P[jy] + ξ[jy] (the full probability+risk weight for that scenario).
    # This is the most general and correct normalisation: it handles any γ, β,
    # and any number of tail scenarios without special-casing.
    #
    # Guard: if an epigraph dual is numerically zero (degenerate solver
    # output), fall back to γ * P[jy] to avoid division by zero.
    epigraph_constr = planner_state[:social_welfare_epigraph]
    gamma_val = Float64(planner_state[:gamma])
    ref_m0 = mdict[agents[:all][1]]
    P_arr = ref_m0.ext[:parameters][:P]
    mu_y = Dict{Int, Float64}()
    for jy in JY
        mu_raw = dual(epigraph_constr[jy])
        # dual() is positive for a ≤ constraint in a MAX problem when the
        # constraint is tight (relaxing it improves the objective).
        mu_eff = abs(mu_raw)
        if mu_eff < 1e-12
            # Fallback: use γ * P[jy] (exact for non-tail, conservative for tail)
            mu_eff = max(gamma_val * Float64(P_arr[jy]), 1e-12)
        end
        mu_y[jy] = mu_eff
    end

    prices_rows = []
    t_index = 1
    for jy in JY, jd in JD, jh in JH
        w = W[jd, jy]
        wmu = w * mu_y[jy]
        push!(prices_rows, (
            Time = t_index,
            Elec_Price    = dual(elec_balance[jy, jh, jd])    / wmu,
            H2_Price      = dual(H2_balance[jy, jh, jd])      / wmu,
            Elec_GC_Price = dual(elec_GC_balance[jy, jh, jd]) / wmu,
            H2_GC_Price   = dual(H2_GC_balance[jy, jh, jd])   / wmu,
            EP_Price      = dual(EP_balance[jy, jh, jd])       / wmu,
        ))
        t_index += 1
    end

    prices_df = DataFrame(prices_rows)
    CSV.write(joinpath(results_folder, "Market_Prices.csv"), prices_df)

    # ── SP_Primal_Quantities.csv — Per-agent g_net per market for ADMM warm-start ─
    # One row per (jy, jd, jh) with columns m_elec, m_H2, m_elec_GC, m_H2_GC, m_EP
    # for each agent m. ME loads this to pre-populate results so iteration 1 has
    # g_bar = SP solution (prev = SP, imb = 0) and converges immediately.
    primal_rows = []
    t_idx = 1
    for jy in JY, jd in JD, jh in JH
        row = Dict{Symbol, Any}(:Time => t_idx, :jh => jh, :jd => jd, :jy => jy)
        for id in agents[:all]
            g_elec = 0.0
            g_H2 = 0.0
            g_elec_GC = 0.0
            g_H2_GC = 0.0
            g_EP = 0.0
            if id in power_consumers
                d = var_dict[:power_d_E][id]
                g_elec = -_val(d[jh, jd, jy])
            elseif id in power_vres || id in power_conv
                q = var_dict[:power_q_E][id]
                g_elec = _val(q[jh, jd, jy])
                if id in power_vres
                    g_elec_GC = _val(q[jh, jd, jy])
                end
            elseif id in H2_producers
                e_buy = var_dict[:H2_e_buy][id]
                gc_e = var_dict[:H2_gc_e_buy][id]
                h_sell = var_dict[:H2_h_sell][id]
                gc_h = var_dict[:H2_gc_h_sell][id]
                g_elec = -_val(e_buy[jh, jd, jy])
                g_elec_GC = -_val(gc_e[jh, jd, jy])
                g_H2 = _val(h_sell[jh, jd, jy])
                g_H2_GC = _val(gc_h[jh, jd, jy])
            elseif id in offtaker_green
                h_buy = var_dict[:offtaker_h_buy][id]
                gc_buy = var_dict[:offtaker_gc_h_buy][id]
                ep_sell = var_dict[:offtaker_ep_sell][id]
                g_H2 = -_val(h_buy[jh, jd, jy])
                g_H2_GC = -_val(gc_buy[jh, jd, jy])
                g_EP = _val(ep_sell[jh, jd, jy])
            elseif id in offtaker_grey
                ep_sell = var_dict[:offtaker_ep_sell][id]
                gc_buy = var_dict[:offtaker_gc_h_buy_G][id]
                g_H2_GC = -_val(gc_buy[jh, jd, jy])
                g_EP = _val(ep_sell[jh, jd, jy])
            elseif id in offtaker_import
                ep_sell = var_dict[:offtaker_ep_sell_import][id]
                g_EP = _val(ep_sell[jh, jd, jy])
            elseif id in agents[:elec_GC_demand]
                dgc = var_dict[:elec_GC_demand_d_GC_E][id]
                g_elec_GC = -_val(dgc[jh, jd, jy])
            end
            row[Symbol(id * "_elec")] = g_elec
            row[Symbol(id * "_H2")] = g_H2
            row[Symbol(id * "_elec_GC")] = g_elec_GC
            row[Symbol(id * "_H2_GC")] = g_H2_GC
            row[Symbol(id * "_EP")] = g_EP
        end
        push!(primal_rows, row)
        t_idx += 1
    end
    primal_df = DataFrame(primal_rows)
    CSV.write(joinpath(results_folder, "SP_Primal_Quantities.csv"), primal_df)

    # ── SP_Capacities.csv — Per-agent installed capacity for ME warm-start ─
    # One row per agent (jy=1 label); capacity is scenario-invariant.
    cap_rows = []
    for id in agents[:all]
        cap_val = nothing
        if id in power_vres && haskey(var_dict, :power_cap_VRES) && haskey(var_dict[:power_cap_VRES], id)
            cap_val = _val(var_dict[:power_cap_VRES][id])
        elseif id in H2_producers && haskey(var_dict, :H2_cap_elec) && haskey(var_dict[:H2_cap_elec], id)
            cap_val = _val(var_dict[:H2_cap_elec][id])
        elseif id in offtaker_green && haskey(var_dict, :offtaker_cap_EP_green) && haskey(var_dict[:offtaker_cap_EP_green], id)
            cap_val = _val(var_dict[:offtaker_cap_EP_green][id])
        end
        if cap_val !== nothing
            push!(cap_rows, (AgentID = id, jy = 1, cap = cap_val))
        end
    end
    if !isempty(cap_rows)
        cap_df = DataFrame(cap_rows)
        CSV.write(joinpath(results_folder, "SP_Capacities.csv"), cap_df)
    end

    # ── Build 3D price arrays [jh, jd, jy] for ADMM-style objective computation ─
    # The duals are indexed [jy, jh, jd]. We build λ[jh, jd, jy] to match the
    # ADMM agent objective formulas.
    n_jh, n_jd, n_jy = length(JH), length(JD), length(JY)
    λ_elec    = zeros(n_jh, n_jd, n_jy)
    λ_H2      = zeros(n_jh, n_jd, n_jy)
    λ_elec_GC = zeros(n_jh, n_jd, n_jy)
    λ_H2_GC   = zeros(n_jh, n_jd, n_jy)
    λ_EP      = zeros(n_jh, n_jd, n_jy)
    for (iy, jy) in enumerate(JY), (id, jd) in enumerate(JD), (ih, jh) in enumerate(JH)
        wmu = W[jd, jy] * mu_y[jy]
        λ_elec[ih, id, iy]    = dual(elec_balance[jy, jh, jd])    / wmu
        λ_H2[ih, id, iy]      = dual(H2_balance[jy, jh, jd])      / wmu
        λ_elec_GC[ih, id, iy] = dual(elec_GC_balance[jy, jh, jd]) / wmu
        λ_H2_GC[ih, id, iy]   = dual(H2_GC_balance[jy, jh, jd])   / wmu
        λ_EP[ih, id, iy]      = dual(EP_balance[jy, jh, jd])       / wmu
    end

    # Helper: compute ADMM-style objective (cost − revenue) for an agent using the
    # shared compute_agent_objective_economic so objectives match market_exposure.
    function _planner_objective(id, ag_type)
        m = mdict[id]
        p = m.ext[:parameters]
        params = merge(Dict(:W => W), Dict(k => v for (k, v) in p))
        prices = Dict(
            :λ_elec => λ_elec, :λ_H2 => λ_H2, :λ_elec_GC => λ_elec_GC,
            :λ_H2_GC => λ_H2_GC, :λ_EP => λ_EP,
        )
        quantities = Dict{Symbol, Any}()

        if id in power_consumers
            d = var_dict[:power_d_E][id]
            quantities[:d] = [_val(d[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_consumer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in power_vres
            q = var_dict[:power_q_E][id]
            quantities[:g] = [_val(q[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :power_cap_VRES) && haskey(var_dict[:power_cap_VRES], id)
                cap = var_dict[:power_cap_VRES][id]
                quantities[:cap_VRES] = [_val(cap)]
            end
            return compute_agent_objective_economic(:power_vres, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in power_conv
            q = var_dict[:power_q_E][id]
            quantities[:g] = [_val(q[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_conv, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in H2_producers
            quantities[:e_in] = [_val(var_dict[:H2_e_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_elec_gc] = [_val(var_dict[:H2_gc_e_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:h2_out] = [_val(var_dict[:H2_h_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:H2_gc_h_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :H2_cap_elec) && haskey(var_dict[:H2_cap_elec], id)
                cap = var_dict[:H2_cap_elec][id]
                quantities[:cap_H2_y] = [_val(cap)]
            end
            return compute_agent_objective_economic(:H2_producer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in H2_consumers
            quantities[:d_H] = [_val(var_dict[:H2_d_H][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:H2_consumer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_green
            quantities[:h2_in] = [_val(var_dict[:offtaker_h_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:offtaker_gc_h_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :offtaker_cap_EP_green) && haskey(var_dict[:offtaker_cap_EP_green], id)
                cap = var_dict[:offtaker_cap_EP_green][id]
                quantities[:cap_EP_y] = [_val(cap)]
            end
            return compute_agent_objective_economic(:offtaker_green, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_grey
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:offtaker_gc_h_buy_G][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:offtaker_grey, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_import
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell_import][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:offtaker_import, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in agents[:elec_GC_demand]
            quantities[:d_gc] = [_val(var_dict[:elec_GC_demand_d_GC_E][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:elec_GC_demand, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in agents[:EP_demand]
            quantities[:d_EP] = [_val(var_dict[:EP_demand_d_EP][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:EP_demand, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        else
            return haskey(agent_welfare, id) ? value(agent_welfare[id]) : 0.0
        end
    end

    # ── Agent_Summary.csv — unified per-agent summary (quantities, investment, objective) ─
    # For every agent, record:
    #   - Group            — high-level sector (power, H2, offtaker, elec_GC_demand, EP_demand)
    #   - Type             — sub-type label (PowerCons, PowerGen, H2Prod, H2Cons, Offtaker, GC_Demand, EP_Demand)
    #   - Net quantities   — total net position in each market (elec, H2, elec_GC, H2_GC, EP)
    #   - Capacity summary — final-year capacity and total investment (for capacity-expanding agents)
    #   - Objective_Value  — ADMM-style cost − revenue (plus fixed CAPEX) at planner prices

    summary = DataFrame(
        AgentID = String[],
        Group = String[],
        Type = String[],
        elec_net_sum = Float64[],
        H2_net_sum = Float64[],
        elec_GC_net_sum = Float64[],
        H2_GC_net_sum = Float64[],
        EP_net_sum = Float64[],
        Capacity_Final_MW = Float64[],
        Investment_Total_MW = Float64[],
        Objective_Value = Float64[],
    )

    # Helper to compute per-agent net market quantities (signs mirror ADMM g_net_*).
    function _net_quantities(id::String)
        elec_q = 0.0
        H2_q = 0.0
        elec_GC_q = 0.0
        H2_GC_q = 0.0
        EP_q = 0.0

        if id in power_consumers
            d = var_dict[:power_d_E][id]
            elec_q -= sum(_val.(d))
        elseif id in power_vres
            q = var_dict[:power_q_E][id]
            elec_q += sum(_val.(q))
            elec_GC_q += sum(_val.(q))
        elseif id in power_conv
            q = var_dict[:power_q_E][id]
            elec_q += sum(_val.(q))
        end

        if id in H2_producers
            e_buy = var_dict[:H2_e_buy][id]
            gc_e_buy = var_dict[:H2_gc_e_buy][id]
            h_sell = var_dict[:H2_h_sell][id]
            gc_h_sell = var_dict[:H2_gc_h_sell][id]
            elec_q -= sum(_val.(e_buy))
            elec_GC_q -= sum(_val.(gc_e_buy))
            H2_q += sum(_val.(h_sell))
            H2_GC_q += sum(_val.(gc_h_sell))
        elseif id in H2_consumers
            dH = var_dict[:H2_d_H][id]
            H2_q -= sum(_val.(dH))
        end

        if id in offtaker_green
            h_buy = var_dict[:offtaker_h_buy][id]
            gc_h_buy = var_dict[:offtaker_gc_h_buy][id]
            ep_sell = var_dict[:offtaker_ep_sell][id]
            H2_q -= sum(_val.(h_buy))
            H2_GC_q -= sum(_val.(gc_h_buy))
            EP_q += sum(_val.(ep_sell))
        elseif id in offtaker_grey
            ep_sell = var_dict[:offtaker_ep_sell][id]
            gc_h_buy_G = var_dict[:offtaker_gc_h_buy_G][id]
            H2_GC_q -= sum(_val.(gc_h_buy_G))
            EP_q += sum(_val.(ep_sell))
        elseif id in offtaker_import
            ep_sell_import = var_dict[:offtaker_ep_sell_import][id]
            EP_q += sum(_val.(ep_sell_import))
        end

        if id in agents[:elec_GC_demand]
            dgc = var_dict[:elec_GC_demand_d_GC_E][id]
            elec_GC_q -= sum(_val.(dgc))
        end

        if id in agents[:EP_demand]
            dep = var_dict[:EP_demand_d_EP][id]
            EP_q -= sum(_val.(dep))
        end

        return elec_q, H2_q, elec_GC_q, H2_GC_q, EP_q
    end

    # Helper for capacity summary: final-year capacity and total investment.
    function _capacity_summary(id::String)
        cap_final = 0.0
        inv_total = 0.0

        if id in power_vres
            cap_dict = get(var_dict, :power_cap_VRES, nothing)
            inv_dict = get(var_dict, :power_inv_VRES, nothing)
            if cap_dict !== nothing && haskey(cap_dict, id)
                cap_var = cap_dict[id]
                cap_final = _val(cap_var)
            end
            if inv_dict !== nothing && haskey(inv_dict, id)
                inv_var = inv_dict[id]
                inv_total = _val(inv_var)
            end
        elseif id in H2_producers
            cap_dict = get(var_dict, :H2_cap_elec, nothing)
            inv_dict = get(var_dict, :H2_inv_elec, nothing)
            if cap_dict !== nothing && haskey(cap_dict, id)
                cap_var = cap_dict[id]
                cap_final = _val(cap_var)
            end
            if inv_dict !== nothing && haskey(inv_dict, id)
                inv_var = inv_dict[id]
                inv_total = _val(inv_var)
            end
        elseif id in offtaker_green
            cap_dict = get(var_dict, :offtaker_cap_EP_green, nothing)
            inv_dict = get(var_dict, :offtaker_inv_EP_green, nothing)
            if cap_dict !== nothing && haskey(cap_dict, id)
                cap_var = cap_dict[id]
                cap_final = _val(cap_var)
            end
            if inv_dict !== nothing && haskey(inv_dict, id)
                inv_var = inv_dict[id]
                inv_total = _val(inv_var)
            end
        end

        return cap_final, inv_total
    end

    for id in agents[:all]
        group = if id in agents[:power]
            "power"
        elseif id in agents[:H2]
            "H2"
        elseif id in agents[:offtaker]
            "offtaker"
        elseif id in agents[:elec_GC_demand]
            "elec_GC_demand"
        elseif id in agents[:EP_demand]
            "EP_demand"
        else
            "unknown"
        end

        type_label = if id in power_consumers
            "PowerCons"
        elseif id in power_vres || id in power_conv
            "PowerGen"
        elseif id in H2_producers
            "H2Prod"
        elseif id in H2_consumers
            "H2Cons"
        elseif id in offtaker_green || id in offtaker_grey || id in offtaker_import
            "Offtaker"
        elseif id in agents[:elec_GC_demand]
            "GC_Demand"
        elseif id in agents[:EP_demand]
            "EP_Demand"
        else
            "Unknown"
        end

        elec_q, H2_q, elec_GC_q, H2_GC_q, EP_q = _net_quantities(id)
        cap_final, inv_total = _capacity_summary(id)
        obj_val = _planner_objective(id, type_label)

        push!(summary, (String(id), group, type_label,
                        elec_q, H2_q, elec_GC_q, H2_GC_q, EP_q,
                        cap_final, inv_total, obj_val))
    end

    CSV.write(joinpath(results_folder, "Agent_Summary.csv"), summary)

    # ── Agent_Objectives_Per_Timestep.csv — Per-hour prices, quantities, objective contributions ─
    # One row per timestep (jh, jd, jy). Column order: Time, jh, jd, jy, W; all prices;
    # all quantities (VRES, CONV, elec demand, elec GC demand, H2 prod, green off, grey off, …);
    # all objective values (same agent order).
    prices_dict = Dict(
        :λ_elec => λ_elec, :λ_H2 => λ_H2, :λ_elec_GC => λ_elec_GC,
        :λ_H2_GC => λ_H2_GC, :λ_EP => λ_EP,
    )
    # Canonical agent order: VRES, CONV, elec demand, elec GC demand, H2 producer, green off, grey off, import off, H2 consumer, EP demand
    ordered_agents = [id for id in vcat(
        power_vres, power_conv, power_consumers,
        get(agents, :elec_GC_demand, String[]),
        H2_producers, offtaker_green, offtaker_grey, offtaker_import,
        H2_consumers, get(agents, :EP_demand, String[]),
    ) if id in agents[:all]]
    qvars = Dict(:power_vres => [:g], :power_conv => [:g], :power_consumer => [:d], :elec_GC_demand => [:d_gc],
                 :H2_producer => [:e_in, :q_elec_gc, :h2_out, :q_h2gc], :offtaker_green => [:h2_in, :q_h2gc, :ep],
                 :offtaker_grey => [:ep, :q_h2gc], :offtaker_import => [:ep], :H2_consumer => [:d_H], :EP_demand => [:d_EP])
    type_of(id) = id in power_vres ? :power_vres : id in power_conv ? :power_conv : id in power_consumers ? :power_consumer :
                  id in get(agents, :elec_GC_demand, []) ? :elec_GC_demand : id in H2_producers ? :H2_producer :
                  id in offtaker_green ? :offtaker_green : id in offtaker_grey ? :offtaker_grey :
                  id in offtaker_import ? :offtaker_import : id in H2_consumers ? :H2_consumer :
                  id in get(agents, :EP_demand, []) ? :EP_demand : :unknown

    function _get_agent_data(id)
        p = mdict[id].ext[:parameters]
        params = merge(Dict(:W => W), Dict(k => v for (k, v) in p))
        quantities = Dict{Symbol, Any}()
        agent_type = nothing
        if id in power_consumers
            agent_type = :power_consumer
            d = var_dict[:power_d_E][id]
            quantities[:d] = [_val(d[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in power_vres
            agent_type = :power_vres
            q = var_dict[:power_q_E][id]
            quantities[:g] = [_val(q[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :power_cap_VRES) && haskey(var_dict[:power_cap_VRES], id)
                quantities[:cap_VRES] = [_val(var_dict[:power_cap_VRES][id])]
            end
        elseif id in power_conv
            agent_type = :power_conv
            q = var_dict[:power_q_E][id]
            quantities[:g] = [_val(q[jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in H2_producers
            agent_type = :H2_producer
            quantities[:e_in] = [_val(var_dict[:H2_e_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_elec_gc] = [_val(var_dict[:H2_gc_e_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:h2_out] = [_val(var_dict[:H2_h_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:H2_gc_h_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :H2_cap_elec) && haskey(var_dict[:H2_cap_elec], id)
                quantities[:cap_H2_y] = [_val(var_dict[:H2_cap_elec][id])]
            end
        elseif id in H2_consumers
            agent_type = :H2_consumer
            quantities[:d_H] = [_val(var_dict[:H2_d_H][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in offtaker_green
            agent_type = :offtaker_green
            quantities[:h2_in] = [_val(var_dict[:offtaker_h_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:offtaker_gc_h_buy][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(var_dict, :offtaker_cap_EP_green) && haskey(var_dict[:offtaker_cap_EP_green], id)
                quantities[:cap_EP_y] = [_val(var_dict[:offtaker_cap_EP_green][id])]
            end
        elseif id in offtaker_grey
            agent_type = :offtaker_grey
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [_val(var_dict[:offtaker_gc_h_buy_G][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in offtaker_import
            agent_type = :offtaker_import
            quantities[:ep] = [_val(var_dict[:offtaker_ep_sell_import][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in get(agents, :elec_GC_demand, String[])
            agent_type = :elec_GC_demand
            quantities[:d_gc] = [_val(var_dict[:elec_GC_demand_d_GC_E][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        elseif id in get(agents, :EP_demand, String[])
            agent_type = :EP_demand
            quantities[:d_EP] = [_val(var_dict[:EP_demand_d_EP][id][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
        end
        contrib = agent_type !== nothing ? compute_agent_objective_contributions(agent_type, quantities, prices_dict, params; JH=JH, JD=JD, JY=JY) : nothing
        return agent_type, quantities, contrib
    end

    # Build column order: Time, jh, jd, jy, W; prices; quantities per agent; obj per agent
    col_order = [:Time, :jh, :jd, :jy, :W, :Elec_Price, :H2_Price, :Elec_GC_Price, :H2_GC_Price, :EP_Price]
    for id in ordered_agents
        t = type_of(id)
        for v in get(qvars, t, [])
            push!(col_order, Symbol(id * "_" * string(v)))
        end
    end
    for id in ordered_agents
        push!(col_order, Symbol(id * "_obj"))
    end

    agent_data = Dict(id => _get_agent_data(id) for id in ordered_agents)
    ts_rows = []
    t_idx = 1
    for jy in JY, jd in JD, jh in JH
        w = W[jd, jy]
        row = Dict(:Time => t_idx, :jh => jh, :jd => jd, :jy => jy, :W => w,
                   :Elec_Price => λ_elec[jh, jd, jy], :H2_Price => λ_H2[jh, jd, jy],
                   :Elec_GC_Price => λ_elec_GC[jh, jd, jy], :H2_GC_Price => λ_H2_GC[jh, jd, jy],
                   :EP_Price => λ_EP[jh, jd, jy])
        for id in ordered_agents
            atype, q, c = agent_data[id]
            row[Symbol(id * "_obj")] = c !== nothing ? c[jh, jd, jy] : 0.0
            if q !== nothing
                for (k, v) in q
                    if v isa AbstractArray && ndims(v) == 3
                        row[Symbol(id * "_" * string(k))] = v[jh, jd, jy]
                    end
                end
            end
        end
        push!(ts_rows, row)
        t_idx += 1
    end
    ts_df = DataFrame(ts_rows)
    present = [c for c in col_order if c in propertynames(ts_df)]
    ts_df = select(ts_df, present)
    CSV.write(joinpath(results_folder, "Agent_Objectives_Per_Timestep.csv"), ts_df)

    write_sp_risk_outputs!(planner, planner_state, mdict, agents, results_folder)
    cost_metrics = collect_cost_metrics(mdict, agents; planner_state=planner_state,
                                        λ_elec=λ_elec, λ_H2=λ_H2, λ_EP=λ_EP)
    write_cost_metrics_csv(cost_metrics, results_folder)
    with_run_summary_log(results_folder) do
        print_social_planner_run_summary!(prices_df, var_dict, agents, JY,
                                          power_vres, H2_producers, offtaker_green;
                                          results_dir=results_folder,
                                          solver_status=get(planner_state, :solver_status, termination_status(planner)))
        print_cost_metrics_summary!(cost_metrics)
    end

    return nothing
end
