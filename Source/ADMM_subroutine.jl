# ==============================================================================
# ADMM_subroutine.jl — Per-agent step: update params, solve, record quantities
# ==============================================================================

if !isdefined(@__MODULE__, :repeat_last_agent_quantities!)
    include(joinpath(@__DIR__, "cap_admm_helpers.jl"))
end
#
# PURPOSE:
#   For one agent m: (1) For each market the agent participates in, set
#   g_bar = previous_own_quantity - (1/(n+1))*previous_imbalance, λ = current
#   price 3D array, ρ = current scalar. On first iteration, previous quantity
#   or imbalance may be missing so we use zeros. (2) Call the appropriate
#   solve_*_agent! so the model is optimized with the new objective. (3) Read
#   the solution (value of g_net_elec, g_net_H2, etc.) and push! into
#   results["g"][m], results["h2"][m], etc. so the main ADMM loop can compute
#   imbalances.
#
# ARGUMENTS:
#   m — Agent ID (string).
#   data — Must contain data["General"] with nTimesteps, nReprDays, nYears.
#   results — Dict with λ (price history) and per-agent quantity lists; we read
#     the last price and last quantity, and we push new quantities.
#   ADMM_state — Imbalances (last iteration), ρ (current); we read them to set g_bar.
#   elec_market, ... — For nAgents (consensus denominator).
#   mod — The JuMP model for this agent (we overwrite ext[:parameters] and optimize).
#   agents — To dispatch which solve_* to call and which result keys to push.
#   TO — TimerOutput for timing the three blocks (update params, solve, query).
#
# ==============================================================================

function ADMM_subroutine!(m::String, data::Dict, results::Dict, ADMM_state::Dict,
                           elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                           H2_GC_market::Dict, EP_market::Dict, mod::Model, agents::Dict, TO::TimerOutput)
    if m in get(agents, :merged, String[])
        return merged_admm_step!(m, data, results, ADMM_state,
            elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, mod, TO)
    end
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    zeros_shp = zeros(n_ts, n_rd, n_yr)

    # ------------------------------------------------------------------
    # Update ADMM parameters (g_bar, lambda, rho) on the agent's JuMP model
    # for every market the agent participates in.
    #
    # g_bar (consensus target) -- sharing-ADMM formula:
    #   g_bar = prev_own_quantity - (1/(n+1)) * prev_imbalance
    # Each agent's target is its own previous position adjusted by a
    # fraction of the total market imbalance. The 1/(n+1) factor
    # distributes the correction equally among n agent copies plus
    # one "market" copy (the sharing ADMM consensus variable).
    # When imbalance = 0, g_bar = prev_own_quantity (no correction).
    #
    # On the first iteration, previous quantities and imbalances are
    # empty, so we fall back to zeros (agents start unconstrained).
    # ------------------------------------------------------------------
    @timeit TO "Update ADMM params" begin
        if mod.ext[:parameters][:in_elec_market]
            n = elec_market["nAgents"]
            prev_g = isempty(results["g"][m]) ? zeros_shp : results["g"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec"]) ? zeros_shp : ADMM_state["Imbalances"]["elec"][end]
            # g_bar_elec = prev_own - (1/(n+1))*imbalance  (consensus target)
            mod.ext[:parameters][:g_bar_elec] = prev_g .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_elec]    = results["λ"]["elec"][end]
            mod.ext[:parameters][:ρ_elec]    = ADMM_state["ρ"]["elec"][end]
        end
        if mod.ext[:parameters][:in_H2_market]
            n = H2_market["nAgents"]
            prev = isempty(results["h2"][m]) ? zeros_shp : results["h2"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2"]) ? zeros_shp : ADMM_state["Imbalances"]["H2"][end]
            # g_bar_H2 = prev_own - (1/(n+1))*imbalance  (consensus target)
            mod.ext[:parameters][:g_bar_H2] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_H2]    = results["λ"]["H2"][end]
            mod.ext[:parameters][:ρ_H2]    = ADMM_state["ρ"]["H2"][end]
        end
        if mod.ext[:parameters][:in_elec_GC_market]
            n = elec_GC_market["nAgents"]
            prev = isempty(results["elec_GC"][m]) ? zeros_shp : results["elec_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["elec_GC"][end]
            # g_bar_elec_GC = prev_own - (1/(n+1))*imbalance  (consensus target)
            mod.ext[:parameters][:g_bar_elec_GC] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_elec_GC]     = results["λ"]["elec_GC"][end]
            mod.ext[:parameters][:ρ_elec_GC]     = ADMM_state["ρ"]["elec_GC"][end]
        end
        if mod.ext[:parameters][:in_H2_GC_market]
            n = H2_GC_market["nAgents"]
            prev = isempty(results["H2_GC"][m]) ? zeros_shp : results["H2_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["H2_GC"][end]
            # g_bar_H2_GC = prev_own - (1/(n+1))*imbalance  (consensus target)
            mod.ext[:parameters][:g_bar_H2_GC] = prev .- (1.0 / (n + 1)) .* imb
            # H2-GC prices are HOURLY (full 3D), same as all other markets.
            # Offtakers have temporal flexibility: they choose WHEN to buy GCs
            # (buying more when cheap, e.g. solar hours) while satisfying their
            # annual mandate constraint internally. This makes H2_GC a proper
            # hourly market that ADMM can clear naturally.
            mod.ext[:parameters][:λ_H2_GC] = results["λ"]["H2_GC"][end]
            mod.ext[:parameters][:ρ_H2_GC] = ADMM_state["ρ"]["H2_GC"][end]
        end
        if mod.ext[:parameters][:in_EP_market]
            n = EP_market["nAgents"]
            prev = isempty(results["EP"][m]) ? zeros_shp : results["EP"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["EP"]) ? zeros_shp : ADMM_state["Imbalances"]["EP"][end]
            # g_bar_EP = prev_own - (1/(n+1))*imbalance  (consensus target)
            mod.ext[:parameters][:g_bar_EP] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_EP]    = results["λ"]["EP"][end]
            mod.ext[:parameters][:ρ_EP]    = ADMM_state["ρ"]["EP"][end]
        end

        # Installed capacity is a private QP variable (Alessio). ADMM
        # coordinates energy only: do not penalize (cap − peak(g/AF)).
        if haskey(mod.ext[:parameters], :z_cap)
            disable_installed_capacity_split!(mod.ext[:parameters])
        end
    end

    # ------------------------------------------------------------------
    # Solve dispatch: route to the correct solve_*_agent! function
    # based on the agent group the agent belongs to. Each solve
    # function rebuilds only the objective (variables and constraints
    # are invariant across ADMM iterations) and calls optimize!.
    # ------------------------------------------------------------------
    @timeit TO "Solve agent" begin
        if m in agents[:power]
            solve_power_agent!(m, mod, elec_market, elec_GC_market)
        elseif m in agents[:H2]
            solve_H2_agent!(m, mod, H2_market, H2_GC_market)
        elseif m in agents[:offtaker]
            solve_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)
        elseif m in agents[:elec_GC_demand]
            solve_elec_GC_demand_agent!(m, mod, elec_GC_market)
        end
    end

    # ------------------------------------------------------------------
    # Result extraction. A failed Gurobi QP (common with a wide CVaR tail,
    # e.g. beta=0.6 after many objective rebuilds) has 0 solutions; value()
    # would throw. Retry, then reuse the previous ADMM iterate if needed.
    # ------------------------------------------------------------------
    ok = has_values(mod)
    ok || (ok = Base.invokelatest(ensure_agent_solution!, mod, m))
    if !ok
        reused = repeat_last_agent_quantities!(results, m, mod)
        reused || error("Agent $(m) has no primal and no previous ADMM iterate to reuse " *
                        "(termination=$(termination_status(mod)), primal=$(primal_status(mod))).")
        dampen_rhos_on_numerical!(ADMM_state, m)
        reset_gurobi_optimizer!(mod)
        return nothing
    end
    @timeit TO "Query results" begin
        if mod.ext[:parameters][:in_elec_market]
            g = collect(value.(mod.ext[:expressions][:g_net_elec]))
            push!(results["g"][m], g)
        end
        if mod.ext[:parameters][:in_H2_market]
            h2 = collect(value.(mod.ext[:expressions][:g_net_H2]))
            push!(results["h2"][m], h2)
        end
        if mod.ext[:parameters][:in_elec_GC_market]
            gc = collect(value.(mod.ext[:expressions][:g_net_elec_GC]))
            push!(results["elec_GC"][m], gc)
        end
        if mod.ext[:parameters][:in_H2_GC_market]
            h2gc = collect(value.(mod.ext[:expressions][:g_net_H2_GC]))
            push!(results["H2_GC"][m], h2gc)
        end
        if mod.ext[:parameters][:in_EP_market]
            ep = collect(value.(mod.ext[:expressions][:g_net_EP]))
            push!(results["EP"][m], ep)
        end

        # Capacity and investment results for green agents (per year, 1D vectors).
        agent_type = String(get(mod.ext[:parameters], :Type, ""))
        if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
            cap_vec = [value(mod.ext[:variables][:cap_VRES])]
            inv_vec = [value(mod.ext[:variables][:inv_VRES])]
            push!(results["Cap_VRES"][m], cap_vec)
            push!(results["Inv_VRES"][m], inv_vec)
        end
        if (haskey(mod.ext[:variables], :cap_H2_y) && haskey(mod.ext[:variables], :inv_cap_H2))
            cap_vec = [value(mod.ext[:variables][:cap_H2_y])]
            inv_vec = [value(mod.ext[:variables][:inv_cap_H2])]
            push!(results["Cap_Elec_H2"][m], cap_vec)
            push!(results["Inv_Elec_H2"][m], inv_vec)
        end
        if agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
            cap_vec = [value(mod.ext[:variables][:cap_EP_y])]
            inv_vec = [value(mod.ext[:variables][:inv_EP])]
            push!(results["Cap_EP_Green"][m], cap_vec)
            push!(results["Inv_EP_Green"][m], inv_vec)
        end
    end
    return nothing
end
