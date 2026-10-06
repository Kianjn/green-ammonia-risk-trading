# ==============================================================================
# ADMM_subroutine_contracts.jl — Per-agent step in the contracts case (ME+C)
# ==============================================================================

if !isdefined(@__MODULE__, :repeat_last_agent_quantities!)
    include(joinpath(@__DIR__, "cap_admm_helpers.jl"))
end

# PURPOSE:
#   Same three blocks as ADMM_subroutine.jl — (1) refresh λ / ḡ / ρ of every
#   spot market the agent trades in, (2) solve the agent, (3) record its net
#   positions — plus the bilateral contract links the agent is party to.
#
#   Each contract link ℓ (one per seller) is a scalar sharing-ADMM market for
#   the contract capacity C_ℓ. The seller's net position is +C_sell, each
#   buyer's is −C_buy, and clearing means C_sell − Σ_b C_buy,b = 0. For agent m
#   on link ℓ the refreshed parameters are
#
#     K_ℓ      current contract price          results["K_<ppa|hpa>"][ℓ][end]
#     C̄_m,ℓ    consensus target for m's NET position:
#                own_prev_net − imb_ℓ,prev / (n_ℓ + 1)     (n_ℓ = 1 + #buyers)
#     ρ_ℓ      penalty weight                  ADMM_state["<ppa|hpa>"]["ρ"][ℓ][end]
#
#   On the first iteration previous positions and imbalances are zero.
#
#   After the solve the agent's contract volumes are pushed to
#   results["C_ppa_sell"][v], results["C_ppa_buy"][b][v], results["C_hpa_sell"][h],
#   results["C_hpa_buy"][b][h] so ADMM_contracts! can compute the link imbalances
#   and update K.
#
# ==============================================================================

function ADMM_subroutine_contracts!(m::String, data::Dict, results::Dict, ADMM_state::Dict,
                                    elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                    H2_GC_market::Dict, EP_market::Dict,
                                    ppa_market::Dict, hpa_market::Dict,
                                    mod::Model, agents::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    zeros_shp = zeros(n_ts, n_rd, n_yr)
    p = mod.ext[:parameters]
    agent_type = String(get(p, :Type, ""))

    # ------------------------------------------------------------------
    # 1. Spot-market parameters: ḡ = prev_own − imb/(n+1), λ, ρ  (as in ME)
    # ------------------------------------------------------------------
    @timeit TO "Update ADMM params" begin
        if p[:in_elec_market]
            n = elec_market["nAgents"]
            prev = isempty(results["g"][m]) ? zeros_shp : results["g"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec"]) ? zeros_shp : ADMM_state["Imbalances"]["elec"][end]
            p[:g_bar_elec] = prev .- (1.0 / (n + 1)) .* imb
            p[:λ_elec]     = results["λ"]["elec"][end]
            p[:ρ_elec]     = ADMM_state["ρ"]["elec"][end]
        end
        if p[:in_H2_market]
            n = H2_market["nAgents"]
            prev = isempty(results["h2"][m]) ? zeros_shp : results["h2"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2"]) ? zeros_shp : ADMM_state["Imbalances"]["H2"][end]
            p[:g_bar_H2] = prev .- (1.0 / (n + 1)) .* imb
            p[:λ_H2]     = results["λ"]["H2"][end]
            p[:ρ_H2]     = ADMM_state["ρ"]["H2"][end]
        end
        if p[:in_elec_GC_market]
            n = elec_GC_market["nAgents"]
            prev = isempty(results["elec_GC"][m]) ? zeros_shp : results["elec_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["elec_GC"][end]
            p[:g_bar_elec_GC] = prev .- (1.0 / (n + 1)) .* imb
            p[:λ_elec_GC]     = results["λ"]["elec_GC"][end]
            p[:ρ_elec_GC]     = ADMM_state["ρ"]["elec_GC"][end]
        end
        if p[:in_H2_GC_market]
            n = H2_GC_market["nAgents"]
            prev = isempty(results["H2_GC"][m]) ? zeros_shp : results["H2_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["H2_GC"][end]
            p[:g_bar_H2_GC] = prev .- (1.0 / (n + 1)) .* imb
            p[:λ_H2_GC]     = results["λ"]["H2_GC"][end]
            p[:ρ_H2_GC]     = ADMM_state["ρ"]["H2_GC"][end]
        end
        if p[:in_EP_market]
            n = EP_market["nAgents"]
            prev = isempty(results["EP"][m]) ? zeros_shp : results["EP"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["EP"]) ? zeros_shp : ADMM_state["Imbalances"]["EP"][end]
            p[:g_bar_EP] = prev .- (1.0 / (n + 1)) .* imb
            p[:λ_EP]     = results["λ"]["EP"][end]
            p[:ρ_EP]     = ADMM_state["ρ"]["EP"][end]
        end

        # Installed capacity is a private QP variable: no capacity split.
        if haskey(p, :z_cap)
            disable_installed_capacity_split!(p)
        end

        # --------------------------------------------------------------
        # 2. Contract-link parameters (K, C̄, ρ)
        # --------------------------------------------------------------
        if get(p, :in_ppa_market, false)
            S = ADMM_state["ppa"]
            n_c = S["n"]
            if agent_type == "VRES"
                # Seller: net position +C_sell.
                prev_C = isempty(results["C_ppa_sell"][m]) ? 0.0 : results["C_ppa_sell"][m][end]
                imb    = isempty(S["Imbalance"][m]) ? 0.0 : S["Imbalance"][m][end]
                p[:C_bar_ppa] = prev_C - imb / (n_c + 1)
                p[:K_ppa]     = results["K_ppa"][m][end]
                p[:ρ_ppa]     = S["ρ"][m][end]
            elseif haskey(results["C_ppa_buy"], m)
                # Buyer: net position −C_buy[v] on every VRES link.
                for v in S["ids"]
                    hist   = results["C_ppa_buy"][m][v]
                    prev_C = isempty(hist) ? 0.0 : hist[end]
                    imb    = isempty(S["Imbalance"][v]) ? 0.0 : S["Imbalance"][v][end]
                    p[:C_bar_ppa][v] = -prev_C - imb / (n_c + 1)
                    p[:K_ppa][v]     = results["K_ppa"][v][end]
                    p[:ρ_ppa][v]     = S["ρ"][v][end]
                end
            end
        end
        if get(p, :in_hpa_market, false)
            S = ADMM_state["hpa"]
            n_c = S["n"]
            if agent_type == "GreenProducer"
                # Seller: net position +C_hpa.
                prev_C = isempty(results["C_hpa_sell"][m]) ? 0.0 : results["C_hpa_sell"][m][end]
                imb    = isempty(S["Imbalance"][m]) ? 0.0 : S["Imbalance"][m][end]
                p[:C_bar_hpa] = prev_C - imb / (n_c + 1)
                p[:K_hpa]     = results["K_hpa"][m][end]
                p[:ρ_hpa]     = S["ρ"][m][end]
            elseif agent_type == "GreenOfftaker" && haskey(results["C_hpa_buy"], m)
                # Buyer: net position −C_buy[h] on every electrolyzer link.
                for h in S["ids"]
                    hist   = results["C_hpa_buy"][m][h]
                    prev_C = isempty(hist) ? 0.0 : hist[end]
                    imb    = isempty(S["Imbalance"][h]) ? 0.0 : S["Imbalance"][h][end]
                    p[:C_bar_hpa][h] = -prev_C - imb / (n_c + 1)
                    p[:K_hpa][h]     = results["K_hpa"][h][end]
                    p[:ρ_hpa][h]     = S["ρ"][h][end]
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # 3. Solve (contract-aware objectives; non-parties fall back to ME solves)
    # ------------------------------------------------------------------
    @timeit TO "Solve agent" begin
        if m in agents[:power]
            solve_power_agent_contracts!(m, mod, elec_market, elec_GC_market)
        elseif m in agents[:H2]
            solve_H2_agent_contracts!(m, mod, H2_market, H2_GC_market)
        elseif m in agents[:offtaker]
            solve_offtaker_agent_contracts!(m, mod, EP_market, H2_market, H2_GC_market)
        elseif m in agents[:elec_GC_demand]
            solve_elec_GC_demand_agent!(m, mod, elec_GC_market)
        end
    end

    # ------------------------------------------------------------------
    # 4. Result extraction (with the same failure path as ME)
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
        p[:in_elec_market]    && push!(results["g"][m],       collect(value.(mod.ext[:expressions][:g_net_elec])))
        p[:in_H2_market]      && push!(results["h2"][m],      collect(value.(mod.ext[:expressions][:g_net_H2])))
        p[:in_elec_GC_market] && push!(results["elec_GC"][m], collect(value.(mod.ext[:expressions][:g_net_elec_GC])))
        p[:in_H2_GC_market]   && push!(results["H2_GC"][m],   collect(value.(mod.ext[:expressions][:g_net_H2_GC])))
        p[:in_EP_market]      && push!(results["EP"][m],      collect(value.(mod.ext[:expressions][:g_net_EP])))

        if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
            push!(results["Cap_VRES"][m], [value(mod.ext[:variables][:cap_VRES])])
            push!(results["Inv_VRES"][m], [value(mod.ext[:variables][:inv_VRES])])
        end
        if haskey(mod.ext[:variables], :cap_H2_y) && haskey(mod.ext[:variables], :inv_cap_H2)
            push!(results["Cap_Elec_H2"][m], [value(mod.ext[:variables][:cap_H2_y])])
            push!(results["Inv_Elec_H2"][m], [value(mod.ext[:variables][:inv_cap_H2])])
        end
        if agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
            push!(results["Cap_EP_Green"][m], [value(mod.ext[:variables][:cap_EP_y])])
            push!(results["Inv_EP_Green"][m], [value(mod.ext[:variables][:inv_EP])])
        end

        # Contract volumes (scalar MW).
        if haskey(mod.ext[:variables], :C_ppa)
            push!(results["C_ppa_sell"][m], max(0.0, value(mod.ext[:variables][:C_ppa])))
        end
        if haskey(mod.ext[:variables], :C_ppa_buy) && haskey(results["C_ppa_buy"], m)
            for (v, var) in mod.ext[:variables][:C_ppa_buy]
                push!(results["C_ppa_buy"][m][v], max(0.0, value(var)))
            end
        end
        if haskey(mod.ext[:variables], :C_hpa)
            push!(results["C_hpa_sell"][m], max(0.0, value(mod.ext[:variables][:C_hpa])))
        end
        if haskey(mod.ext[:variables], :C_hpa_buy) && haskey(results["C_hpa_buy"], m)
            for (h, var) in mod.ext[:variables][:C_hpa_buy]
                push!(results["C_hpa_buy"][m][h], max(0.0, value(var)))
            end
        end
    end
    return nothing
end
