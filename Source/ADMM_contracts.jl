# ==============================================================================
# ADMM_contracts.jl — ADMM coordination loop for the contracts case (ME+C)
# ==============================================================================

if !isdefined(@__MODULE__, :repeat_last_agent_quantities!)
    include(joinpath(@__DIR__, "cap_admm_helpers.jl"))
end

# PURPOSE:
#   Same sharing-ADMM as ADMM.jl for the five spot markets (elec, elec_GC, H2,
#   H2_GC, EP), plus one scalar contract market per bilateral link:
#
#     PPA link v   (VRES v sells,           electrolyzers buy)   volume C_ppa,v   price K_ppa,v
#     HPA link h   (electrolyzer h sells,   green offtakers buy) volume C_hpa,h   price K_hpa,h
#
#   Each iteration:
#     1. Every agent solves its QP (ADMM_subroutine_contracts!) given the
#        current spot prices λ, contract prices K, consensus targets and ρ.
#     2. Spot imbalances, primal/dual residuals, price update
#        λ ← λ − η·ρ·imbalance (η damping, H2_GC floor)      — as in ADMM.jl.
#     3. Contract link imbalances  imb_ℓ = C_sell,ℓ − Σ_b C_buy,b,ℓ  (MW),
#        primal residual |imb_ℓ|, sharing-ADMM dual residual, and the contract
#        price update
#              K_ℓ ← max(0, K_ℓ − η_ℓ·ρ_ℓ·imb_ℓ).
#        Excess contract supply (seller offers more than buyers take) lowers
#        K; excess demand raises it — K is the dual of C_sell = Σ C_buy.
#     4. Residual balancing of every ρ (update_rho_contracts!).
#     5. Convergence when all five spot markets satisfy the Boyd criterion
#        (ε_abs·√n + ε_rel·scale) AND every contract link satisfies
#        |imb_ℓ| ≤ ε_cap + ε_rel·scale and dual ≤ the same (ε_cap in MW; a
#        contract volume is one scalar, so no √n factor).
#
#   The leftover installed-capacity bookkeeping (ADMM_state["Capacity"]) is
#   kept aligned with the iteration count (zero residuals, unchanged λ_cap) so
#   the shared summary printer and result writer keep working; installed
#   capacity is a private QP variable here exactly as in market_exposure.jl.
#
# ==============================================================================

function ADMM_contracts!(results::Dict, ADMM_state::Dict, elec_market::Dict, H2_market::Dict,
                         elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
                         ppa_market::Dict, hpa_market::Dict,
                         mdict::Dict, agents::Dict, data::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    max_iter = data["ADMM"]["max_iter"]
    convergence = 0
    iterations = ProgressBar(1:max_iter)

    flow_markets = ("elec", "H2", "elec_GC", "H2_GC", "EP")
    flow_results = Dict("elec" => "g", "H2" => "h2", "elec_GC" => "elec_GC", "H2_GC" => "H2_GC", "EP" => "EP")
    flow_agents  = Dict("elec" => agents[:elec_market], "H2" => agents[:H2_market],
                        "elec_GC" => agents[:elec_GC_market], "H2_GC" => agents[:H2_GC_market],
                        "EP" => agents[:EP_market])
    flow_n = Dict("elec" => elec_market["nAgents"], "H2" => H2_market["nAgents"],
                  "elec_GC" => elec_GC_market["nAgents"], "H2_GC" => H2_GC_market["nAgents"],
                  "EP" => EP_market["nAgents"])
    cap_agents = get(agents, :cap_agents, String[])

    ADMM_state["n_slots"] = n_ts * n_rd * n_yr
    ADMM_state["n_yr"]    = n_yr
    ADMM_state["rho_contract_max"] = get(get(data, "ADMM", Dict()), "rho_contract_max", 0.5)

    n_slots = n_ts * n_rd * n_yr
    sqrt_n  = sqrt(n_slots)
    eps_abs = ADMM_state["EpsilonAbs"]
    eps_rel = ADMM_state["EpsilonRel"]
    eps_cap = Float64(get(ADMM_state, "EpsilonCap", eps_abs))
    η_min   = 0.25

    # Per-market / per-link dual-step scaling adapted from merit movement.
    η_scale = Dict{String, Float64}(mkt => 1.0 for mkt in flow_markets)
    η_scale_c = Dict("ppa" => Dict{String, Float64}(id => 1.0 for id in ADMM_state["ppa"]["ids"]),
                     "hpa" => Dict{String, Float64}(id => 1.0 for id in ADMM_state["hpa"]["ids"]))

    # Boyd tolerances: flow market (per-slot ε_abs·√n) vs contract link (scalar ε_cap).
    function _flow_eps(mkt::String)
        sp = max(ADMM_state["ResidualScale"]["Primal"][mkt], 1.0)
        sd = max(ADMM_state["ResidualScale"]["Dual"][mkt], 1.0)
        return eps_abs * sqrt_n + eps_rel * sp, eps_abs * sqrt_n + eps_rel * sd
    end
    function _link_eps(S::Dict, id::String)
        sp = max(S["ResidualScale_Primal"][id], 1.0)
        sd = max(S["ResidualScale_Dual"][id], 1.0)
        return eps_cap + eps_rel * sp, eps_cap + eps_rel * sd
    end
    _merit(rp, rd, eps_pr, eps_du) = begin
        v = max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))
        isfinite(v) ? v : 1e12
    end
    _eta(rp, rd, eps_pr, eps_du) = begin
        base = max(rp, rd)
        eps_m = max(eps_pr, eps_du)
        base >= 1.5 * eps_m ? 1.0 : max(η_min, base / max(1.5 * eps_m, 1e-9))
    end

    # Net contract positions of every party on link `id` (seller +C, buyers −C);
    # `back` = 0 for the current iterate, 1 for the previous one.
    function _link_positions(ck::String, id::String, back::Int)
        sell_key = ck == "ppa" ? "C_ppa_sell" : "C_hpa_sell"
        buy_key  = ck == "ppa" ? "C_ppa_buy"  : "C_hpa_buy"
        x = Float64[]
        hs = results[sell_key][id]
        push!(x, length(hs) > back ? hs[end - back] : 0.0)
        for b in sort(collect(keys(results[buy_key])))
            hb = results[buy_key][b][id]
            push!(x, length(hb) > back ? -hb[end - back] : 0.0)
        end
        return x
    end

    for mkt in flow_markets
        push!(ADMM_state["PriceHistory"][mkt], mean(results["λ"][mkt][end]))
    end

    for iter in iterations
        convergence == 1 && break

        # ── 1. Agent solves ────────────────────────────────────────────
        for m in agents[:all]
            Base.invokelatest(ADMM_subroutine_contracts!, m, data, results, ADMM_state,
                              elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market,
                              ppa_market, hpa_market, mdict[m], agents, TO)
        end

        # ── 2. Spot markets: imbalances, residuals ─────────────────────
        @timeit TO "Compute imbalances" begin
            for mkt in flow_markets
                imb = sum(results[flow_results[mkt]][m][end] for m in flow_agents[mkt]; init = zeros(shp...))
                mkt == "EP" && (imb = imb .- EP_market["D_EP"])
                push!(ADMM_state["Imbalances"][mkt], imb)
                push!(ADMM_state["ImbalanceMean"][mkt], mean(imb))
                rp = sqrt(sum(imb .^ 2))
                push!(ADMM_state["Residuals"]["Primal"][mkt], rp)
                if ADMM_state["ResidualScale"]["Primal"][mkt] == 0.0 && rp > 0.0
                    ADMM_state["ResidualScale"]["Primal"][mkt] = rp
                end
            end
        end

        @timeit TO "Dual residuals" begin
            for mkt in flow_markets
                if iter > 1
                    ids = flow_agents[mkt]
                    n = flow_n[mkt]
                    key = flow_results[mkt]
                    ρ = ADMM_state["ρ"][mkt][end]
                    mean_now  = sum(results[key][ms][end]   for ms in ids; init = zeros(shp...)) ./ (n + 1)
                    mean_prev = sum(results[key][ms][end-1] for ms in ids; init = zeros(shp...)) ./ (n + 1)
                    d = 0.0
                    for m in ids
                        diff = (results[key][m][end] .- mean_now) .- (results[key][m][end-1] .- mean_prev)
                        d += sum((ρ .* diff) .^ 2)
                    end
                    rd = sqrt(d)
                    push!(ADMM_state["Residuals"]["Dual"][mkt], rd)
                    if ADMM_state["ResidualScale"]["Dual"][mkt] == 0.0 && isfinite(rd) && rd > 0.0
                        ADMM_state["ResidualScale"]["Dual"][mkt] = rd
                    end
                else
                    push!(ADMM_state["Residuals"]["Dual"][mkt], Inf)
                end
            end
        end

        # ── 3. Contract links: imbalances, residuals ───────────────────
        @timeit TO "Contract residuals" begin
            for ck in ("ppa", "hpa")
                S = ADMM_state[ck]
                for id in S["ids"]
                    x_now = _link_positions(ck, id, 0)
                    imb = sum(x_now)                       # C_sell − Σ C_buy  (MW)
                    push!(S["Imbalance"][id], imb)
                    rp = abs(imb)
                    push!(S["Primal"][id], rp)
                    if S["ResidualScale_Primal"][id] == 0.0 && rp > 0.0
                        S["ResidualScale_Primal"][id] = rp
                    end
                    if iter > 1
                        x_prev = _link_positions(ck, id, 1)
                        n_c = S["n"]
                        ρ = S["ρ"][id][end]
                        mean_now  = sum(x_now)  / (n_c + 1)
                        mean_prev = sum(x_prev) / (n_c + 1)
                        d = 0.0
                        for i in eachindex(x_now)
                            diff = (x_now[i] - mean_now) - (x_prev[i] - mean_prev)
                            d += (ρ * diff)^2
                        end
                        rd = sqrt(d)
                        push!(S["Dual"][id], rd)
                        if S["ResidualScale_Dual"][id] == 0.0 && rd > 0.0
                            S["ResidualScale_Dual"][id] = rd
                        end
                    else
                        push!(S["Dual"][id], Inf)
                    end
                end
            end
        end

        # ── Leftover installed-capacity bookkeeping (private variable → zero residuals) ──
        cap_state = ADMM_state["Capacity"]
        for m in cap_agents
            push!(cap_state["Primal"][m], 0.0)
            push!(cap_state["Dual"][m], iter > 1 ? 0.0 : Inf)
            push!(cap_state["λ"][m], copy(cap_state["λ"][m][end]))
        end
        push!(ADMM_state["Residuals"]["Primal"]["cap"], 0.0)
        push!(ADMM_state["Residuals"]["Dual"]["cap"], iter > 1 ? 0.0 : Inf)

        # ── 4. Step-scale adaptation from one-step merit movement ──────
        if iter > 1
            for mkt in flow_markets
                eps_pr, eps_du = _flow_eps(mkt)
                m_prev = _merit(ADMM_state["Residuals"]["Primal"][mkt][end-1],
                                ADMM_state["Residuals"]["Dual"][mkt][end-1], eps_pr, eps_du)
                m_now  = _merit(ADMM_state["Residuals"]["Primal"][mkt][end],
                                ADMM_state["Residuals"]["Dual"][mkt][end], eps_pr, eps_du)
                if m_now > 1.02 * m_prev
                    η_scale[mkt] = max(0.15, 0.85 * η_scale[mkt])
                elseif m_now < 0.98 * m_prev
                    η_scale[mkt] = min(1.0, 1.03 * η_scale[mkt])
                end
            end
            for ck in ("ppa", "hpa")
                S = ADMM_state[ck]
                for id in S["ids"]
                    eps_pr, eps_du = _link_eps(S, id)
                    m_prev = _merit(S["Primal"][id][end-1], S["Dual"][id][end-1], eps_pr, eps_du)
                    m_now  = _merit(S["Primal"][id][end],   S["Dual"][id][end],   eps_pr, eps_du)
                    if m_now > 1.02 * m_prev
                        η_scale_c[ck][id] = max(0.15, 0.85 * η_scale_c[ck][id])
                    elseif m_now < 0.98 * m_prev
                        η_scale_c[ck][id] = min(1.0, 1.03 * η_scale_c[ck][id])
                    end
                end
            end
        end

        # ── 5. Price updates ───────────────────────────────────────────
        @timeit TO "Update prices" begin
            # Spot: λ ← λ − η·ρ·imbalance (damped near tolerance), H2_GC ≥ 0.
            for mkt in flow_markets
                eps_pr, eps_du = _flow_eps(mkt)
                rp = ADMM_state["Residuals"]["Primal"][mkt][end]
                rd = ADMM_state["Residuals"]["Dual"][mkt][end]
                η = η_scale[mkt] * _eta(rp, rd, eps_pr, eps_du)
                push!(results["λ"][mkt],
                      results["λ"][mkt][end] .- η .* ADMM_state["ρ"][mkt][end] .* ADMM_state["Imbalances"][mkt][end])
            end
            results["λ"]["H2_GC"][end] .= max.(results["λ"]["H2_GC"][end], 0.0)

            # Contracts: K ← max(0, K − η·ρ·(C_sell − Σ C_buy)).
            # A fixed price below zero would mean the seller pays the buyer to
            # take green energy; no seller offers that, so K is projected to ≥ 0
            # exactly like the H2_GC floor.
            for ck in ("ppa", "hpa")
                S = ADMM_state[ck]
                K_key = ck == "ppa" ? "K_ppa" : "K_hpa"
                for id in S["ids"]
                    eps_pr, eps_du = _link_eps(S, id)
                    rp = S["Primal"][id][end]
                    rd = S["Dual"][id][end]
                    η = η_scale_c[ck][id] * _eta(rp, rd, eps_pr, eps_du)
                    K_new = results[K_key][id][end] - η * S["ρ"][id][end] * S["Imbalance"][id][end]
                    push!(results[K_key][id], max(0.0, K_new))
                end
            end
        end

        for mkt in flow_markets
            push!(ADMM_state["PriceHistory"][mkt], mean(results["λ"][mkt][end]))
        end

        @timeit TO "Update ρ" begin
            update_rho_contracts!(ADMM_state, iter)
        end

        set_description(iterations, "")

        # ── 6. Convergence: all spot markets AND all contract links ────
        function within_tol_flow(mkt::String)
            eps_pr, eps_du = _flow_eps(mkt)
            return ADMM_state["Residuals"]["Primal"][mkt][end] <= eps_pr &&
                   ADMM_state["Residuals"]["Dual"][mkt][end]   <= eps_du
        end
        function within_tol_link(ck::String, id::String)
            S = ADMM_state[ck]
            eps_pr, eps_du = _link_eps(S, id)
            return S["Primal"][id][end] <= eps_pr && S["Dual"][id][end] <= eps_du
        end
        if all(within_tol_flow(mkt) for mkt in flow_markets) &&
           all(within_tol_link("ppa", id) for id in ADMM_state["ppa"]["ids"]) &&
           all(within_tol_link("hpa", id) for id in ADMM_state["hpa"]["ids"])
            convergence = 1
        end

        ADMM_state["n_iter"] = iter
    end

    println()
    ADMM_state["converged"] = (convergence == 1)
    if !ADMM_state["converged"]
        println("ADMM reached max_iter without convergence.")
    end
    return nothing
end
