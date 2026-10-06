# ==============================================================================
# print_run_summary.jl — Formatted console summary for all entry points
# ==============================================================================

import Printf: @sprintf, @printf

const _MARKET_LABELS = Dict(
    "elec"    => "Electricity",
    "H2"      => "Hydrogen",
    "elec_GC" => "Electricity_GC",
    "H2_GC"   => "H2_GC",
    "EP"      => "End_Product",
)

function _fmt_res(x::Real)
    return @sprintf("%.3e", x)
end

function _fmt_price(x::Real)
    return @sprintf("%10.4f", x)
end

function _print_rule(title::String)
    w = 72
    println()
    println("-" ^ w)
    println("  ", title)
    println("-" ^ w)
end

function _agent_capacity_meta(id::String, agents::Dict)
    power = get(agents, :power, String[])
    h2 = get(agents, :H2, String[])
    offtaker = get(agents, :offtaker, String[])
    if id in power
        return "VRES", "MW"
    elseif id in h2
        return "Electrolyzer", "MW_H2"
    elseif id in offtaker
        return "EP plant", "MW_EP"
    end
    return "Capacity", "MW"
end

function _admm_capacity_row(results::Dict, id::String, agents::Dict)
    cap_vec = Float64[]
    inv_vec = Float64[]
    if haskey(results, "Cap_VRES") && !isempty(get(results["Cap_VRES"], id, []))
        cap_vec = results["Cap_VRES"][id][end]
        inv_hist = get(results["Inv_VRES"], id, [])
        inv_vec = isempty(inv_hist) ? Float64[] : inv_hist[end]
    elseif haskey(results, "Cap_Elec_H2") && !isempty(get(results["Cap_Elec_H2"], id, []))
        cap_vec = results["Cap_Elec_H2"][id][end]
        inv_hist = get(results["Inv_Elec_H2"], id, [])
        inv_vec = isempty(inv_hist) ? Float64[] : inv_hist[end]
    elseif haskey(results, "Cap_EP_Green") && !isempty(get(results["Cap_EP_Green"], id, []))
        cap_vec = results["Cap_EP_Green"][id][end]
        inv_hist = get(results["Inv_EP_Green"], id, [])
        inv_vec = isempty(inv_hist) ? Float64[] : inv_hist[end]
    end
    cap_final = isempty(cap_vec) ? 0.0 : cap_vec[end]
    inv_total = isempty(inv_vec) ? 0.0 : sum(inv_vec)
    role, unit = _agent_capacity_meta(id, agents)
    return (id = id, role = role, unit = unit, cap_final = cap_final,
            inv_total = inv_total, cap_vec = cap_vec)
end

const _MERGED_SLOT_META = Dict(
    "VRES_solar" => ("Solar", "MW"),
    "VRES_wind"  => ("Wind", "MW"),
    "H2"         => ("Electrolyzer", "MW_H2"),
    "EP"         => ("EP plant", "MW_EP"),
)

function _admm_capacity_rows(results::Dict, id::String, agents::Dict)
    if haskey(results, "Cap_Merged") && !isempty(get(results["Cap_Merged"], id, []))
        cap_vec = results["Cap_Merged"][id][end]
        inv_hist = get(get(results, "Inv_Merged", Dict()), id, [])
        inv_vec = isempty(inv_hist) ? zeros(length(cap_vec)) : copy(inv_hist[end])
        length(inv_vec) != length(cap_vec) && (inv_vec = zeros(length(cap_vec)))
        slots = get(get(results, "Cap_Merged_slots", Dict()), id,
                    ["slot$i" for i in 1:length(cap_vec)])
        rows = NamedTuple[]
        for i in 1:length(cap_vec)
            s = i <= length(slots) ? String(slots[i]) : "slot$i"
            role, unit = get(_MERGED_SLOT_META, s, (s, "MW"))
            label = "$(id) / $(role)"
            push!(rows, (id = label, role = role, unit = unit, cap_final = cap_vec[i],
                         inv_total = inv_vec[i], cap_vec = [cap_vec[i]]))
        end
        return rows
    end
    return [_admm_capacity_row(results, id, agents)]
end

function _print_capacity_table(rows::Vector)
    isempty(rows) && return
    println()
    @printf("  %-32s  %-14s  %8s  %12s  %s\n",
            "Agent", "Role", "Unit", "Final cap.", "New investment")
    @printf("  %-32s  %-14s  %8s  %12s  %s\n",
            repeat("-", 32), repeat("-", 14), repeat("-", 8),
            repeat("-", 12), repeat("-", 14))
    for r in rows
        cap_str = length(r.cap_vec) <= 1 ?
            @sprintf("%.2f", r.cap_final) :
            join([@sprintf("%.1f", c) for c in r.cap_vec], " → ")
        @printf("  %-32s  %-14s  %8s  %12s  %12.2f\n",
                r.id, r.role, r.unit, cap_str, r.inv_total)
    end
end

function _sp_capacity_rows(var_dict, agents, JY, power_vres, H2_producers, offtaker_green)
    rows = NamedTuple[]
    for id in vcat(collect(power_vres), collect(H2_producers), collect(offtaker_green))
        cap_vec = Float64[]
        inv_vec = Float64[]
        if id in power_vres && haskey(var_dict, :power_cap_VRES) && haskey(var_dict[:power_cap_VRES], id)
            cap_var = var_dict[:power_cap_VRES][id]
            cap_vec = [value(cap_var)]
            if haskey(var_dict, :power_inv_VRES) && haskey(var_dict[:power_inv_VRES], id)
                inv_var = var_dict[:power_inv_VRES][id]
                inv_vec = [value(inv_var)]
            end
        elseif id in H2_producers && haskey(var_dict, :H2_cap_elec) && haskey(var_dict[:H2_cap_elec], id)
            cap_var = var_dict[:H2_cap_elec][id]
            cap_vec = [value(cap_var)]
            if haskey(var_dict, :H2_inv_elec) && haskey(var_dict[:H2_inv_elec], id)
                inv_var = var_dict[:H2_inv_elec][id]
                inv_vec = [value(inv_var)]
            end
        elseif id in offtaker_green && haskey(var_dict, :offtaker_cap_EP_green) &&
               haskey(var_dict[:offtaker_cap_EP_green], id)
            cap_var = var_dict[:offtaker_cap_EP_green][id]
            cap_vec = [value(cap_var)]
            if haskey(var_dict, :offtaker_inv_EP_green) && haskey(var_dict[:offtaker_inv_EP_green], id)
                inv_var = var_dict[:offtaker_inv_EP_green][id]
                inv_vec = [value(inv_var)]
            end
        end
        isempty(cap_vec) && continue
        role, unit = _agent_capacity_meta(String(id), agents)
        cap_final = cap_vec[end]
        inv_total = isempty(inv_vec) ? 0.0 : sum(inv_vec)
        push!(rows, (id = String(id), role = role, unit = unit, cap_final = cap_final,
                     inv_total = inv_total, cap_vec = cap_vec))
    end
    return rows
end

"""Contract table for me_contracts.jl: one row per PPA / HPA link.

Shows the cleared contract capacity (seller offer vs. buyer demand — equal at
convergence), the fixed price K, and, when the relevant plant capacity is
known, the share of the seller's plant that is contracted.
"""
function _print_contract_table(results::Dict, ppa_market, hpa_market)
    rows = NamedTuple[]
    if ppa_market !== nothing && haskey(results, "K_ppa")
        for v in get(ppa_market, "ppa_vres", String[])
            haskey(results["K_ppa"], v) || continue
            C_s = isempty(results["C_ppa_sell"][v]) ? 0.0 : results["C_ppa_sell"][v][end]
            C_b = sum((isempty(results["C_ppa_buy"][b][v]) ? 0.0 : results["C_ppa_buy"][b][v][end])
                      for b in keys(results["C_ppa_buy"]); init = 0.0)
            cap_hist = get(get(results, "Cap_VRES", Dict()), v, [])
            cap = isempty(cap_hist) ? NaN : cap_hist[end][1]
            push!(rows, (label = "PPA_$(v)", unit = "MW", C_s = C_s, C_b = C_b,
                         K = results["K_ppa"][v][end], cap = cap))
        end
    end
    if hpa_market !== nothing && haskey(results, "K_hpa")
        for h in get(hpa_market, "hpa_h2", String[])
            haskey(results["K_hpa"], h) || continue
            C_s = isempty(results["C_hpa_sell"][h]) ? 0.0 : results["C_hpa_sell"][h][end]
            C_b = sum((isempty(results["C_hpa_buy"][b][h]) ? 0.0 : results["C_hpa_buy"][b][h][end])
                      for b in keys(results["C_hpa_buy"]); init = 0.0)
            cap_hist = get(get(results, "Cap_Elec_H2", Dict()), h, [])
            cap = isempty(cap_hist) ? NaN : cap_hist[end][1]
            push!(rows, (label = "HPA_$(h)", unit = "MW_H2", C_s = C_s, C_b = C_b,
                         K = results["K_hpa"][h][end], cap = cap))
        end
    end
    isempty(rows) && return
    println()
    println("  Bilateral contracts (fixed capacity, fixed price)")
    @printf("  %-18s  %6s  %12s  %12s  %12s  %10s\n",
            "Contract", "Unit", "Seller C", "Buyers C", "K [€/MWh]", "Share")
    @printf("  %-18s  %6s  %12s  %12s  %12s  %10s\n",
            repeat("-", 18), repeat("-", 6), repeat("-", 12), repeat("-", 12), repeat("-", 12), repeat("-", 10))
    for r in rows
        share = (isfinite(r.cap) && r.cap > 1e-9) ? @sprintf("%9.1f%%", 100 * r.C_s / r.cap) : "n/a"
        @printf("  %-18s  %6s  %12.2f  %12.2f  %12.4f  %10s\n",
                r.label, r.unit, r.C_s, r.C_b, r.K, share)
    end
end

function print_admm_run_summary!(ADMM_state::Dict, results::Dict, agents::Dict;
                                 results_dir::String,
                                 ppa_market::Union{Dict, Nothing}=nothing,
                                 hpa_market::Union{Dict, Nothing}=nothing)
    converged = get(ADMM_state, "converged", false)
    n_it = get(ADMM_state, "n_iter", 0)

    _print_rule("ADMM run summary")
    println("  Status:     ", converged ? "Converged" : "Stopped at max_iter")
    println("  Iterations: ", n_it)

    λ_elec    = results["λ"]["elec"][end]
    λ_H2      = results["λ"]["H2"][end]
    λ_elec_GC = results["λ"]["elec_GC"][end]
    λ_H2_GC   = results["λ"]["H2_GC"][end]
    λ_EP      = results["λ"]["EP"][end]
    shp = size(λ_elec)
    n_yr = shp[3]

    println()
    @printf("  %-16s  %10s  %10s  %12s\n", "Market", "Primal", "Dual", "Mean price")
    @printf("  %-16s  %10s  %10s  %12s\n",
            repeat("-", 16), repeat("-", 10), repeat("-", 10), repeat("-", 12))
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        price = mean(results["λ"][key][end])
        @printf("  %-16s  %10s  %10s  %12s\n",
                _MARKET_LABELS[key], _fmt_res(rp), _fmt_res(rd), _fmt_price(price))
    end

    # Bilateral contract links (me_contracts.jl): residuals are in MW of
    # contract capacity, the "price" is the fixed contract price K (€/MWh).
    if ppa_market !== nothing && haskey(ADMM_state, "ppa")
        S = ADMM_state["ppa"]
        for vres_id in get(ppa_market, "ppa_vres", String[])
            isempty(S["Primal"][vres_id]) && continue
            rp = S["Primal"][vres_id][end]
            rd = S["Dual"][vres_id][end]
            K  = results["K_ppa"][vres_id][end]
            @printf("  %-16s  %10s  %10s  %12s\n",
                    "PPA_$(vres_id)", _fmt_res(rp), _fmt_res(isfinite(rd) ? rd : 0.0), _fmt_price(K))
        end
    end
    if hpa_market !== nothing && haskey(ADMM_state, "hpa")
        S = ADMM_state["hpa"]
        for h2_id in get(hpa_market, "hpa_h2", String[])
            isempty(S["Primal"][h2_id]) && continue
            rp = S["Primal"][h2_id][end]
            rd = S["Dual"][h2_id][end]
            K  = results["K_hpa"][h2_id][end]
            @printf("  %-16s  %10s  %10s  %12s\n",
                    "HPA_$(h2_id)", _fmt_res(rp), _fmt_res(isfinite(rd) ? rd : 0.0), _fmt_price(K))
        end
    end

    cap_state = get(ADMM_state, "Capacity", Dict())
    cap_agents = get(cap_state, "agents", String[])
    if !isempty(cap_agents)
        rp_agg = ADMM_state["Residuals"]["Primal"]["cap"][end]
        rd_agg = ADMM_state["Residuals"]["Dual"]["cap"][end]
        @printf("  %-16s  %10s  %10s  %12s\n",
                "Capacity (agg.)", _fmt_res(rp_agg), _fmt_res(rd_agg), "n/a")
    end

    cap_rows = NamedTuple[]
    for m in cap_agents
        append!(cap_rows, _admm_capacity_rows(results, m, agents))
    end
    _print_capacity_table(cap_rows)

    if !isempty(cap_agents)
        println()
        @printf("  %-22s  %10s  %10s  %8s\n", "Agent", "Primal", "Dual", "ρ")
        @printf("  %-22s  %10s  %10s  %8s\n",
                repeat("-", 22), repeat("-", 10), repeat("-", 10), repeat("-", 8))
        worst_m = ""
        worst_v = -Inf
        for m in cap_agents
            rp_m = cap_state["Primal"][m][end]
            rd_m = cap_state["Dual"][m][end]
            ρ_m  = cap_state["ρ"][m][end]
            @printf("  %-22s  %10s  %10s  %8.3f\n",
                    m, _fmt_res(rp_m), _fmt_res(isfinite(rd_m) ? rd_m : 0.0), ρ_m)
            v = max(rp_m, isfinite(rd_m) ? rd_m : 0.0)
            if v > worst_v
                worst_v = v
                worst_m = m
            end
        end
        if worst_m != ""
            println("  Slowest capacity agent: ", worst_m,
                    " (max residual = ", _fmt_res(worst_v), ")")
        end
    end

    if ppa_market !== nothing || hpa_market !== nothing
        _print_contract_table(results, ppa_market, hpa_market)
    end

    sp_path = joinpath(@__DIR__, "..", "social_planner_results", "Market_Prices.csv")
    if isfile(sp_path)
        sp_df = CSV.read(sp_path, DataFrame)
        n_ts, n_rd = shp[1], shp[2]
        admm_slots = n_ts * n_rd * n_yr
        sp_slots = nrow(sp_df)
        use_base = (sp_slots == n_ts * n_rd) && (n_yr > 1)
        admm_elec    = use_base ? mean(λ_elec[:, :, 1])    : mean(λ_elec)
        admm_H2      = use_base ? mean(λ_H2[:, :, 1])      : mean(λ_H2)
        admm_elec_GC = use_base ? mean(λ_elec_GC[:, :, 1]) : mean(λ_elec_GC)
        admm_H2_GC   = use_base ? mean(λ_H2_GC[:, :, 1])   : mean(λ_H2_GC)
        admm_EP      = use_base ? mean(λ_EP[:, :, 1])      : mean(λ_EP)

        println()
        println("  Social planner benchmark",
                use_base ? " (ADMM base scenario jy=1)" :
                (sp_slots == admm_slots ? "" : " (all-scenario ADMM mean)"))
        @printf("  %-16s  %12s  %12s  %10s\n", "Market", "SP", "ADMM", "Δ")
        @printf("  %-16s  %12s  %12s  %10s\n",
                repeat("-", 16), repeat("-", 12), repeat("-", 12), repeat("-", 10))
        pairs = (
            ("Electricity",    mean(sp_df.Elec_Price),    admm_elec),
            ("Hydrogen",       mean(sp_df.H2_Price),      admm_H2),
            ("Electricity_GC", mean(sp_df.Elec_GC_Price), admm_elec_GC),
            ("H2_GC",          mean(sp_df.H2_GC_Price),   admm_H2_GC),
            ("End_Product",    mean(sp_df.EP_Price),      admm_EP),
        )
        for (label, sp_p, admm_p) in pairs
            @printf("  %-16s  %12.4f  %12.4f  %+10.4f\n", label, sp_p, admm_p, admm_p - sp_p)
        end
    end

    println()
    println("  Results saved to: ", results_dir)
    return nothing
end

function print_social_planner_run_summary!(prices_df::DataFrame, var_dict, agents::Dict,
                                           JY, power_vres, H2_producers, offtaker_green;
                                           results_dir::String, solver_status=nothing)
    _print_rule("Social planner run summary")
    status_label = solver_status === nothing ? "Optimal" : string(solver_status)
    println("  Status:     ", status_label)

    println()
    @printf("  %-16s  %12s\n", "Market", "Mean price")
    @printf("  %-16s  %12s\n", repeat("-", 16), repeat("-", 12))
    for (label, col) in (
        ("Electricity",    :Elec_Price),
        ("Hydrogen",       :H2_Price),
        ("Electricity_GC", :Elec_GC_Price),
        ("H2_GC",          :H2_GC_Price),
        ("End_Product",    :EP_Price),
    )
        @printf("  %-16s  %12s\n", label, _fmt_price(mean(prices_df[!, col])))
    end

    cap_rows = _sp_capacity_rows(var_dict, agents, JY, power_vres, H2_producers, offtaker_green)
    _print_capacity_table(cap_rows)

    println()
    println("  Results saved to: ", results_dir)
    return nothing
end
