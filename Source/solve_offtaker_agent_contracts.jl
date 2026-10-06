# ==============================================================================
# solve_offtaker_agent_contracts.jl — Re-set objective and solve offtakers (ME+C)
# ==============================================================================
#
# PURPOSE:
#   For the GreenOfftaker with an HPA, rebuild the risk-adjusted objective with
#   the current spot prices λ, the current HPA price(s) K_hpa, consensus targets
#   and penalties, then optimize!. Grey offtaker and importer delegate to
#   solve_offtaker_agent! (they hold no contracts).
#
#   Per-scenario loss (enters CVaR):
#
#     loss_y = Σ W ( λ_H2 h2_in + λ_H2GC gc + proc·ep − λ_EP ep )   pool, as in ME
#              + Σ_h M^hpa_{h,y} · C_hpa_buy[h]                     HPA cash PAID to electrolyzer h
#
#   with M^hpa_{h,y} = Σ W (K_hpa,h − λ_H2 − λ_H2_GC)  (baseload profile; contract_settlement.jl).
#   The offtaker pays the electrolyzer K_hpa on C_hpa_buy every hour. Pool
#   purchases stay on the physical intake; together the contracted MW_H2 cost K_hpa.
#
#   Objective:
#     γ (F_cap cap + Σ P_y loss_y) + (1−γ) CVaR
#     + spot ADMM penalties                                    (as in ME)
#     + Σ_h (ρ_hpa,h/2)·A_hpa,h·(−C_hpa_buy[h] − C̄_hpa,h)²      HPA volume consensus (buyer: −C)
#
# ==============================================================================

function solve_offtaker_agent_contracts!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)
    p = mod.ext[:parameters]
    agent_type = String(get(p, :Type, ""))
    has_hpa = agent_type == "GreenOfftaker" && get(p, :in_hpa_market, false) &&
              haskey(mod.ext[:variables], :C_hpa_buy) && !isempty(mod.ext[:variables][:C_hpa_buy])
    if !has_hpa
        return solve_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = p[:W]
    P  = p[:P]

    λ_H2    = p[:λ_H2];    g_bar_H2    = p[:g_bar_H2];    ρ_H2    = p[:ρ_H2]
    λ_H2_GC = p[:λ_H2_GC]; g_bar_H2_GC = p[:g_bar_H2_GC]; ρ_H2_GC = p[:ρ_H2_GC]
    λ_EP    = p[:λ_EP];    g_bar_EP    = p[:g_bar_EP];    ρ_EP    = p[:ρ_EP]

    h2_in     = mod.ext[:variables][:h2_in]
    q_h2gc    = mod.ext[:variables][:q_h2gc]
    ep        = mod.ext[:variables][:ep]
    cap_EP_y  = mod.ext[:variables][:cap_EP_y]
    C_hpa_buy = mod.ext[:variables][:C_hpa_buy]

    gamma_G   = get(p, :γ, 1.0)
    beta_conf = get(p, :β, 0.95)
    proc_cost = get(p, :ProcessingCost, 0.0)
    F_cap     = get(p, :FixedCost_per_MW_EP_Out, 0.0)
    alpha_G   = mod.ext[:variables][:alpha_GreenOfftaker]
    cvar_G    = mod.ext[:variables][:CVaR_GreenOfftaker]
    u_G       = mod.ext[:variables][:u_GreenOfftaker]

    # ── HPA settlement coefficients (buyer pays M_{h,y} · C to electrolyzer h) ──
    shp = (length(JH), length(JD), length(JY))
    λb_h = hpa_bundle_spot(λ_H2, λ_H2_GC)
    prof = flat_contract_profile(shp)
    M_hpa = Dict{String, Dict{Int, Float64}}()
    for h in keys(C_hpa_buy)
        M_hpa[h] = contract_settlement_coeffs(W, p[:K_hpa][h], λb_h, prof, JH, JD, JY)
    end
    mod.ext[:expressions][:hpa_settlement_coeff] = M_hpa

    # ── Per-scenario loss with the HPA inside (inside CVaR) ───────────────
    loss_G = Dict{Int, JuMP.AffExpr}()
    loss_total = Dict{Int, JuMP.AffExpr}()
    for jy in JY
        loss_G[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_H2[jh, jd, jy]      * h2_in[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy]    * ep[jh, jd, jy]
            ) for jh in JH, jd in JD)
            + sum(M_hpa[h][jy] * C_hpa_buy[h] for h in keys(C_hpa_buy))
        )
        loss_total[jy] = @expression(mod, loss_G[jy] + F_cap * cap_EP_y)
    end
    mod.ext[:expressions][:loss_GreenOfftaker] = loss_G

    pen_hpa = @expression(mod, sum(p[:ρ_hpa][h] / 2 * p[:A_hpa][h] * (-C_hpa_buy[h] - p[:C_bar_hpa][h])^2
                                   for h in keys(C_hpa_buy)))

    # Installed capacity is private (λ_cap, ρ_cap zeroed each iteration).
    z_cap = get(p, :z_cap, 0.0); λ_cap = get(p, :λ_cap, 0.0); ρ_cap = get(p, :ρ_cap, 0.0)
    cap_pen = haskey(p, :z_cap) ? λ_cap * (cap_EP_y - z_cap) + ρ_cap / 2 * (cap_EP_y - z_cap)^2 : 0.0

    mod.ext[:objective] = @objective(mod, Min,
        gamma_G * (F_cap * cap_EP_y + sum(P[jy] * loss_G[jy] for jy in JY))
        + (1 - gamma_G) * cvar_G
        + sum(ρ_H2 / 2    * W[jd, jy] * ((-h2_in[jh, jd, jy])  - g_bar_H2[jh, jd, jy])^2    for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC / 2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_EP / 2    * W[jd, jy] * (ep[jh, jd, jy]        - g_bar_EP[jh, jd, jy])^2    for jh in JH, jd in JD, jy in JY)
        + pen_hpa
        + cap_pen
    )

    # CVaR constraints bake in λ and K: delete and re-add with the fresh losses.
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_Green_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_Green_link])
    mod.ext[:constraints][:CVaR_Green_shortfall] = @constraint(mod, [jy in JY],
        u_G[jy] >= loss_total[jy] - alpha_G)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_Green_link] = @constraint(mod,
        cvar_G >= alpha_G + (1 / one_minus_beta) * sum(P[jy] * u_G[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
