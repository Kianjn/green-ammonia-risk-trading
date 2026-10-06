# ==============================================================================
# solve_H2_agent_contracts.jl — Re-set objective and solve electrolyzer (ME+C)
# ==============================================================================
#
# PURPOSE:
#   Rebuild the electrolyzer's risk-adjusted objective with current spot prices
#   λ, current contract prices K (one per PPA link, one for the HPA), consensus
#   targets and penalties, then optimize!.
#
#   Per-scenario loss (enters CVaR):
#
#     loss_y = Σ W ( λ_elec e_in + λ_elec_GC gc_e + op·h2 − λ_H2 h2 − λ_H2GC gc_h2 )   pool, as in ME
#              + Σ_v M^ppa_{v,y} · C_ppa_buy[v]        PPA cash PAID to VRES v
#              − M^hpa_y · C_hpa                      HPA cash RECEIVED from the offtaker
#
#   Each M·C is the bilateral fixed-price payment net of the spot value of the
#   same volume: the counterparty pays K on prof·C, and the pool term already
#   prices the physical MWh at λ (contract_settlement.jl), with
#     M^ppa_{v,y} = Σ W (K_ppa,v − λ_elec − λ_elec_GC) AF_v
#     M^hpa_y     = Σ W (K_hpa − λ_H2 − λ_H2_GC) .
#
#   Objective:
#     γ (F_cap cap + Σ P_y loss_y) + (1−γ) CVaR
#     + spot ADMM penalties                                  (as in ME)
#     + Σ_v (ρ_ppa,v/2)·A_ppa,v·(−C_ppa_buy[v] − C̄_ppa,v)²    PPA volume consensus (buyer: −C)
#     + (ρ_hpa/2)·A_hpa·(C_hpa − C̄_hpa)²                     HPA volume consensus (seller: +C)
#
# ==============================================================================

function solve_H2_agent_contracts!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict)
    p = mod.ext[:parameters]
    has_ppa = get(p, :in_ppa_market, false) && haskey(mod.ext[:variables], :C_ppa_buy) &&
              !isempty(mod.ext[:variables][:C_ppa_buy])
    has_hpa = get(p, :in_hpa_market, false) && haskey(mod.ext[:variables], :C_hpa)
    if !(has_ppa || has_hpa)
        return solve_H2_agent!(m, mod, H2_market, H2_GC_market)
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = p[:W]
    P  = p[:P]
    op_cost = p[:OperationalCost]

    λ_elec    = p[:λ_elec];    g_bar_elec    = p[:g_bar_elec];    ρ_elec    = p[:ρ_elec]
    λ_elec_GC = p[:λ_elec_GC]; g_bar_elec_GC = p[:g_bar_elec_GC]; ρ_elec_GC = p[:ρ_elec_GC]
    λ_H2      = p[:λ_H2];      g_bar_H2      = p[:g_bar_H2];      ρ_H2      = p[:ρ_H2]
    λ_H2_GC   = p[:λ_H2_GC];   g_bar_H2_GC   = p[:g_bar_H2_GC];   ρ_H2_GC   = p[:ρ_H2_GC]

    e_in      = mod.ext[:variables][:e_in]
    h2_out    = mod.ext[:variables][:h2_out]
    q_elec_gc = mod.ext[:variables][:q_elec_gc]
    q_h2gc    = mod.ext[:variables][:q_h2gc]
    cap_H2_y  = mod.ext[:variables][:cap_H2_y]

    gamma     = get(p, :γ, 1.0)
    beta_conf = get(p, :β, 0.95)
    F_cap     = electrolyzer_h2_annuity(p)
    alpha_H2  = mod.ext[:variables][:alpha_H2]
    cvar_H2   = mod.ext[:variables][:CVaR_H2]
    u_H2      = mod.ext[:variables][:u_H2]

    # ── Contract settlement coefficients (€/MW-year per scenario) ─────────
    # PPA (buyer side): pays M^ppa_{v,y} · C_ppa_buy[v] to VRES v.
    M_ppa = Dict{String, Dict{Int, Float64}}()
    C_ppa_buy = has_ppa ? mod.ext[:variables][:C_ppa_buy] : Dict{String, VariableRef}()
    if has_ppa
        λb_e = ppa_bundle_spot(λ_elec, λ_elec_GC)
        for v in keys(C_ppa_buy)
            M_ppa[v] = contract_settlement_coeffs(W, p[:K_ppa][v], λb_e, p[:AF_ppa][v], JH, JD, JY)
        end
    end
    # HPA (seller side): receives M^hpa_y · C_hpa from the offtaker.
    M_hpa = Dict{Int, Float64}(jy => 0.0 for jy in JY)
    if has_hpa
        shp = (length(JH), length(JD), length(JY))
        M_hpa = contract_settlement_coeffs(W, p[:K_hpa], hpa_bundle_spot(λ_H2, λ_H2_GC),
                                           flat_contract_profile(shp), JH, JD, JY)
    end
    mod.ext[:expressions][:ppa_settlement_coeff] = M_ppa
    mod.ext[:expressions][:hpa_settlement_coeff] = M_hpa

    # ── Per-scenario loss with both contracts inside (inside CVaR) ────────
    loss_H2 = Dict{Int, JuMP.AffExpr}()
    loss_total = Dict{Int, JuMP.AffExpr}()
    for jy in JY
        pool = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy]      * e_in[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy]      * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
            ) for jh in JH, jd in JD))
        ppa_paid = has_ppa ? @expression(mod, sum(M_ppa[v][jy] * C_ppa_buy[v] for v in keys(C_ppa_buy))) : 0.0
        hpa_recv = has_hpa ? @expression(mod, M_hpa[jy] * mod.ext[:variables][:C_hpa]) : 0.0
        loss_H2[jy] = @expression(mod, pool + ppa_paid - hpa_recv)
        loss_total[jy] = @expression(mod, loss_H2[jy] + F_cap * cap_H2_y)
    end
    mod.ext[:expressions][:loss_H2] = loss_H2

    # ── Contract-volume consensus penalties ───────────────────────────────
    pen_ppa = has_ppa ?
        @expression(mod, sum(p[:ρ_ppa][v] / 2 * p[:A_ppa][v] * (-C_ppa_buy[v] - p[:C_bar_ppa][v])^2
                              for v in keys(C_ppa_buy))) : 0.0
    pen_hpa = has_hpa ?
        @expression(mod, p[:ρ_hpa] / 2 * p[:A_hpa] * (mod.ext[:variables][:C_hpa] - p[:C_bar_hpa])^2) : 0.0

    # Installed capacity is private (λ_cap, ρ_cap zeroed each iteration).
    z_cap = get(p, :z_cap, 0.0); λ_cap = get(p, :λ_cap, 0.0); ρ_cap = get(p, :ρ_cap, 0.0)
    cap_pen = haskey(p, :z_cap) ? λ_cap * (cap_H2_y - z_cap) + ρ_cap / 2 * (cap_H2_y - z_cap)^2 : 0.0

    mod.ext[:objective] = @objective(mod, Min,
        gamma * (F_cap * cap_H2_y + sum(P[jy] * loss_H2[jy] for jy in JY))
        + (1 - gamma) * cvar_H2
        + sum(ρ_elec / 2    * W[jd, jy] * ((-e_in[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2    for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC / 2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2 / 2      * W[jd, jy] * (h2_out[jh, jd, jy]       - g_bar_H2[jh, jd, jy])^2      for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC / 2   * W[jd, jy] * (q_h2gc[jh, jd, jy]       - g_bar_H2_GC[jh, jd, jy])^2   for jh in JH, jd in JD, jy in JY)
        + pen_ppa + pen_hpa
        + cap_pen
    )

    # CVaR constraints bake in λ and K: delete and re-add with the fresh losses.
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_H2_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_H2_link])
    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_total[jy] - alpha_H2)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
