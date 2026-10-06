# ==============================================================================
# solve_power_agent_contracts.jl — Re-set objective and solve power agents (ME+C)
# ==============================================================================
#
# PURPOSE:
#   For a VRES with a PPA, rebuild the risk-adjusted objective with the current
#   spot prices λ, the current contract price K_ppa, consensus targets and
#   penalties, then optimize!. Conventional and Consumer agents delegate to
#   solve_power_agent! (they hold no contracts).
#
#   VRES per-scenario loss (enters CVaR). The buyer pays this seller K_ppa on
#   the as-produced volume AF·C_ppa. Spot revenue stays on the residual
#   (g − AF·C_ppa). That is the same loss as full pool revenue on g plus the
#   transfer the buyer pays (see contract_settlement.jl):
#
#     loss_y = Σ_{h,d} W ( MC·g − λ_elec·g − λ_elec_GC·g )        pool, as in ME
#              − M_y · C_ppa                                       PPA cash received from the buyer
#
#   with M_y = Σ W (K_ppa − λ_elec − λ_elec_GC) AF   (€ per MW-year).
#   loss_total_y = loss_y + F_cap·cap_VRES.
#
#   Objective:
#     γ (F_cap cap + Σ_y P_y loss_y) + (1−γ) CVaR
#     + spot ADMM penalties on g                                   (as in ME)
#     + (ρ_ppa/2)·A_ppa·(C_ppa − C̄_ppa)²                           PPA volume consensus
#
#   The last term is the sharing-ADMM penalty of the contract market. The
#   seller's net position is +C_ppa; the electrolyzer's is −C_ppa_buy; ADMM
#   drives C_ppa = C_ppa_buy and updates K_ppa as the dual of that condition.
#   There is NO separate linear "λ·q" ADMM term: K_ppa itself is the price the
#   agent responds to, through M_y. Both terms vanish at consensus.
#
# ==============================================================================

function solve_power_agent_contracts!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict)
    p = mod.ext[:parameters]
    agent_type = String(get(p, :Type, ""))
    if !(agent_type == "VRES" && get(p, :in_ppa_market, false) && haskey(mod.ext[:variables], :C_ppa))
        return solve_power_agent!(m, mod, elec_market, elec_GC_market)
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = p[:W]
    P  = p[:P]
    AF = mod.ext[:timeseries][:AF]

    λ_elec     = p[:λ_elec];     g_bar_elec    = p[:g_bar_elec];    ρ_elec    = p[:ρ_elec]
    λ_elec_GC  = p[:λ_elec_GC];  g_bar_elec_GC = p[:g_bar_elec_GC]; ρ_elec_GC = p[:ρ_elec_GC]

    # Contract parameters refreshed by ADMM_subroutine_contracts!.
    K_ppa     = p[:K_ppa]
    C_bar_ppa = p[:C_bar_ppa]
    ρ_ppa     = p[:ρ_ppa]
    A_ppa     = p[:A_ppa]

    gamma     = get(p, :γ, 1.0)
    beta_conf = get(p, :β, 0.95)
    F_cap     = get(p, :FixedCost_per_MW, 0.0)
    MC        = p[:MarginalCost]

    cap_VRES   = mod.ext[:variables][:cap_VRES]
    g          = mod.ext[:variables][:g]
    C_ppa      = mod.ext[:variables][:C_ppa]
    alpha_VRES = mod.ext[:variables][:alpha_VRES]
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]
    u_VRES     = mod.ext[:variables][:u_VRES]

    # Per-scenario PPA settlement coefficient (€/MW-year): buyer pays M_y·C to the seller.
    M = contract_settlement_coeffs(W, K_ppa, ppa_bundle_spot(λ_elec, λ_elec_GC), AF, JH, JD, JY)
    mod.ext[:expressions][:ppa_settlement_coeff] = M

    # Per-scenario loss with the PPA settlement inside (so it is inside CVaR).
    loss_VRES = Dict{Int, JuMP.AffExpr}()
    loss_total = Dict{Int, JuMP.AffExpr}()
    for jy in JY
        loss_VRES[jy] = @expression(mod,
            sum(W[jd, jy] * (MC * g[jh, jd, jy]
                - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD)
            - M[jy] * C_ppa
        )
        loss_total[jy] = @expression(mod, loss_VRES[jy] + F_cap * cap_VRES)
    end
    mod.ext[:expressions][:loss_VRES] = loss_VRES

    # Installed capacity is private (disable_installed_capacity_split! zeroes λ_cap, ρ_cap).
    z_cap = get(p, :z_cap, 0.0); λ_cap = get(p, :λ_cap, 0.0); ρ_cap = get(p, :ρ_cap, 0.0)
    cap_pen = haskey(p, :z_cap) ? λ_cap * (cap_VRES - z_cap) + ρ_cap / 2 * (cap_VRES - z_cap)^2 : 0.0

    mod.ext[:objective] = @objective(mod, Min,
        gamma * (F_cap * cap_VRES + sum(P[jy] * loss_VRES[jy] for jy in JY))
        + (1 - gamma) * cvar_VRES
        + sum(ρ_elec / 2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC / 2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + ρ_ppa / 2 * A_ppa * (C_ppa - C_bar_ppa)^2
        + cap_pen
    )

    # CVaR constraints bake in λ and K: delete and re-add with the fresh losses.
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_VRES_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_VRES_link])
    mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
        u_VRES[jy] >= loss_total[jy] - alpha_VRES)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
        cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
