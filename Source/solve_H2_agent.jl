# ==============================================================================
# solve_H2_agent.jl — Re-set objective and solve electrolyzer
# ==============================================================================
#
# PURPOSE:
#   After ADMM_subroutine has updated λ, g_bar, ρ for elec, elec_GC, H2, H2_GC
#   on the model, this re-builds the objective (cost - revenue + penalties) and
#   calls optimize!(mod). Physical constraints from build_H2_agent! (conversion,
#   GC limits, capacity, annual green-backing) are invariant across iterations.
#
#   Because the H2 producer implements CVaR, the per-year loss expressions
#   (which involve λ_elec, λ_elec_GC, λ_H2, λ_H2_GC) must be recomputed each
#   iteration — JuMP expressions freeze coefficient values at creation time.
#   The CVaR shortfall and linking constraints that reference these losses are
#   therefore deleted and re-added with the freshly computed loss expressions.
#
# ==============================================================================

function solve_H2_agent!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict)
    # ── Index sets, weights, and ADMM parameters ──────────────────────────
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]
    op_cost = mod.ext[:parameters][:OperationalCost]

    # ADMM dual prices (λ), consensus targets (ḡ), and penalty weights (ρ)
    # for each market the electrolyzer participates in.
    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2  = mod.ext[:parameters][:g_bar_H2]
    ρ_H2      = mod.ext[:parameters][:ρ_H2]
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]

    # ── Decision variables (created once in build_H2_agent!) ──────────────
    e_in      = mod.ext[:variables][:e_in]
    h2_out    = mod.ext[:variables][:h2_out]
    q_elec_gc = mod.ext[:variables][:q_elec_gc]
    q_h2gc    = mod.ext[:variables][:q_h2gc]
    cap_H2_y  = mod.ext[:variables][:cap_H2_y]

    # ── Risk parameters and CVaR auxiliary variables ──────────────────────
    gamma     = get(mod.ext[:parameters], :γ, 1.0)    # risk weight (1 = risk-neutral)
    F_cap     = electrolyzer_h2_annuity(mod.ext[:parameters])
    alpha_H2  = mod.ext[:variables][:alpha_H2]         # VaR proxy
    cvar_H2   = mod.ext[:variables][:CVaR_H2]          # CVaR of loss
    u_H2      = mod.ext[:variables][:u_H2]             # shortfall per scenario year
    beta_conf = get(mod.ext[:parameters], :β, 0.95)    # CVaR confidence level
    P         = mod.ext[:parameters][:P]               # scenario probabilities

    # ── Recompute per-year loss expressions with current λ ────────────────
    # loss_H2[jy] = per-year operational loss (procurement + op cost minus sales).
    # loss_total[jy] = loss_H2[jy] + F_cap·cap_H2_y[jy] = FULL per-year loss.
    # CRITICAL: CVaR must use loss_total so that with nYears=1, γ has no effect.
    # JuMP expressions freeze coefficient values at creation time; rebuild each iter.
    loss_H2 = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_H2[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy]       * e_in[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
        loss_total[jy] = @expression(mod, loss_H2[jy] + F_cap * cap_H2_y)
    end
    mod.ext[:expressions][:loss_H2] = loss_H2

    z_cap   = get(mod.ext[:parameters], :z_cap, 0.0)
    λ_cap   = get(mod.ext[:parameters], :λ_cap, 0.0)
    ρ_cap   = get(mod.ext[:parameters], :ρ_cap, 0.1)
    cap_pen = haskey(mod.ext[:parameters], :z_cap) ?
        λ_cap * (cap_H2_y - z_cap) + ρ_cap/2 * (cap_H2_y - z_cap)^2 : 0.0
    mod.ext[:objective] = @objective(mod, Min,
        gamma * (F_cap * cap_H2_y + sum(P[jy] * loss_H2[jy] for jy in JY))
        + (1 - gamma) * cvar_H2
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy]         - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + cap_pen
    )

    # ── Delete stale CVaR constraints and re-add with fresh losses ────────
    # Shortfall uses loss_total (operational + F_cap·cap) so γ has no effect when nYears=1.
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_H2_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_H2_link])

    # Shortfall constraints: u_H2[jy] ≥ loss_total[jy] − α_H2.
    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_total[jy] - alpha_H2
    )
    # CVaR linking: CVaR_H2 ≥ α_H2 + (1/(1−β)) · Σ P[jy]·u_H2[jy].
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY)
    )

    optimize!(mod)
    return nothing
end
