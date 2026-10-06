# ==============================================================================
# solve_offtaker_agent.jl — Re-set objective and solve offtaker (green/grey/importer)
# ==============================================================================
#
# PURPOSE:
#   Re-builds the offtaker objective with current λ, g_bar, ρ and calls
#   optimize!(mod). Physical constraints from build_offtaker_agent! (stoichiometry,
#   GC mandates, capacity limits) are invariant across iterations.
#
#   - GreenOfftaker: cost (H2, H2_GC, processing) − revenue (EP) + penalties
#     + CVaR risk term. Because the GreenOfftaker implements CVaR, its per-year
#     loss expressions (which involve λ_H2, λ_H2_GC, λ_EP) must be recomputed
#     each iteration, and the CVaR shortfall/linking constraints are deleted and
#     re-added with the fresh losses.
#   - GreyOfftaker: cost (MC*ep, H2_GC) − revenue (EP) + penalties.
#   - EPImporter: cost (import) − revenue (EP) + penalty.
#
# ==============================================================================

function solve_offtaker_agent!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]
    gamma_GC = get(mod.ext[:parameters], :gamma_GC, 0.42)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2 = mod.ext[:parameters][:g_bar_H2]
    ρ_H2     = mod.ext[:parameters][:ρ_H2]
    # Hourly H2-GC price (full 3D), same as all other markets.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]
    λ_EP     = mod.ext[:parameters][:λ_EP]
    g_bar_EP = mod.ext[:parameters][:g_bar_EP]
    ρ_EP     = mod.ext[:parameters][:ρ_EP]

    if agent_type == "GreenOfftaker"
        # ── GreenOfftaker variables and parameters ────────────────────────
        h2_in     = mod.ext[:variables][:h2_in]
        q_h2gc    = mod.ext[:variables][:q_h2gc]
        ep        = mod.ext[:variables][:ep]
        cap_EP_y  = mod.ext[:variables][:cap_EP_y]
        gamma_G   = get(mod.ext[:parameters], :γ, 1.0)    # risk weight (1 = risk-neutral)
        proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)
        F_cap     = get(mod.ext[:parameters], :FixedCost_per_MW_EP_Out, 0.0)

        # CVaR auxiliary variables (created once in build_offtaker_agent!):
        # alpha_G = VaR proxy, cvar_G = CVaR of loss, u_G[jy] = shortfall.
        alpha_G   = mod.ext[:variables][:alpha_GreenOfftaker]
        cvar_G    = mod.ext[:variables][:CVaR_GreenOfftaker]
        u_G       = mod.ext[:variables][:u_GreenOfftaker]
        beta_conf = get(mod.ext[:parameters], :β, 0.95)    # CVaR confidence level
        P         = mod.ext[:parameters][:P]               # scenario probabilities

        # ── Recompute per-year loss expressions with current λ ────────────
        # loss_G[jy] = per-year operational loss (H₂ + GC + processing − EP revenue).
        # loss_total[jy] = loss_G[jy] + F_cap·cap_EP_y[jy] = FULL per-year loss.
        # CRITICAL: CVaR must use loss_total so that with nYears=1, γ has no effect.
        loss_G = Dict{Int,JuMP.AffExpr}()
        loss_total = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_G[jy] = @expression(mod,
                sum(W[jd, jy] * (
                    λ_H2[jh, jd, jy]        * h2_in[jh, jd, jy]
                    + λ_H2_GC[jh, jd, jy]  * q_h2gc[jh, jd, jy]
                    + proc_cost * ep[jh, jd, jy]
                    - λ_EP[jh, jd, jy]      * ep[jh, jd, jy]
                ) for jh in JH, jd in JD)
            )
            loss_total[jy] = @expression(mod, loss_G[jy] + F_cap * cap_EP_y)
        end
        mod.ext[:expressions][:loss_GreenOfftaker] = loss_G

        z_cap   = get(mod.ext[:parameters], :z_cap, 0.0)
        λ_cap   = get(mod.ext[:parameters], :λ_cap, 0.0)
        ρ_cap   = get(mod.ext[:parameters], :ρ_cap, 0.1)
        cap_pen = haskey(mod.ext[:parameters], :z_cap) ?
            λ_cap * (cap_EP_y - z_cap) + ρ_cap/2 * (cap_EP_y - z_cap)^2 : 0.0
        mod.ext[:objective] = @objective(mod, Min,
            gamma_G * (F_cap * cap_EP_y + sum(P[jy] * loss_G[jy] for jy in JY))
            + (1 - gamma_G) * cvar_G
            + sum(ρ_H2/2    * W[jd, jy] * ((-h2_in[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2     for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2    * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2           for jh in JH, jd in JD, jy in JY)
            + cap_pen
        )

        # ── Delete stale CVaR constraints and re-add with fresh losses ────
        # Shortfall uses loss_total (operational + F_cap·cap) so γ has no effect when nYears=1.
        for jy in JY
            delete(mod, mod.ext[:constraints][:CVaR_Green_shortfall][jy])
        end
        delete(mod, mod.ext[:constraints][:CVaR_Green_link])

        # Shortfall constraints: u_G[jy] ≥ loss_total[jy] − α_G.
        mod.ext[:constraints][:CVaR_Green_shortfall] = @constraint(mod, [jy in JY],
            u_G[jy] >= loss_total[jy] - alpha_G
        )
        # CVaR linking: CVaR_G ≥ α_G + (1/(1−β)) · Σ P[jy]·u_G[jy].
        one_minus_beta = max(1e-6, 1.0 - beta_conf)
        mod.ext[:constraints][:CVaR_Green_link] = @constraint(mod,
            cvar_G >= alpha_G + (1 / one_minus_beta) * sum(P[jy] * u_G[jy] for jy in JY)
        )

    elseif agent_type == "GreyOfftaker"
        # GreyOfftaker objective:
        #   min  sum W * ( MC*ep                        [marginal production cost]
        #                + lambda_H2_GC*q_h2gc          [H2 GC purchase cost (compliance)]
        #                - lambda_EP*ep )                [end-product revenue]
        #      + rho_H2_GC/2 * sum W * (-q_h2gc - g_bar_H2_GC)^2  [ADMM H2 GC penalty; buyer]
        #      + rho_EP/2    * sum W * (ep - g_bar_EP)^2            [ADMM EP penalty; seller]
        # Note: GreyOfftaker does NOT buy physical H2 (no H2 market penalty),
        # but must purchase H2 GCs for regulatory compliance.
        ep     = mod.ext[:variables][:ep]
        q_h2gc = mod.ext[:variables][:q_h2gc]
        MC     = get(mod.ext[:parameters], :MarginalCostByYear,
                     fill(mod.ext[:parameters][:MarginalCost], length(JY)))
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC[jy] * ep[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    else
        # Importer objective (fallback for any non-Green/Grey offtaker):
        #   min  sum W * ( imp_cost*ep                  [import cost per unit of EP]
        #                - lambda_EP*ep )                [end-product revenue]
        #      + rho_EP/2 * sum W * (ep - g_bar_EP)^2   [ADMM EP penalty; seller]
        # The importer participates only in the EP market (no H2 or H2_GC).
        ep = mod.ext[:variables][:ep]
        imp_cost = mod.ext[:parameters][:ImportCost]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    end
    optimize!(mod)
    return nothing
end
