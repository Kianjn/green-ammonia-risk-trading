# ==============================================================================
# solve_power_agent.jl — Re-set objective and solve power-sector agent
# ==============================================================================
#
# PURPOSE:
#   Called from ADMM_subroutine after the model's λ, g_bar, and ρ have been
#   updated from the current ADMM state. This function re-builds the objective
#   (min cost - revenue + ADMM penalties) using those updated parameters and
#   then calls optimize!(mod).
#
#   For VRES agents with CVaR, the per-year loss expressions (which depend on
#   λ_elec and λ_elec_GC) must also be recomputed, because JuMP expressions
#   bake in coefficient values at creation time. The CVaR shortfall and linking
#   constraints that reference these loss expressions are therefore deleted and
#   re-added each iteration with the fresh losses.
#
#   For Conventional and Consumer agents no CVaR logic is needed; only the
#   objective is rebuilt (for Conventional: stagewise convex variable-cost stack).
#
# ARGUMENTS:
#   m — Agent ID (used only for dispatch; parameters are on mod).
#   mod — JuMP model (ext[:parameters] already updated by ADMM_subroutine).
#   elec_market, elec_GC_market — Passed for interface consistency; nAgents used elsewhere.
#
# ==============================================================================

function solve_power_agent!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict)
    # Only the objective is rebuilt each ADMM iteration. Variables and
    # constraints were defined once in build_power_agent! and are invariant;
    # only the prices (lambda), consensus targets (g_bar), and penalty
    # parameter (rho) change between iterations.
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W   = mod.ext[:parameters][:W]
    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    agent_type = mod.ext[:parameters][:Type]

    if agent_type == "VRES"
        # ── VRES parameters and variables ─────────────────────────────────
        gamma      = get(mod.ext[:parameters], :γ, 1.0)   # risk weight (1 = risk-neutral)
        F_cap      = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        MC         = mod.ext[:parameters][:MarginalCost]
        cap_VRES   = mod.ext[:variables][:cap_VRES]
        g          = mod.ext[:variables][:g]

        # CVaR auxiliary variables (created once in build_power_agent!):
        # alpha_VRES = VaR proxy, cvar_VRES = CVaR of loss, u_VRES[jy] = shortfall.
        alpha_VRES = mod.ext[:variables][:alpha_VRES]
        cvar_VRES  = mod.ext[:variables][:CVaR_VRES]
        u_VRES     = mod.ext[:variables][:u_VRES]
        beta_conf  = get(mod.ext[:parameters], :β, 0.95)   # CVaR confidence level
        P          = mod.ext[:parameters][:P]               # scenario probabilities

        # ── Recompute per-year loss expressions with current λ ────────────
        # loss_VRES[jy] = Σ_{h,d} W[d,y]·( MC·g − λ_elec·g − λ_GC·g )
        #   = per-year operational loss (production cost minus market revenues).
        # loss_total[jy] = loss_VRES[jy] + F_cap·cap_VRES[jy] = FULL per-year loss.
        # CRITICAL: CVaR must use loss_total, not just loss_VRES. Otherwise, when
        # γ<1, the fixed cost is only in the γ-weighted term, so the effective
        # weight on F_cap becomes γ instead of 1. With nYears=1 (no scenarios),
        # changing γ would then change the objective, breaking SP/ME equivalence.
        # JuMP expressions bake in coefficient values at creation time, so we
        # must rebuild these expressions from scratch each iteration.
        loss_VRES = Dict{Int,JuMP.AffExpr}()
        loss_total = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_VRES[jy] = @expression(mod,
                sum(W[jd, jy] * (MC * g[jh, jd, jy]
                    - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                    - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD)
            )
            loss_total[jy] = @expression(mod, loss_VRES[jy] + F_cap * cap_VRES)
        end
        mod.ext[:expressions][:loss_VRES] = loss_VRES

        # ── Risk-adjusted objective ───────────────────────────────────────
        # Expected loss: F_cap·cap once + P-weighted operational loss per scenario.
        z_cap   = get(mod.ext[:parameters], :z_cap, 0.0)
        λ_cap   = get(mod.ext[:parameters], :λ_cap, 0.0)
        ρ_cap   = get(mod.ext[:parameters], :ρ_cap, 0.1)
        cap_pen = haskey(mod.ext[:parameters], :z_cap) ?
            λ_cap * (cap_VRES - z_cap) + ρ_cap/2 * (cap_VRES - z_cap)^2 : 0.0
        mod.ext[:objective] = @objective(mod, Min,
            gamma * (F_cap * cap_VRES + sum(P[jy] * loss_VRES[jy] for jy in JY))
            + (1 - gamma) * cvar_VRES
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec_GC/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + cap_pen
        )

        # ── Delete stale CVaR constraints and re-add with fresh losses ────
        # Shortfall uses loss_total (operational + F_cap·cap) so that with
        # nYears=1, CVaR = loss_total and γ has no effect (SP/ME equivalence).
        for jy in JY
            delete(mod, mod.ext[:constraints][:CVaR_VRES_shortfall][jy])
        end
        delete(mod, mod.ext[:constraints][:CVaR_VRES_link])

        # Shortfall constraints: u_VRES[jy] ≥ loss_total[jy] − α_VRES.
        mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
            u_VRES[jy] >= loss_total[jy] - alpha_VRES
        )
        # CVaR linking: CVaR_VRES ≥ α_VRES + (1/(1−β)) · Σ P[jy]·u_VRES[jy].
        one_minus_beta = max(1e-6, 1.0 - beta_conf)
        mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
            cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY)
        )

    elseif agent_type == "Conventional"
        # Conventional generator objective: same structure as VRES but
        # WITHOUT the green certificate (GC) revenue/penalty terms,
        # since conventional plants do not earn GCs.
        #   min  sum W * (stagewise convex cost - lambda_elec*g)  [cost - revenue]
        #      + rho_elec/2 * sum W * (g - g_bar_elec)^2  [ADMM elec penalty]
        g  = mod.ext[:variables][:g]
        MC = mod.ext[:parameters][:MarginalCost]
        MC_y = get(mod.ext[:parameters], :MarginalCostByYear, nothing)
        if haskey(mod.ext[:variables], :g_stage) &&
           haskey(mod.ext[:parameters], :ConvStageBaseCost) &&
           haskey(mod.ext[:parameters], :ConvStageSlope)
            g_stage = mod.ext[:variables][:g_stage]
            stage_base = mod.ext[:parameters][:ConvStageBaseCost]
            stage_slope = mod.ext[:parameters][:ConvStageSlope]
            mod.ext[:objective] = @objective(mod, Min,
                sum(W[jd, jy] * (
                    sum(stage_base[s, jy] * g_stage[s, jh, jd, jy] + 0.5 * stage_slope[s, jy] * g_stage[s, jh, jd, jy]^2 for s in 1:3)
                    - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                ) for jh in JH, jd in JD, jy in JY)
                + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
        else
        if MC_y === nothing
            MC_y = fill(MC, length(JY))
        end
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC_y[jy] * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
        end
    elseif agent_type == "Consumer"
        # Consumer objective:
        #   min  sum W * (lambda*d - U(d))                  [cost - utility]
        #      + rho/2 * sum W * (-d - g_bar_elec)^2        [ADMM penalty]
        # where U(d) = A_E*d - B_E/2*d^2 is the quadratic utility function.
        # The net market position is -d (negative because consumer is a buyer),
        # hence the penalty uses (-d - g_bar_elec)^2.
        d   = mod.ext[:variables][:d]
        A_E = mod.ext[:parameters][:A_E]
        B_E = mod.ext[:parameters][:B_E]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    end
    optimize!(mod)
    return nothing
end
