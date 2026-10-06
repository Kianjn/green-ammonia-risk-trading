# ==============================================================================
# build_H2_agent.jl — JuMP model for electrolytic H₂ producer
# ==============================================================================
#
# PURPOSE:
#   One agent type: electrolyzer. ADMM build creates variables for electricity
#   intake e_in, hydrogen output h2_out, electricity GCs q_elec_gc, hydrogen
#   GCs q_h2gc, and YEARLY H₂ output capacity cap_H2_y with investment inv_cap_H2.
#   Constraints: h2_out = η * e_in, q_h2gc <= h2_out, and annual green-backing
#   so that only the share of H₂ that can be backed by certified electricity is
#   green. Net positions: elec -e_in, elec_GC -q_elec_gc, H2 +h2_out, H2_GC
#   +q_h2gc. The objective is cost (elec, elec_GC, op) − revenue (H2, H2_GC)
#   plus ADMM penalties, fixed annualised CAPEX on cap_H2_y, and an optional
#   CVaR-based risk term (γ·β_H2). solve_H2_agent! re-sets the objective with
#   current λ/ρ/g_bar and calls optimize!.
#
# ==============================================================================

function build_H2_agent!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict)
    # ── Index sets & weights ──────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W  = mod.ext[:parameters][:W]     # W[jd,jy] = representative-day weight

    # ── Electrolyzer technology parameters ────────────────────────────────
    η  = mod.ext[:parameters][:η_elec_H2]
    # Single effective capacity: H₂ output nameplate in first model year (MW_H2).
    cap_H2_initial = mod.ext[:parameters][:Capacity_H2_Output]
    op_cost  = mod.ext[:parameters][:OperationalCost]        # operating cost (EUR/MWh_H2)
    # Annualised fixed investment cost per MW of electrolyser capacity (€/MW_elec-year).
    # Default 0.0 preserves original behaviour if not provided.
    F_cap = electrolyzer_h2_annuity(mod.ext[:parameters])
    # Risk parameters (CVaR skeleton; γ = 0 ⇒ risk-neutral by default).
    gamma = get(mod.ext[:parameters], :γ, 1.0)
    beta_conf = get(mod.ext[:parameters], :β, 0.95)   # confidence level β
    P = mod.ext[:parameters][:P]

    # ── ADMM parameters — electricity market ──────────────────────────────
    λ_elec     = mod.ext[:parameters][:λ_elec]       # Lagrange multiplier (price)
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]   # consensus target
    ρ_elec     = mod.ext[:parameters][:ρ_elec]       # penalty weight

    # ── ADMM parameters — electricity-GC market ──────────────────────────
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]

    # ── ADMM parameters — H₂ market ──────────────────────────────────────
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2  = mod.ext[:parameters][:g_bar_H2]
    ρ_H2      = mod.ext[:parameters][:ρ_H2]

    # ── ADMM parameters — H₂-GC market ───────────────────────────────────
    # H₂-GC price is hourly (full 3D), like all other markets. Agents see
    # while ḡ and the ADMM penalties remain defined on the full 3D grid.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]

    # ── Decision variables ───────────────────────────────────────────────
    # e_in      = electricity purchased from the electricity market (MWh)
    # h2_out    = H₂ produced and sold on the H₂ market (MWh_H2)
    # q_elec_gc = electricity GCs purchased to certify green electricity use (MWh)
    # q_h2gc    = H₂ GCs issued and sold on the H₂-GC market (MWh_H2)
    e_in      = mod.ext[:variables][:e_in]      = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "elec_in")
    h2_out    = mod.ext[:variables][:h2_out]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_out")
    q_elec_gc = mod.ext[:variables][:q_elec_gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "elec_GC")
    q_h2gc    = mod.ext[:variables][:q_h2gc]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC_prod")

    # Single capacity and one-shot investment (same H₂ capacity in all weather scenarios).
    cap_H2_y   = mod.ext[:variables][:cap_H2_y]   = @variable(mod, lower_bound = 0, base_name = "cap_H2")
    inv_cap_H2 = mod.ext[:variables][:inv_cap_H2] = @variable(mod, lower_bound = 0, base_name = "inv_cap_H2")
    mod.ext[:constraints][:cap_H2_init] = @constraint(mod, cap_H2_y == cap_H2_initial + inv_cap_H2)

    # ── Net market positions ──────────────────────────────────────────────
    # Sign convention: NEGATIVE for purchases, POSITIVE for sales.
    # elec    = −e_in      (buyer on the electricity market)
    # elec_GC = −q_elec_gc (buyer on the electricity-GC market)
    # H2      = +h2_out    (seller on the H₂ market)
    # H2_GC   = +q_h2gc    (seller on the H₂-GC market)
    mod.ext[:expressions][:g_net_elec]     = @expression(mod, -e_in)
    mod.ext[:expressions][:g_net_elec_GC]  = @expression(mod, -q_elec_gc)
    mod.ext[:expressions][:g_net_H2]       = @expression(mod, h2_out)
    mod.ext[:expressions][:g_net_H2_GC]    = @expression(mod, q_h2gc)

    # ── Physical / capacity constraints ──────────────────────────────────

    # Conversion constraint (stoichiometry): H₂ output is proportional to
    # electricity input with efficiency η.  This is the fundamental mass/
    # energy balance of the electrolyzer.
    mod.ext[:constraints][:h2_from_elec]   = @constraint(mod, [jh in JH, jd in JD, jy in JY], h2_out[jh, jd, jy] == η * e_in[jh, jd, jy])

    # GC physical limit: cannot issue more H₂ green certificates than the
    # physical H₂ actually produced in that hour (no phantom certificates).
    mod.ext[:constraints][:gc_phys_limit]  = @constraint(mod, [jh in JH, jd in JD, jy in JY], q_h2gc[jh, jd, jy] <= h2_out[jh, jd, jy])

    # Single equipment capacity limit on H₂ output (implicitly bounds electricity input via stoichiometry).
    mod.ext[:constraints][:cap_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY], h2_out[jh, jd, jy] <= cap_H2_y)

    # Annual green-backing constraint: over each year (weighted by
    # representative-day weights W), the electricity GCs purchased must be
    # enough to back all H₂ GCs issued.  The (1/η) factor arises because
    # producing 1 MWh_H2 of certified green H₂ requires (1/η) MWh of
    # certified green electricity (inverse of conversion efficiency).
    # This is ANNUAL rather than hourly to allow temporal flexibility in
    # GC procurement — the electrolyzer may buy electricity GCs in hours
    # different from when it actually produces green H₂.
    mod.ext[:constraints][:gc_backing_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_elec_gc[jh, jd, jy] for jh in JH, jd in JD) >=
        (1 / η) * sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD)
    )

    # ── Risk variables (agent-level CVaR) ───────────────────────────────────
    # α_H2: VaR proxy; CVaR_H2: Conditional Value-at-Risk of loss;
    # u_H2[jy]: shortfall per scenario year.
    # α and CVaR unbounded: losses (cost − revenue) can be negative.
    alpha_H2 = mod.ext[:variables][:alpha_H2] = @variable(mod, base_name = "alpha_H2_$(m)")
    cvar_H2  = mod.ext[:variables][:CVaR_H2]  = @variable(mod, base_name = "CVaR_H2_$(m)")
    u_H2     = mod.ext[:variables][:u_H2]     = @variable(mod, [jy in JY], lower_bound = 0, base_name = "u_H2_$(m)")

    # Per-year economic loss (cost − revenue) excluding ADMM penalties.
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

    # Shortfall constraints: u_H2[jy] ≥ loss_total[jy] − α_H2 (full loss for CVaR).
    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_total[jy] - alpha_H2
    )

    # CVaR definition: CVaR_H2 ≥ α_H2 + (1/(1−β)) * Σ P[jy]*u_H2[jy].
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY)
    )

    # ── Objective ──────────────────────────────────────────────────────────
    # min  Σ W·( λ_elec·e_in            ← electricity purchase cost
    #          + λ_GC·gc_e              ← elec-GC purchase cost
    #          + op_cost·h2             ← operational (non-fuel) cost
    #          − λ_H2·h2               ← H₂ sales revenue
    #          − λ_H2GC·gc_h2 )        ← H₂-GC sales revenue
    #    + (ρ_elec/2)  ·Σ W·(−e_in     − ḡ_elec)²     ← ADMM penalty (elec)
    #    + (ρ_GC/2)    ·Σ W·(−gc_e     − ḡ_GC)²       ← ADMM penalty (elec-GC)
    #    + (ρ_H2/2)    ·Σ W·(+h2       − ḡ_H2)²       ← ADMM penalty (H₂)
    #    + (ρ_H2GC/2)  ·Σ W·(+gc_h2    − ḡ_H2GC)²     ← ADMM penalty (H₂-GC)
    #
    # The agent minimizes: total procurement cost + operational cost
    #                     − total revenue from H₂ and H₂-GC sales
    #                     + ADMM augmented-Lagrangian penalties that push each
    #                       net position toward its market consensus target.
    # Penalty net positions use the sign convention: −e_in, −gc_e (purchases),
    # +h2, +gc_h2 (sales) minus the respective consensus targets ḡ.
    n_years = length(JY)

    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_elec[jh, jd, jy]       * e_in[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
            + op_cost * h2_out[jh, jd, jy]
            - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
            - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy]         - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + F_cap * cap_H2_y
    )
    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add H₂ producer block to shared planner model.
#
# Same physical constraints as the ADMM build (conversion, GC limits, capacity,
# annual green-backing) but WITHOUT any ADMM terms — no λ (prices), no ρ
# (penalty weights), no ḡ (consensus targets).  The planner optimizes all
# agents jointly; prices emerge as duals of market-clearing constraints.
#
# Returns: welfare contribution = −(operational cost).  Electricity purchase
# cost and GC costs are transfers that cancel out in the planner's aggregate.
# ------------------------------------------------------------------------------

function add_H2_agent_to_planner!(planner::Model, id::String, mod::Model,
                                  var_dict::Dict, W::AbstractArray)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    p = mod.ext[:parameters]
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    H_bar = p[:Capacity_H2_Output]
    η = 1.0 / p[:SpecificConsumption]
    C_H = p[:OperationalCost]
    F_cap = electrolyzer_h2_annuity(p)

    e_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="e_buy_$(id)")
    gc_e_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_e_buy_$(id)")
    h_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h_sell_$(id)")
    gc_h_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_h_sell_$(id)")

    cap_H2_y = @variable(planner, lower_bound=0, base_name="cap_H2_$(id)")
    inv_cap_H2 = @variable(planner, lower_bound=0, base_name="inv_cap_H2_$(id)")
    @constraint(planner, cap_H2_y == H_bar + inv_cap_H2)

    @constraint(planner, [jh in JH, jd in JD, jy in JY], h_sell[jh, jd, jy] <= cap_H2_y)
    @constraint(planner, [jh in JH, jd in JD, jy in JY], h_sell[jh, jd, jy] == η * e_buy[jh, jd, jy])
    @constraint(planner, [jh in JH, jd in JD, jy in JY], gc_h_sell[jh, jd, jy] <= h_sell[jh, jd, jy])

    @constraint(planner, [jy in JY],
        sum(W_dict[jy][jd] * gc_e_buy[jh, jd, jy] for jh in JH, jd in JD) >=
        (1/η) * sum(W_dict[jy][jd] * gc_h_sell[jh, jd, jy] for jh in JH, jd in JD)
    )

    # Per-year welfare = −(operational cost + fixed capacity cost).
    # No per-agent CVaR: a single social CVaR is applied in
    # build_social_planner! to the aggregate social welfare.
    welfare_per_year = Dict{Int, Any}()
    for jy in JY
        welfare_per_year[jy] = @expression(planner,
            -(sum(W_dict[jy][jd] * (C_H * h_sell[jh, jd, jy]) for jh in JH, jd in JD)
              + F_cap * cap_H2_y)
        )
    end

    var_dict[:H2_e_buy][id] = e_buy
    var_dict[:H2_gc_e_buy][id] = gc_e_buy
    var_dict[:H2_h_sell][id] = h_sell
    var_dict[:H2_gc_h_sell][id] = gc_h_sell
    var_dict[:H2_cap_elec][id] = cap_H2_y
    var_dict[:H2_inv_elec][id] = inv_cap_H2
    return welfare_per_year
end
