# ==============================================================================
# build_offtaker_agent.jl — JuMP models for offtakers (green, grey, importer)
# ==============================================================================
#
# PURPOSE:
#   Builds JuMP models for offtakers (green, grey, importer) for both ADMM and
#   social planner formulations.
#
#   - GreenOfftaker: decision variables h2_in (H₂ purchases), q_h2gc (H₂ GCs),
#     ep (end-product output), yearly EP capacity cap_EP_y[jy] and investment
#     inv_EP[jy], and optional CVaR risk variables (α, β, u[jy]). Tight
#     stoichiometry ep = α_H2→EP * h2_in (no H₂ waste). Must satisfy an annual
#     GC mandate (γ_GC share of H₂ intake backed by H₂ GCs). Net positions:
#     g_net_H2 = −h2_in, g_net_H2_GC = −q_h2gc, g_net_EP = ep. ADMM objective:
#     cost (H₂, H₂ GCs, processing) − EP revenue + ADMM penalties + fixed
#     CAPEX on cap_EP_y + optional CVaR term (γ·β_G).
#
#   - GreyOfftaker: decision variables ep (EP output) and q_h2gc (H₂ GCs).
#     Does not buy physical H₂ on the market. GC mandate/compliance is imposed
#     on an inferred H₂-stream derived from EP output via gamma_NH3. Net
#     positions: g_net_H2_GC = −q_h2gc, g_net_EP = ep. Objective: production
#     cost + GC purchase − EP revenue + ADMM penalties.
#
#   - EPImporter: decision variable ep (EP imports) with a simple capacity
#     constraint and import cost. Net position: g_net_EP = ep. Objective:
#     import cost − EP revenue + ADMM penalties.
#
# ==============================================================================

function build_offtaker_agent!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)
    # ── Index sets & weights ──────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W = mod.ext[:parameters][:W]      # W[jd,jy] = representative-day weight

    # [NL] gamma_GC = 0.42 (42%) is the green certificate mandate fraction, set to the
    # EU RED III RFNBO-in-industry target (≥42% of hydrogen used in industry must be
    # renewable/RFNBO by 2030; Directive (EU) 2023/2413) [RED-III]. It specifies the
    # minimum share of H2 (intake or implied) that must be backed by green H2 certificates.
    # See docs/TECHNICAL.md (Calibration).
    gamma_GC = get(mod.ext[:parameters], :gamma_GC, 0.42)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))

    # ── ADMM parameters — H₂ market ──────────────────────────────────────
    λ_H2     = mod.ext[:parameters][:λ_H2]       # Lagrange multiplier (price)
    g_bar_H2 = mod.ext[:parameters][:g_bar_H2]   # consensus target
    ρ_H2     = mod.ext[:parameters][:ρ_H2]       # penalty weight

    # ── ADMM parameters — H₂-GC market ───────────────────────────────────
    # H₂-GC price is hourly (full 3D), like all other markets.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC  = mod.ext[:parameters][:ρ_H2_GC]

    # ── ADMM parameters — EP (energy product) market ─────────────────────
    λ_EP     = mod.ext[:parameters][:λ_EP]
    g_bar_EP = mod.ext[:parameters][:g_bar_EP]
    ρ_EP     = mod.ext[:parameters][:ρ_EP]

    # ══════════════════════════════════════════════════════════════════════
    # GreenOfftaker: buys H₂ and H₂-GCs, sells energy product (EP).
    # alpha = H₂-to-EP conversion ratio (alpha=1 means 1 MWh_H2 -> 1 MWh_EP).
    # ══════════════════════════════════════════════════════════════════════
    if agent_type == "GreenOfftaker"
        cap_h2  = mod.ext[:parameters][:Capacity_H2_In]    # max H2 intake (MW_H2)
        cap_ep_initial  = mod.ext[:parameters][:Capacity_EP_Out]   # initial max EP output (MW_EP) in first year
        alpha   = get(mod.ext[:parameters], :Alpha, 1.0)   # H2-to-EP conversion
                                      # ratio: alpha=1 means 1 MWh_H2 -> 1 MWh_EP
        proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)  # processing cost (EUR/MWh_EP)
        # Annualised fixed investment cost per MW of EP output capacity (€/MW_EP-year).
        # Read from data.yaml if present; default 0.0 keeps previous behaviour.
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW_EP_Out, 0.0)
        # Risk parameters (CVaR skeleton; γ = 0 ⇒ risk-neutral by default).
        gamma = get(mod.ext[:parameters], :γ, 1.0)
        beta_conf = get(mod.ext[:parameters], :β, 0.95)   # confidence level β
        P = mod.ext[:parameters][:P]

        # Decision variables.
        h2_in  = mod.ext[:variables][:h2_in]  = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_in")
        q_h2gc = mod.ext[:variables][:q_h2gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC")
        ep     = mod.ext[:variables][:ep]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep")

        # EP capacity and one-shot investment (same capacity in all weather scenarios).
        cap_EP_y = mod.ext[:variables][:cap_EP_y] = @variable(mod, lower_bound = 0, base_name = "cap_EP")
        inv_EP   = mod.ext[:variables][:inv_EP]   = @variable(mod, lower_bound = 0, base_name = "inv_EP")
        mod.ext[:constraints][:cap_EP_init] = @constraint(mod, cap_EP_y == cap_ep_initial + inv_EP)

        # Net positions: H2 purchased (negative), H2-GCs purchased (negative),
        # EP sold (positive).
        mod.ext[:expressions][:g_net_H2]    = @expression(mod, -h2_in)     # buyer on H2 market
        mod.ext[:expressions][:g_net_H2_GC]  = @expression(mod, -q_h2gc)   # buyer on H2-GC market
        mod.ext[:expressions][:g_net_EP]     = @expression(mod, ep)         # seller on EP market

        # ── Risk variables (agent-level CVaR) ───────────────────────────────
        # α_G: VaR proxy; CVaR_G: Conditional Value-at-Risk of loss;
        # u_G[jy]: shortfall per scenario year.
        # α and CVaR unbounded: losses (cost − revenue) can be negative.
        alpha_G = mod.ext[:variables][:alpha_GreenOfftaker] = @variable(mod, base_name = "alpha_GreenOfftaker_$(m)")
        cvar_G  = mod.ext[:variables][:CVaR_GreenOfftaker]  = @variable(mod, base_name = "CVaR_GreenOfftaker_$(m)")
        u_G     = mod.ext[:variables][:u_GreenOfftaker]     = @variable(mod, [jy in JY], lower_bound = 0, base_name = "u_GreenOfftaker_$(m)")

        # Per-year economic loss (cost − revenue) excluding ADMM penalties.
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

        # Shortfall constraints: u_G[jy] ≥ loss_total[jy] − α_G (full loss for CVaR).
        mod.ext[:constraints][:CVaR_Green_shortfall] = @constraint(mod, [jy in JY],
            u_G[jy] >= loss_total[jy] - alpha_G
        )

        # CVaR definition: CVaR_G ≥ α_G + (1/(1−β)) * Σ P[jy]*u_G[jy].
        one_minus_beta = max(1e-6, 1.0 - beta_conf)
        mod.ext[:constraints][:CVaR_Green_link] = @constraint(mod,
            cvar_G >= alpha_G + (1 / one_minus_beta) * sum(P[jy] * u_G[jy] for jy in JY)
        )

        # Tight stoichiometric link: ep == alpha * h2_in.  ALL purchased
        # H₂ must be converted to EP (no H₂ waste).  EP output is exactly
        # proportional to H₂ input via the conversion ratio alpha
        # (MWh_EP per MWh_H2).  Matches the planner formulation.
        mod.ext[:constraints][:ep_from_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] == alpha * h2_in[jh, jd, jy])

        # Annual GC mandate for green offtaker: at least gamma_GC (42%) of the
        # H₂ intake must be backed by green H₂ certificates, computed on a
        # weighted yearly basis (more realistic than hourly matching — allows
        # temporal flexibility in GC procurement within the year). Since we
        # observe H₂ purchases explicitly (h2_in), we can impose the mandate
        # directly on H₂ quantities rather than inferring them from EP output.
        mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
            sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * sum(W[jd, jy] * h2_in[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Single capacity limit based on EP output; H₂ intake is implied via stoichiometry ep = alpha * h2_in.
        mod.ext[:constraints][:cap_ep] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_EP_y)

        # GC purchase upper bound: cannot buy more green certificates than
        # the H₂ actually consumed (each certificate certifies 1 MWh_H2).
        mod.ext[:constraints][:gc_cap]    = @constraint(mod, [jh in JH, jd in JD, jy in JY], q_h2gc[jh, jd, jy] <= h2_in[jh, jd, jy])

        # Objective — min(cost - revenue + ADMM penalties):
        #   cost    = H2 purchase (lambda_H2 * h2_in) + GC purchase (lambda_H2GC * gc)
        #             + processing cost
        #   revenue = EP sales (lambda_EP * ep)
        #   penalties use net positions: -h2_in (H2), -gc (H2-GC), +ep (EP)
        n_years = length(JY)
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (
                λ_H2[jh, jd, jy]        * h2_in[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy]  * q_h2gc[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy]      * ep[jh, jd, jy]
            ) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2/2 * W[jd, jy] * ((-h2_in[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            # Fixed annualised investment cost, summed over model years (no W weighting).
            + F_cap * cap_EP_y
        )

    # ══════════════════════════════════════════════════════════════════════
    # GreyOfftaker: EP seller using conventional feedstock.  Does NOT buy
    # H₂ on the market, but a fraction of its EP output requires H₂ as
    # feedstock internally (e.g. ammonia synthesis).  Must still purchase
    # H₂-GCs to certify the green share of that internal H₂ usage.
    # ══════════════════════════════════════════════════════════════════════
    elseif agent_type == "GreyOfftaker"
        cap_ep = mod.ext[:parameters][:Capacity]        # max EP output (MW_EP)
        # Grey ammonia is SMR-based, so its marginal cost moves with the scenario
        # gas price: MC[jy] in EUR/MWh_EP (see define_offtaker_parameters.jl).
        MC     = get(mod.ext[:parameters], :MarginalCostByYear,
                     fill(mod.ext[:parameters][:MarginalCost], length(JY)))

        ep     = mod.ext[:variables][:ep]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep")
        q_h2gc = mod.ext[:variables][:q_h2gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC")

        # Net positions: H2-GCs purchased (negative), EP sold (positive).
        mod.ext[:expressions][:g_net_H2_GC] = @expression(mod, -q_h2gc)  # buyer on H2-GC market
        mod.ext[:expressions][:g_net_EP]    = @expression(mod, ep)        # seller on EP market

        # Annual GC mandate for grey offtaker.  gamma_NH3 is the EP-to-H2
        # conversion ratio (MWh_EP per MWh_H2). We do not model the grey
        # offtaker's internal H₂ stream explicitly, so we infer its H₂ intake
        # from observed EP output via H2_intake ≈ ep / gamma_NH3. The GC
        # mandate applies to this inferred H₂-equivalent portion of EP output:
        #   gc_h2 >= gamma_GC * (ep / gamma_NH3)
        # i.e., at least 42% of the implied H₂ usage must be green-certified.
        mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
            sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * (1 / mod.ext[:parameters][:gamma_NH3]) *
            sum(W[jd, jy] * ep[jh, jd, jy] for jh in JH, jd in JD)
        )
        mod.ext[:constraints][:cap_ep]     = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_ep)

        # GC purchase upper bound: cannot certify more H₂ as green than
        # the inferred H₂ feedstock used (ep / gamma_NH3).
        mod.ext[:constraints][:gc_cap]     = @constraint(mod, [jh in JH, jd in JD, jy in JY], q_h2gc[jh, jd, jy] <= (1 / mod.ext[:parameters][:gamma_NH3]) * ep[jh, jd, jy])

        # Objective — min(cost - revenue + ADMM penalties):
        #   cost    = production cost (MC * ep) + GC purchase (lambda_H2GC * gc)
        #   revenue = EP sales (lambda_EP * ep)
        #   penalties use net positions: -gc (H2-GC, purchase), +ep (EP, sale)
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC[jy] * ep[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

    # ══════════════════════════════════════════════════════════════════════
    # EPImporter: simple price-taking EP supplier via imports.  No H₂
    # purchase and no GC involvement — just imports EP at a fixed cost.
    # ══════════════════════════════════════════════════════════════════════
    else  # EPImporter
        cap   = mod.ext[:parameters][:Capacity]       # max import capacity (MW_EP)
        imp_cost = mod.ext[:parameters][:ImportCost]   # import cost (EUR/MWh_EP)

        ep = mod.ext[:variables][:ep] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep_import")
        mod.ext[:expressions][:g_net_EP] = @expression(mod, ep)  # seller on EP market (positive)
        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap)

        # Objective — min(import cost - EP revenue + ADMM penalty):
        #   Only participates in the EP market; net position = +ep (sale).
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )
    end

    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add offtaker block to shared planner model.
#
# Same physical constraints as the ADMM build (stoichiometric conversion,
# GC mandates, capacity limits) but WITHOUT any ADMM terms — no prices (lambda),
# no penalty weights (rho), no consensus targets (g_bar).  The planner optimizes
# all agents jointly; market prices emerge as duals of clearing constraints.
#
# Returns: welfare contribution = -(processing/production/import cost).
# Market revenues and expenditures cancel in the planner's aggregate.
# ------------------------------------------------------------------------------

function add_offtaker_agent_to_planner!(planner::Model, id::String, mod::Model,
                                        var_dict::Dict, W::AbstractArray)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    p = mod.ext[:parameters]
    gamma_GC = get(p, :gamma_GC, 0.42)
    agent_type = String(get(p, :Type, ""))
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    # ── GreenOfftaker (planner) ──────────────────────────────────────────
    if agent_type == "GreenOfftaker"
        EP_sell_bar_initial = p[:Capacity_EP_Out]
        alpha = get(p, :Alpha, 1.0)
        C_proc = get(p, :ProcessingCost, 0.0)
        F_cap = get(p, :FixedCost_per_MW_EP_Out, 0.0)

        h_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h_buy_$(id)")
        gc_h_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_h_buy_$(id)")
        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_sell_$(id)")

        cap_EP_y = @variable(planner, lower_bound=0, base_name="cap_EP_$(id)")
        inv_EP = @variable(planner, lower_bound=0, base_name="inv_EP_$(id)")
        @constraint(planner, cap_EP_y == EP_sell_bar_initial + inv_EP)

        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= cap_EP_y)
        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] == alpha * h_buy[jh, jd, jy])
        @constraint(planner, [jh in JH, jd in JD, jy in JY], gc_h_buy[jh, jd, jy] <= h_buy[jh, jd, jy])

        @constraint(planner, [jy in JY],
            sum(W_dict[jy][jd] * gc_h_buy[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * sum(W_dict[jy][jd] * h_buy[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Per-year welfare = −(processing cost + fixed capacity cost).
        # No per-agent CVaR: a single social CVaR is applied in
        # build_social_planner! to the aggregate social welfare.
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -(sum(W_dict[jy][jd] * (C_proc * ep_sell[jh, jd, jy]) for jh in JH, jd in JD)
                  + F_cap * cap_EP_y)
            )
        end
        var_dict[:offtaker_h_buy][id] = h_buy
        var_dict[:offtaker_gc_h_buy][id] = gc_h_buy
        var_dict[:offtaker_ep_sell][id] = ep_sell
        var_dict[:offtaker_cap_EP_green][id] = cap_EP_y
        var_dict[:offtaker_inv_EP_green][id] = inv_EP
        return welfare_per_year

    # ── GreyOfftaker (planner) ───────────────────────────────────────────
    elseif agent_type == "GreyOfftaker"
        EP_sell_bar = p[:Capacity]
        gamma_NH3 = p[:gamma_NH3]
        C_proc = get(p, :MarginalCostByYear, fill(p[:MarginalCost], length(JY)))

        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_sell_$(id)")
        gc_h_buy_G = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_h_buy_G_$(id)")

        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= EP_sell_bar)
        @constraint(planner, [jh in JH, jd in JD, jy in JY], gc_h_buy_G[jh, jd, jy] <= (1 / gamma_NH3) * ep_sell[jh, jd, jy])

        @constraint(planner, [jy in JY],
            sum(W_dict[jy][jd] * gc_h_buy_G[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * (1 / gamma_NH3) * sum(W_dict[jy][jd] * ep_sell[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Per-year welfare = −(production cost). EP revenue and GC
        # expenditures are transfers that cancel in the planner aggregate.
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -sum(W_dict[jy][jd] * (C_proc[jy] * ep_sell[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        var_dict[:offtaker_ep_sell][id] = ep_sell
        var_dict[:offtaker_gc_h_buy_G][id] = gc_h_buy_G
        return welfare_per_year

    # ── EPImporter (planner) ─────────────────────────────────────────────
    else  # EPImporter
        EP_sell_bar = p[:Capacity]
        C_proc = p[:ImportCost]

        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_import_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= EP_sell_bar)

        # Per-year welfare = −(import cost). EP revenue is a transfer
        # that cancels in the aggregate planner objective.
        welfare_per_year = Dict{Int, Any}()
        for jy in JY
            welfare_per_year[jy] = @expression(planner,
                -sum(W_dict[jy][jd] * (C_proc * ep_sell[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        var_dict[:offtaker_ep_sell_import][id] = ep_sell
        return welfare_per_year
    end
end
