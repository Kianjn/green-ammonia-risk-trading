# ==============================================================================
# build_elec_GC_demand_agent.jl — JuMP model for electricity GC demand
# ==============================================================================
#
# PURPOSE:
#   Elastic demand: variable d_gc (GC demand), bounded by peak*LOAD_GC. Objective:
#   min (λ_elec_GC*d - (A_GC*d - B_GC/2*d²)) + ADMM penalty; i.e. max utility minus
#   expenditure. Net position g_net_elec_GC = -d_gc. solve_elec_GC_demand_agent! re-sets objective and optimizes.
#
# ==============================================================================

function build_elec_GC_demand_agent!(m::String, mod::Model, elec_GC_market::Dict)
    # ── Index sets & weights ──────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W  = mod.ext[:parameters][:W]     # W[jd,jy] = representative-day weight

    # ── Demand parameters ─────────────────────────────────────────────────
    peak = mod.ext[:parameters][:PeakLoad]   # peak GC demand (MW-equivalent)
    A_GC = mod.ext[:parameters][:A_GC]       # intercept of inverse demand for GCs
    B_GC = mod.ext[:parameters][:B_GC]       # slope of inverse demand for GCs
    load = mod.ext[:timeseries][:LOAD_GC]    # normalized hourly load profile (0-1)

    # ── ADMM parameters — electricity-GC market ──────────────────────────
    λ_elec_GC    = mod.ext[:parameters][:λ_elec_GC]       # Lagrange multiplier (price)
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]   # consensus target
    ρ_elec_GC   = mod.ext[:parameters][:ρ_elec_GC]        # penalty weight

    # GC demand variable d >= 0 (number of GCs demanded per hour).
    d = mod.ext[:variables][:d_gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gc_demand")

    # Net position: demand is NEGATIVE supply in the elec-GC market.
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, -d)   # g_net_elec_GC = -d

    # Objective:
    #   min  Σ W·( λ_GC·d  −  U(d) )                  ← (1)
    #      + Σ (ρ_GC/2)·W·(−d − ḡ_GC)²               ← (2)
    #
    # U(d) = A_GC·d − (B_GC/2)·d² is the quadratic utility for GC demand
    #   (area under the inverse demand curve for green certificates).
    # (1) Agent minimizes expenditure (λ_GC·d) minus utility U(d),
    #     equivalent to maximizing consumer surplus U(d) − λ_GC·d.
    # (2) ADMM penalty on the net position (-d) toward consensus ḡ_GC.
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (λ_elec_GC[jh, jd, jy] * d[jh, jd, jy] - (A_GC * d[jh, jd, jy] - B_GC/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
    )

    # Load constraint: GC demand bounded by peak × hourly load profile,
    # representing the physical maximum GC consumption in each hour.
    mod.ext[:constraints][:load] = @constraint(mod, [jh in JH, jd in JD, jy in JY], d[jh, jd, jy] <= peak * load[jh, jd, jy])
    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add electricity GC demand block to shared planner model.
#
# Same demand structure but WITHOUT ADMM terms — no lambda (prices), no rho
# (penalty weights), no g_bar (consensus targets).  The planner optimizes all
# agents jointly; GC prices emerge as duals of the GC market-clearing constraint.
#
# Returns: welfare contribution = consumer utility U(d) = A_GC·d - (B_GC/2)·d².
# No expenditure term because market payments are transfers in the aggregate.
# ------------------------------------------------------------------------------

function add_elec_GC_demand_agent_to_planner!(planner::Model, id::String, mod::Model,
                                              var_dict::Dict, W::AbstractArray)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    p = mod.ext[:parameters]
    ts = mod.ext[:timeseries]
    D_GC_E_bar = p[:PeakLoad] .* ts[:LOAD_GC]
    A_GC = p[:A_GC]
    B_GC = p[:B_GC]
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    d_GC_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_GC_E_$(id)")
    @constraint(planner, [jh in JH, jd in JD, jy in JY], d_GC_E[jh, jd, jy] <= D_GC_E_bar[jh, jd, jy])

    # Per-year welfare = GC consumer utility U(d) = A_GC·d − (B_GC/2)·d².
    # No expenditure term: GC payments are transfers that cancel in the
    # aggregate planner objective. No per-agent CVaR: a single social CVaR
    # is applied in build_social_planner! to the aggregate social welfare.
    welfare_per_year = Dict{Int, Any}()
    for jy in JY
        welfare_per_year[jy] = @expression(planner,
            sum(W_dict[jy][jd] * (A_GC * d_GC_E[jh, jd, jy] - 0.5 * B_GC * d_GC_E[jh, jd, jy]^2)
                for jh in JH, jd in JD)
        )
    end
    var_dict[:elec_GC_demand_d_GC_E][id] = d_GC_E
    return welfare_per_year
end
