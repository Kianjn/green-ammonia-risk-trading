# ==============================================================================
# build_EP_demand_agent.jl — Placeholder for optional elastic EP demand agent
# ==============================================================================
#
# PURPOSE:
#   EP demand is currently fixed via EP_market["D_EP"]. This builds a minimal
#   model (one variable q_ep, trivial objective) so the main script can call
#   build_EP_demand_agent! for any future EP_Demand agents without error.
#
# ==============================================================================

function build_EP_demand_agent!(m::String, mod::Model, EP_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]

    # Placeholder variable and trivial objective.  EP demand is currently
    # INELASTIC — the fixed demand quantity D_EP is handled directly in the
    # EP market-clearing constraint (via EP_market["D_EP"]), not through
    # this agent's optimization.  This build function exists so that the
    # main script can call build_EP_demand_agent! for any EP_Demand agents
    # without error, and to provide a hook for future elastic EP demand.
    q_ep = mod.ext[:variables][:q_ep] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep_demand")
    mod.ext[:objective] = @objective(mod, Min, sum(q_ep[jh, jd, jy] for jh in JH, jd in JD, jy in JY))
    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add EP demand block to shared planner model.
#
# Placeholder for future elastic EP demand with quadratic utility.
# Currently provides a demand variable bounded by D_EP_bar = LOAD_EP * D_EP_bar
# and a welfare expression U(d) = A_EP·d - (B_EP/2)·d².  No ADMM terms —
# the planner optimizes all agents jointly and EP prices emerge as duals.
# When EP demand becomes elastic, this will mirror the structure of the
# electricity and GC demand agents.
# ------------------------------------------------------------------------------

function add_EP_demand_agent_to_planner!(planner::Model, id::String, mod::Model,
                                         var_dict::Dict, W::AbstractArray)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    p = mod.ext[:parameters]
    ts = mod.ext[:timeseries]
    D_EP_bar = ts[:LOAD_EP] .* p[:D_EP_bar]
    A_EP = p[:A_EP]
    B_EP = p[:B_EP]
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    d_EP = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_EP_$(id)")
    @constraint(planner, [jh in JH, jd in JD, jy in JY], d_EP[jh, jd, jy] <= D_EP_bar[jh, jd, jy])

    # Per-year welfare = EP consumer utility U(d) = A_EP·d − (B_EP/2)·d².
    # No expenditure term: EP payments are transfers that cancel in the
    # aggregate planner objective. No per-agent CVaR: a single social CVaR
    # is applied in build_social_planner! to the aggregate social welfare.
    welfare_per_year = Dict{Int, Any}()
    for jy in JY
        welfare_per_year[jy] = @expression(planner,
            sum(W_dict[jy][jd] * (A_EP * d_EP[jh, jd, jy] - 0.5 * B_EP * d_EP[jh, jd, jy]^2)
                for jh in JH, jd in JD)
        )
    end
    var_dict[:EP_demand_d_EP][id] = d_EP
    return welfare_per_year
end
