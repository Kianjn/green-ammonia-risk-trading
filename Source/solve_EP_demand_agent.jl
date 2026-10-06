# ==============================================================================
# solve_EP_demand_agent.jl — Solve optional elastic EP demand agent
# ==============================================================================
#
# PURPOSE:
#   Placeholder: if an EP_Demand agent is ever added, this would re-set its
#   objective and call optimize!. Currently EP demand is fixed via D_EP, so
#   this is not called from ADMM_subroutine for any agent.
#
# ==============================================================================

function solve_EP_demand_agent!(m::String, mod::Model, EP_market::Dict)
    # Placeholder: currently EP demand is fixed via D_EP (inelastic demand
    # subtracted directly in the imbalance computation in ADMM.jl), so no
    # EP_Demand agent exists and this function is never called from
    # ADMM_subroutine. If an elastic EP demand agent is added in the future,
    # this function would rebuild its objective (cost - utility + ADMM
    # penalty) before calling optimize!. For now it just solves the trivial
    # (empty-objective) model.
    optimize!(mod)
    return nothing
end
