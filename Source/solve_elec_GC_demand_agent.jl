# ==============================================================================
# solve_elec_GC_demand_agent.jl — Re-set objective and solve GC demand agent
# ==============================================================================
#
# PURPOSE:
#   Re-builds the objective (λ*d - utility(d) + ADMM penalty) with current
#   λ_elec_GC, g_bar_elec_GC, ρ_elec_GC and calls optimize!(mod).
#
# ==============================================================================

function solve_elec_GC_demand_agent!(m::String, mod::Model, elec_GC_market::Dict)
    # Only the objective is rebuilt each ADMM iteration; variables and
    # constraints are invariant (defined once during model construction).
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]
    A_GC = mod.ext[:parameters][:A_GC]
    B_GC = mod.ext[:parameters][:B_GC]
    λ_elec_GC    = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC   = mod.ext[:parameters][:ρ_elec_GC]
    d = mod.ext[:variables][:d_gc]
    # GC demand agent objective:
    #   min  sum W * ( lambda_GC*d - U(d) )               [cost - utility]
    #      + rho_GC/2 * sum W * (-d - g_bar_elec_GC)^2    [ADMM penalty]
    # where U(d) = A_GC*d - B_GC/2*d^2 is the quadratic utility of GC
    # consumption. The net market position is -d (buyer), so the penalty
    # uses (-d - g_bar_elec_GC)^2.
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (λ_elec_GC[jh, jd, jy] * d[jh, jd, jy] - (A_GC * d[jh, jd, jy] - B_GC/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    optimize!(mod)
    return nothing
end
