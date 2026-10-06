# ==============================================================================
# build_merged_agent.jl — JuMP models for merged partial-planner agents
# ==============================================================================

function _build_green_h2_chain!(m::String, mod::Model)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]
    p = mod.ext[:parameters]

    η = p[:η_elec_H2]
    cap_h2_0 = p[:Capacity_H2_Output]
    cap_ep_0 = p[:Capacity_EP_Out]
    op_cost = p[:OperationalCost]
    proc_cost = get(p, :ProcessingCost, 0.0)
    alpha = get(p, :Alpha, 1.0)
    gamma_gc = get(p, :gamma_GC, 0.42)
    F_h2 = electrolyzer_h2_annuity(p)
    F_ep = get(p, :FixedCost_per_MW_EP_Out, 0.0)
    beta_conf = get(p, :β, 0.95)
    P = p[:P]

    λ_elec = p[:λ_elec]
    g_bar_elec = p[:g_bar_elec]
    ρ_elec = p[:ρ_elec]
    λ_elec_GC = p[:λ_elec_GC]
    g_bar_elec_GC = p[:g_bar_elec_GC]
    ρ_elec_GC = p[:ρ_elec_GC]
    λ_EP = p[:λ_EP]
    g_bar_EP = p[:g_bar_EP]
    ρ_EP = p[:ρ_EP]

    e_in = mod.ext[:variables][:e_in] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_in")
    h2 = mod.ext[:variables][:h2] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_internal")
    q_elec_gc = mod.ext[:variables][:q_elec_gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_GC")
    q_h2gc_int = mod.ext[:variables][:q_h2gc_int] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_GC_internal")
    q_h2gc_ext = mod.ext[:variables][:q_h2gc_ext] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_GC_market")
    ep = mod.ext[:variables][:ep] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep")

    cap_H2_y = mod.ext[:variables][:cap_H2_y] = @variable(mod, lower_bound=0, base_name="cap_H2")
    inv_cap_H2 = mod.ext[:variables][:inv_cap_H2] = @variable(mod, lower_bound=0, base_name="inv_cap_H2")
    cap_EP_y = mod.ext[:variables][:cap_EP_y] = @variable(mod, lower_bound=0, base_name="cap_EP")
    inv_EP = mod.ext[:variables][:inv_EP] = @variable(mod, lower_bound=0, base_name="inv_EP")

    mod.ext[:constraints][:cap_H2_init] = @constraint(mod, cap_H2_y == cap_h2_0 + inv_cap_H2)
    mod.ext[:constraints][:cap_EP_init] = @constraint(mod, cap_EP_y == cap_ep_0 + inv_EP)
    mod.ext[:constraints][:h2_from_elec] = @constraint(mod, [jh in JH, jd in JD, jy in JY], h2[jh, jd, jy] == η * e_in[jh, jd, jy])
    mod.ext[:constraints][:ep_from_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] == alpha * h2[jh, jd, jy])
    mod.ext[:constraints][:gc_phys_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        q_h2gc_int[jh, jd, jy] + q_h2gc_ext[jh, jd, jy] <= h2[jh, jd, jy])
    mod.ext[:constraints][:cap_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY], h2[jh, jd, jy] <= cap_H2_y)
    mod.ext[:constraints][:cap_ep] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_EP_y)
    mod.ext[:constraints][:gc_backing_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_elec_gc[jh, jd, jy] for jh in JH, jd in JD) >=
        (1 / η) * sum(W[jd, jy] * (q_h2gc_int[jh, jd, jy] + q_h2gc_ext[jh, jd, jy]) for jh in JH, jd in JD))
    mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_h2gc_int[jh, jd, jy] for jh in JH, jd in JD) >=
        gamma_gc * sum(W[jd, jy] * h2[jh, jd, jy] for jh in JH, jd in JD))

    mod.ext[:expressions][:g_net_elec] = @expression(mod, -e_in)
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, -q_elec_gc)
    mod.ext[:expressions][:g_net_H2_GC] = @expression(mod, q_h2gc_ext)
    mod.ext[:expressions][:g_net_EP] = @expression(mod, ep)

    λ_H2_GC = p[:λ_H2_GC]
    g_bar_H2_GC = p[:g_bar_H2_GC]
    ρ_H2_GC = p[:ρ_H2_GC]

    alpha_c = mod.ext[:variables][:alpha_coalition] = @variable(mod, base_name="alpha_coalition_$(m)")
    cvar_c = mod.ext[:variables][:CVaR_coalition] = @variable(mod, base_name="CVaR_coalition_$(m)")
    u_c = mod.ext[:variables][:u_coalition] = @variable(mod, [jy in JY], lower_bound=0, base_name="u_coalition_$(m)")

    loss_op = Dict{Int, JuMP.AffExpr}()
    loss_total = Dict{Int, JuMP.AffExpr}()
    for jy in JY
        loss_op[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy] * e_in[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + op_cost * h2[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc_ext[jh, jd, jy]
            ) for jh in JH, jd in JD))
        loss_total[jy] = @expression(mod, loss_op[jy] + F_h2 * cap_H2_y + F_ep * cap_EP_y)
    end
    mod.ext[:expressions][:loss_coalition] = loss_op
    mod.ext[:constraints][:CVaR_coalition_shortfall] = @constraint(mod, [jy in JY],
        u_c[jy] >= loss_total[jy] - alpha_c)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_coalition_link] = @constraint(mod,
        cvar_c >= alpha_c + (1 / one_minus_beta) * sum(P[jy] * u_c[jy] for jy in JY))

    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_elec[jh, jd, jy] * e_in[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
            + op_cost * h2[jh, jd, jy]
            + proc_cost * ep[jh, jd, jy]
            - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
            - λ_H2_GC[jh, jd, jy] * q_h2gc_ext[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec / 2 * W[jd, jy] * ((-e_in[jh, jd, jy]) - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC / 2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC / 2 * W[jd, jy] * (q_h2gc_ext[jh, jd, jy] - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_EP / 2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + F_h2 * cap_H2_y + F_ep * cap_EP_y)
    return mod
end

function _build_green_coalition!(m::String, mod::Model)
    _build_green_h2_chain!(m, mod)
    p = mod.ext[:parameters]
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]

    e_in = mod.ext[:variables][:e_in]
    q_elec_gc = mod.ext[:variables][:q_elec_gc]
    g_vars = mod.ext[:variables][:g_vres] = Dict{Symbol, Any}()
    cap_vars = mod.ext[:variables][:cap_vres] = Dict{Symbol, Any}()
    inv_vars = mod.ext[:variables][:inv_vres] = Dict{Symbol, Any}()
    g_exprs = Any[]
    for u in p[:vres_units]
        g = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_$(u.label)")
        cap = @variable(mod, lower_bound=0, base_name="cap_$(u.label)")
        inv = @variable(mod, lower_bound=0, base_name="inv_$(u.label)")
        @constraint(mod, cap == u.Capacity + inv)
        @constraint(mod, [jh in JH, jd in JD, jy in JY], g[jh, jd, jy] <= u.AF[jh, jd, jy] * cap)
        g_vars[u.label] = g
        cap_vars[u.label] = cap
        inv_vars[u.label] = inv
        push!(g_exprs, g)
    end
    g_total = @expression(mod, sum(g_exprs[i] for i in eachindex(g_exprs)))
    mod.ext[:expressions][:g_total_vres] = g_total
    mod.ext[:expressions][:g_net_elec] = @expression(mod, g_total - e_in)
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g_total - q_elec_gc)
    return mod
end

function build_merged_agent!(m::String, mod::Model)
    t = String(get(mod.ext[:parameters], :Type, ""))
    if t == "GreenH2Coalition"
        return _build_green_h2_chain!(m, mod)
    elseif t == "GreenCoalition"
        return _build_green_coalition!(m, mod)
    end
    error("Unknown merged agent Type: $t")
end
