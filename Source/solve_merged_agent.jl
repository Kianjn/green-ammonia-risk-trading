# ==============================================================================
# solve_merged_agent.jl — Re-solve merged agents + ADMM step helpers
# ==============================================================================

function _merged_cap_variables(mod::Model)
    t = String(mod.ext[:parameters][:Type])
    if t == "GreenH2Coalition"
        return (mod.ext[:variables][:cap_H2_y], mod.ext[:variables][:cap_EP_y])
    elseif t == "GreenCoalition"
        caps = mod.ext[:variables][:cap_vres]
        return (caps[:solar], caps[:wind], mod.ext[:variables][:cap_H2_y], mod.ext[:variables][:cap_EP_y])
    end
    error("Unknown merged agent Type: $t")
end

"""Existing (pre-investment) capacity per merged slot, same order as `cap_slots`."""
function merged_cap_floors(mod::Model)
    p = mod.ext[:parameters]
    t = String(p[:Type])
    if t == "GreenH2Coalition"
        return Float64[get(p, :Capacity_H2_Output, 0.0), get(p, :Capacity_EP_Out, 0.0)]
    end
    v = Float64[u.Capacity for u in p[:vres_units]]
    append!(v, Float64[get(p, :Capacity_H2_Output, 0.0), get(p, :Capacity_EP_Out, 0.0)])
    return v
end

function merged_cap_snapshot(mod::Model)
    return Float64[value(v) for v in _merged_cap_variables(mod)]
end

function merged_inv_snapshot(mod::Model)
    t = String(mod.ext[:parameters][:Type])
    if t == "GreenH2Coalition"
        return Float64[value(mod.ext[:variables][:inv_cap_H2]),
                       value(mod.ext[:variables][:inv_EP])]
    end
    invs = Float64[value(mod.ext[:variables][:inv_vres][u.label])
                   for u in mod.ext[:parameters][:vres_units]]
    append!(invs, Float64[value(mod.ext[:variables][:inv_cap_H2]),
                          value(mod.ext[:variables][:inv_EP])])
    return invs
end

function _peak_value(var)
    return maximum(value(v) for v in var)
end

function _z_from_profile(g, AF::AbstractArray)
    z = 0.0
    n1, n2, n3 = size(AF)
    for jy in 1:n3, jd in 1:n2, jh in 1:n1
        af = AF[jh, jd, jy]
        af > 1e-9 || continue
        z = max(z, value(g[jh, jd, jy]) / af)
    end
    return z
end

"""Flow-implied capacity targets per slot (same construction as standalone VRES/H2/EP)."""
function merged_flow_z_target(mod::Model)
    p = mod.ext[:parameters]
    z = merged_cap_floors(mod)
    vars = mod.ext[:variables]
    t = String(p[:Type])
    if t == "GreenCoalition"
        for (i, u) in enumerate(p[:vres_units])
            z[i] = max(z[i], _z_from_profile(vars[:g_vres][u.label], u.AF))
        end
        z[end - 1] = max(z[end - 1], _peak_value(vars[:h2]))
        z[end] = max(z[end], _peak_value(vars[:ep]))
    else
        z[1] = max(z[1], _peak_value(vars[:h2]))
        z[2] = max(z[2], _peak_value(vars[:ep]))
    end
    return z
end

function _merged_λ_vec(λ_raw, n_cap::Int)
    if λ_raw isa AbstractVector && length(λ_raw) == n_cap
        return Float64.(λ_raw)
    end
    return fill(_cap_scalar(λ_raw), n_cap)
end

function update_merged_z_cap!(mod::Model, m::String, results::Dict, ADMM_state::Dict, data::Dict)
    haskey(mod.ext[:parameters], :z_cap) || return nothing
    disable_installed_capacity_split!(mod.ext[:parameters])
    return nothing
end

function solve_merged_agent!(m::String, mod::Model)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]
    p = mod.ext[:parameters]

    γ = get(p, :γ, 1.0)
    β = get(p, :β, 0.95)
    P = p[:P]
    F_h2 = electrolyzer_h2_annuity(p)
    F_ep = get(p, :FixedCost_per_MW_EP_Out, 0.0)
    op_cost = p[:OperationalCost]
    proc_cost = get(p, :ProcessingCost, 0.0)

    λ_elec = p[:λ_elec]
    g_bar_elec = p[:g_bar_elec]
    ρ_elec = p[:ρ_elec]
    λ_elec_GC = p[:λ_elec_GC]
    g_bar_elec_GC = p[:g_bar_elec_GC]
    ρ_elec_GC = p[:ρ_elec_GC]
    λ_H2_GC = p[:λ_H2_GC]
    g_bar_H2_GC = p[:g_bar_H2_GC]
    ρ_H2_GC = p[:ρ_H2_GC]
    λ_EP = p[:λ_EP]
    g_bar_EP = p[:g_bar_EP]
    ρ_EP = p[:ρ_EP]

    e_in = mod.ext[:variables][:e_in]
    h2 = mod.ext[:variables][:h2]
    q_elec_gc = mod.ext[:variables][:q_elec_gc]
    q_h2gc_ext = mod.ext[:variables][:q_h2gc_ext]
    ep = mod.ext[:variables][:ep]
    cap_H2_y = mod.ext[:variables][:cap_H2_y]
    cap_EP_y = mod.ext[:variables][:cap_EP_y]
    alpha_c = mod.ext[:variables][:alpha_coalition]
    cvar_c = mod.ext[:variables][:CVaR_coalition]
    u_c = mod.ext[:variables][:u_coalition]
    t = String(p[:Type])

    loss_op = Dict{Int, JuMP.AffExpr}()
    loss_total = Dict{Int, JuMP.AffExpr}()
    for jy in JY
        base = @expression(mod, sum(W[jd, jy] * (
            λ_elec[jh, jd, jy] * e_in[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
            + op_cost * h2[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc_ext[jh, jd, jy]
            ) for jh in JH, jd in JD))
        if t == "GreenCoalition"
            vres_loss = @expression(mod, sum(
                sum(W[jd, jy] * (
                    u.MarginalCost * mod.ext[:variables][:g_vres][u.label][jh, jd, jy]
                    - λ_elec[jh, jd, jy] * mod.ext[:variables][:g_vres][u.label][jh, jd, jy]
                    - λ_elec_GC[jh, jd, jy] * mod.ext[:variables][:g_vres][u.label][jh, jd, jy]
                ) for jh in JH, jd in JD) for u in p[:vres_units]))
            base = @expression(mod, base + vres_loss)
        end
        cap_fixed = @expression(mod, F_h2 * cap_H2_y + F_ep * cap_EP_y)
        if t == "GreenCoalition"
            cap_fixed = @expression(mod, cap_fixed + sum(
                u.FixedCost_per_MW * mod.ext[:variables][:cap_vres][u.label] for u in p[:vres_units]))
        end
        loss_op[jy] = base
        loss_total[jy] = @expression(mod, base + cap_fixed)
    end
    mod.ext[:expressions][:loss_coalition] = loss_op

    cap_vars = _merged_cap_variables(mod)
    z_cap = p[:z_cap]
    λ_cap = p[:λ_cap]
    ρ_cap = p[:ρ_cap]
    cap_pen = @expression(mod, sum(
        λ_cap[i] * (cap_vars[i] - z_cap[i]) + ρ_cap / 2 * (cap_vars[i] - z_cap[i])^2
        for i in eachindex(cap_vars)))

    cap_fixed = @expression(mod, F_h2 * cap_H2_y + F_ep * cap_EP_y)
    if t == "GreenCoalition"
        cap_fixed = @expression(mod, cap_fixed + sum(
            u.FixedCost_per_MW * mod.ext[:variables][:cap_vres][u.label] for u in p[:vres_units]))
    end

    mod.ext[:objective] = @objective(mod, Min,
        γ * (cap_fixed + sum(P[jy] * loss_op[jy] for jy in JY))
        + (1 - γ) * cvar_c
        + sum(ρ_elec / 2 * W[jd, jy] * (mod.ext[:expressions][:g_net_elec][jh, jd, jy] - g_bar_elec[jh, jd, jy])^2
              for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC / 2 * W[jd, jy] * (mod.ext[:expressions][:g_net_elec_GC][jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2
              for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC / 2 * W[jd, jy] * (q_h2gc_ext[jh, jd, jy] - g_bar_H2_GC[jh, jd, jy])^2
              for jh in JH, jd in JD, jy in JY)
        + sum(ρ_EP / 2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + cap_pen)

    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_coalition_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_coalition_link])
    mod.ext[:constraints][:CVaR_coalition_shortfall] = @constraint(mod, [jy in JY],
        u_c[jy] >= loss_total[jy] - alpha_c)
    one_minus_beta = max(1e-6, 1.0 - β)
    mod.ext[:constraints][:CVaR_coalition_link] = @constraint(mod,
        cvar_c >= alpha_c + (1 / one_minus_beta) * sum(P[jy] * u_c[jy] for jy in JY))

    optimize!(mod)
    return nothing
end

function merged_admm_step!(m::String, data::Dict, results::Dict, ADMM_state::Dict,
                             elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                             H2_GC_market::Dict, EP_market::Dict, mod::Model, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    zeros_shp = zeros(n_ts, n_rd, n_yr)

    @timeit TO "Update ADMM params" begin
        n = elec_market["nAgents"]
        prev_g = isempty(results["g"][m]) ? zeros_shp : results["g"][m][end]
        imb = isempty(ADMM_state["Imbalances"]["elec"]) ? zeros_shp : ADMM_state["Imbalances"]["elec"][end]
        mod.ext[:parameters][:g_bar_elec] = prev_g .- (1.0 / (n + 1)) .* imb
        mod.ext[:parameters][:λ_elec] = results["λ"]["elec"][end]
        mod.ext[:parameters][:ρ_elec] = ADMM_state["ρ"]["elec"][end]

        n = elec_GC_market["nAgents"]
        prev = isempty(results["elec_GC"][m]) ? zeros_shp : results["elec_GC"][m][end]
        imb = isempty(ADMM_state["Imbalances"]["elec_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["elec_GC"][end]
        mod.ext[:parameters][:g_bar_elec_GC] = prev .- (1.0 / (n + 1)) .* imb
        mod.ext[:parameters][:λ_elec_GC] = results["λ"]["elec_GC"][end]
        mod.ext[:parameters][:ρ_elec_GC] = ADMM_state["ρ"]["elec_GC"][end]

        n = H2_GC_market["nAgents"]
        prev = isempty(results["H2_GC"][m]) ? zeros_shp : results["H2_GC"][m][end]
        imb = isempty(ADMM_state["Imbalances"]["H2_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["H2_GC"][end]
        mod.ext[:parameters][:g_bar_H2_GC] = prev .- (1.0 / (n + 1)) .* imb
        mod.ext[:parameters][:λ_H2_GC] = results["λ"]["H2_GC"][end]
        mod.ext[:parameters][:ρ_H2_GC] = ADMM_state["ρ"]["H2_GC"][end]

        n = EP_market["nAgents"]
        prev = isempty(results["EP"][m]) ? zeros_shp : results["EP"][m][end]
        imb = isempty(ADMM_state["Imbalances"]["EP"]) ? zeros_shp : ADMM_state["Imbalances"]["EP"][end]
        mod.ext[:parameters][:g_bar_EP] = prev .- (1.0 / (n + 1)) .* imb
        mod.ext[:parameters][:λ_EP] = results["λ"]["EP"][end]
        mod.ext[:parameters][:ρ_EP] = ADMM_state["ρ"]["EP"][end]

        update_merged_z_cap!(mod, m, results, ADMM_state, data)
    end

    @timeit TO "Solve agent" begin
        solve_merged_agent!(m, mod)
    end

    ok = has_values(mod)
    ok || (ok = Base.invokelatest(ensure_agent_solution!, mod, m))
    if !ok
        reused = repeat_last_agent_quantities!(results, m, mod)
        reused || error("Agent $(m) has no primal and no previous ADMM iterate to reuse " *
                        "(termination=$(termination_status(mod)), primal=$(primal_status(mod))).")
        dampen_rhos_on_numerical!(ADMM_state, m)
        reset_gurobi_optimizer!(mod)
        return nothing
    end

    @timeit TO "Query results" begin
        push!(results["g"][m], collect(value.(mod.ext[:expressions][:g_net_elec])))
        push!(results["elec_GC"][m], collect(value.(mod.ext[:expressions][:g_net_elec_GC])))
        push!(results["H2_GC"][m], collect(value.(mod.ext[:expressions][:g_net_H2_GC])))
        push!(results["EP"][m], collect(value.(mod.ext[:expressions][:g_net_EP])))
        cap_snap = merged_cap_snapshot(mod)
        inv_snap = merged_inv_snapshot(mod)
        p = mod.ext[:parameters]
        haskey(results, "Cap_Merged") || (results["Cap_Merged"] = Dict{String, Vector{Vector{Float64}}}())
        haskey(results, "Inv_Merged") || (results["Inv_Merged"] = Dict{String, Vector{Vector{Float64}}}())
        haskey(results, "Merged_z_flow") || (results["Merged_z_flow"] = Dict{String, Vector{Vector{Float64}}}())
        haskey(results, "Cap_Merged_slots") || (results["Cap_Merged_slots"] = Dict{String, Vector{String}}())
        haskey(results, "Cap_Merged_floors") || (results["Cap_Merged_floors"] = Dict{String, Vector{Float64}}())
        results["Cap_Merged_slots"][m] = String.(p[:cap_slots])
        results["Cap_Merged_floors"][m] = merged_cap_floors(mod)
        push!(get!(results["Cap_Merged"], m, []), cap_snap)
        push!(get!(results["Inv_Merged"], m, []), inv_snap)
        push!(get!(results["Merged_z_flow"], m, []), merged_flow_z_target(mod))
    end
    return nothing
end

function merged_cap_warmstart!(mod::Model, sp_cap_df::DataFrame, member_ids::Vector{String})
    t = String(mod.ext[:parameters][:Type])
    floors = merged_cap_floors(mod)
    if t == "GreenH2Coalition"
        for (i, (var, invvar, mid)) in enumerate((
                (:cap_H2_y, :inv_cap_H2, member_ids[1]),
                (:cap_EP_y, :inv_EP, member_ids[2])))
            row = sp_cap_df[sp_cap_df.AgentID .== mid, :]
            val = _sp_cap_scalar(row)
            val === nothing && continue
            set_start_value(mod.ext[:variables][var], val)
            set_start_value(mod.ext[:variables][invvar], max(0.0, val - floors[i]))
        end
    elseif t == "GreenCoalition"
        id_map = Dict(:solar => member_ids[1], :wind => member_ids[2],
                      :cap_H2_y => member_ids[3], :cap_EP_y => member_ids[4])
        for (label, mid) in id_map
            row = sp_cap_df[sp_cap_df.AgentID .== mid, :]
            val = _sp_cap_scalar(row)
            val === nothing && continue
            if label in (:solar, :wind)
                set_start_value(mod.ext[:variables][:cap_vres][label], val)
                set_start_value(mod.ext[:variables][:inv_vres][label], max(0.0, val - floors[label == :solar ? 1 : 2]))
            elseif label == :cap_H2_y
                set_start_value(mod.ext[:variables][:cap_H2_y], val)
                set_start_value(mod.ext[:variables][:inv_cap_H2], max(0.0, val - floors[end - 1]))
            else
                set_start_value(mod.ext[:variables][:cap_EP_y], val)
                set_start_value(mod.ext[:variables][:inv_EP], max(0.0, val - floors[end]))
            end
        end
    end
    return nothing
end

"""Set VRES cap and inv starts together so `cap == cap0 + inv` stays feasible."""
function vres_cap_inv_warmstart!(mod::Model, cap_val::Real)
    haskey(mod.ext[:variables], :cap_VRES) || return false
    val = Float64(cap_val)
    set_start_value(mod.ext[:variables][:cap_VRES], val)
    if haskey(mod.ext[:variables], :inv_VRES)
        floor = Float64(get(mod.ext[:parameters], :Capacity, 0.0))
        set_start_value(mod.ext[:variables][:inv_VRES], max(0.0, val - floor))
    end
    return true
end
