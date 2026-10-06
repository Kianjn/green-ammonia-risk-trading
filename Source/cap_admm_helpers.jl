# ==============================================================================
# cap_admm_helpers.jl — Shared helpers for scalar capacity ADMM (x = z split)
# ==============================================================================
#
# Capacity consensus stores z-history as Vector{Vector{Float64}} where each
# snapshot is a length-1 (or legacy length-nYears) vector. Always push via
# _cap_z_push! so a bare Float64 is never appended to z history.

"""True iff this ADMM run is risk-neutral (γ = 1).

RA ADMM must not inherit the matching-β social-planner dispatch: that hedge
builds extra green, which is the opposite of the private weather-tail response.
"""
function admm_is_risk_neutral(data)::Bool
    admm = get(data, "ADMM", data)
    g = Float64(get(admm, "gamma", get(data, "gamma", 1.0)))
    return abs(g - 1.0) <= 1e-12
end

"""Drop the installed-capacity ADMM split (Alessio).

`cap` stays a private QP variable with `g ≤ AF·cap`. ADMM coordinates energy
only. Zeroing λ_cap and ρ_cap makes the `(x − z)` penalty a no-op even if the
objective still contains the term.
"""
function disable_installed_capacity_split!(p::Dict)
    haskey(p, :z_cap) || return nothing
    z = p[:z_cap]
    if z isa AbstractVector
        p[:λ_cap] = zeros(length(z))
    else
        p[:λ_cap] = 0.0
    end
    p[:ρ_cap] = 0.0
    return nothing
end

"""Extract scalar capacity / dual value from scalar or legacy vector storage."""
_cap_scalar(x) = x isa Real ? Float64(x) : (isempty(x) ? 0.0 : Float64(x[1]))

"""Normalize a capacity target to Vector{Float64} for z-history storage."""
function _cap_z_vec(val)
    val isa Real && return [Float64(val)]
    return Float64.(collect(val))
end

"""Append one capacity-target snapshot to z history."""
function _cap_z_push!(hist::Vector{Vector{Float64}}, val)
    push!(hist, _cap_z_vec(val))
    return hist
end

"""Apply numerically robust Gurobi defaults for ADMM agent QPs."""
function configure_gurobi_agent!(mod::Model)
    set_silent(mod)
    set_optimizer_attribute(mod, "NumericFocus", 1)
    set_optimizer_attribute(mod, "BarHomogeneous", 1)
    set_optimizer_attribute(mod, "ScaleFlag", 2)
    return nothing
end

"""Rebuild a clean Gurobi backend (clears stale deleted CVaR rows / KKT ill-conditioning)."""
function reset_gurobi_optimizer!(mod::Model)
    if isdefined(Main, :GUROBI_ENV)
        set_optimizer(mod, () -> Gurobi.Optimizer(Main.GUROBI_ENV))
    else
        set_optimizer(mod, Gurobi.Optimizer)
    end
    configure_gurobi_agent!(mod)
    return nothing
end

"""Copy the last successful primal into MIP/QP starts for the next solve.

Query every value first, then set starts. JuMP invalidates the primal if the
model is modified (including `set_start_value`) while values are still being
read.
"""
function snapshot_primal_starts!(mod::Model)
    has_values(mod) || return nothing
    starts = Vector{Tuple{VariableRef, Float64}}()
    for v in all_variables(mod)
        val = value(v)
        isfinite(val) && push!(starts, (v, val))
    end
    for (v, val) in starts
        set_start_value(v, val)
    end
    return nothing
end

function _gurobi_optimize_with!(mod::Model, attrs::Pair...)
    for (k, v) in attrs
        set_optimizer_attribute(mod, k, v)
    end
    optimize!(mod)
    return has_values(mod)
end

"""Retry a failed agent QP. Returns `true` if a primal exists; never throws."""
function ensure_agent_solution!(mod::Model, m::String)
    has_values(mod) && return true
    st = termination_status(mod)

    if st == MOI.INFEASIBLE_OR_UNBOUNDED || st == MOI.DUAL_INFEASIBLE
        _gurobi_optimize_with!(mod, "DualReductions" => 0) && return true
    end

    # A new Gurobi model is the most reliable recovery after hundreds of
    # objective rebuilds / CVaR delete-and-readd cycles.
    reset_gurobi_optimizer!(mod)
    snapshot_primal_starts!(mod)
    _gurobi_optimize_with!(mod, "NumericFocus" => 3, "BarHomogeneous" => 1,
            "ScaleFlag" => 2, "DualReductions" => 0) && return true
    _gurobi_optimize_with!(mod, "Method" => 2, "Crossover" => 0,
            "BarConvTol" => 1e-4, "FeasibilityTol" => 1e-5, "Presolve" => 0) && return true
    @warn "Agent $(m) has no primal after Gurobi retries " *
          "(termination=$(termination_status(mod)), primal=$(primal_status(mod))). " *
          "Reusing the previous ADMM iterate."
    return false
end

"""Duplicate the last stored quantities for `m` so ADMM can continue without `value()`."""
function repeat_last_agent_quantities!(results::Dict, m::String, mod::Model)
    function take!(hist)
        hist isa AbstractVector || return false
        isempty(hist) && return false
        v = hist[end]
        push!(hist, v isa Number ? v : (v isa AbstractArray ? copy(v) : deepcopy(v)))
        return true
    end
    ok = true
    p = mod.ext[:parameters]
    get(p, :in_elec_market, false) && (ok &= take!(get(results["g"], m, nothing)))
    get(p, :in_H2_market, false) && (ok &= take!(get(results["h2"], m, nothing)))
    get(p, :in_elec_GC_market, false) && (ok &= take!(get(results["elec_GC"], m, nothing)))
    get(p, :in_H2_GC_market, false) && (ok &= take!(get(results["H2_GC"], m, nothing)))
    get(p, :in_EP_market, false) && (ok &= take!(get(results["EP"], m, nothing)))
    atype = String(get(p, :Type, ""))
    # Bilateral contract volumes (me_contracts.jl): scalar MW per link.
    if get(p, :in_ppa_market, false)
        if atype == "VRES" && haskey(results, "C_ppa_sell")
            ok &= take!(get(results["C_ppa_sell"], m, nothing))
        elseif haskey(results, "C_ppa_buy") && haskey(results["C_ppa_buy"], m)
            for vres_id in keys(results["C_ppa_buy"][m])
                ok &= take!(results["C_ppa_buy"][m][vres_id])
            end
        end
    end
    if get(p, :in_hpa_market, false)
        if atype == "GreenProducer" && haskey(results, "C_hpa_sell")
            ok &= take!(get(results["C_hpa_sell"], m, nothing))
        elseif atype == "GreenOfftaker" && haskey(results, "C_hpa_buy") && haskey(results["C_hpa_buy"], m)
            for h2_id in keys(results["C_hpa_buy"][m])
                ok &= take!(results["C_hpa_buy"][m][h2_id])
            end
        end
    end
    if atype == "VRES" && haskey(results, "Cap_VRES") && haskey(results["Cap_VRES"], m) &&
            !isempty(results["Cap_VRES"][m])
        ok &= take!(results["Cap_VRES"][m])
        ok &= take!(results["Inv_VRES"][m])
    end
    if haskey(mod.ext[:variables], :cap_H2_y) && haskey(results, "Cap_Elec_H2") &&
            haskey(results["Cap_Elec_H2"], m) && !isempty(results["Cap_Elec_H2"][m])
        ok &= take!(results["Cap_Elec_H2"][m])
        ok &= take!(results["Inv_Elec_H2"][m])
    end
    if atype == "GreenOfftaker" && haskey(results, "Cap_EP_Green") &&
            haskey(results["Cap_EP_Green"], m) && !isempty(results["Cap_EP_Green"][m])
        ok &= take!(results["Cap_EP_Green"][m])
        ok &= take!(results["Inv_EP_Green"][m])
    end
    if haskey(results, "Cap_Merged") && haskey(results["Cap_Merged"], m) &&
            !isempty(results["Cap_Merged"][m])
        ok &= take!(results["Cap_Merged"][m])
        haskey(results, "Inv_Merged") && haskey(results["Inv_Merged"], m) &&
            !isempty(results["Inv_Merged"][m]) && (ok &= take!(results["Inv_Merged"][m]))
        haskey(results, "Merged_z_flow") && haskey(results["Merged_z_flow"], m) &&
            !isempty(results["Merged_z_flow"][m]) && (ok &= take!(results["Merged_z_flow"][m]))
    end
    return ok
end

"""Halve current ρ after a numerical failure so the next QP is better scaled."""
function dampen_rhos_on_numerical!(ADMM_state::Dict, m::String)
    fac = 0.5
    ρmin = 1e-4
    if haskey(ADMM_state, "ρ")
        for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
            haskey(ADMM_state["ρ"], key) || continue
            isempty(ADMM_state["ρ"][key]) && continue
            ADMM_state["ρ"][key][end] = max(ρmin, fac * ADMM_state["ρ"][key][end])
        end
    end
    cap = get(ADMM_state, "Capacity", nothing)
    if cap !== nothing && haskey(cap, "ρ") && haskey(cap["ρ"], m) && !isempty(cap["ρ"][m])
        cap["ρ"][m][end] = max(0.10, fac * cap["ρ"][m][end])
    end
    # Bilateral contract links (me_contracts.jl): one ρ per PPA / HPA link.
    for ck in ("ppa", "hpa")
        haskey(ADMM_state, ck) || continue
        C = ADMM_state[ck]
        haskey(C, "ρ") || continue
        for id in keys(C["ρ"])
            isempty(C["ρ"][id]) || (C["ρ"][id][end] = max(ρmin, fac * C["ρ"][id][end]))
        end
    end
    return nothing
end

"""Keep capacity duals from dominating the agent QP (units: €/MW)."""
_clip_λ_cap(λ; bound::Float64 = 1.0e6) = clamp(_cap_scalar(λ), -bound, bound)

"""Read installed capacity vector from ADMM results (coalition or legacy single-cap)."""
function _agent_cap_vec_from_results(m::String, results::Dict)
    if haskey(results, "Cap_Merged") && !isempty(get(results["Cap_Merged"], m, []))
        return copy(results["Cap_Merged"][m][end])
    end
    if haskey(results, "Cap_VRES") && !isempty(get(results["Cap_VRES"], m, []))
        return copy(results["Cap_VRES"][m][end])
    end
    if haskey(results, "Cap_Elec_H2") && !isempty(get(results["Cap_Elec_H2"], m, []))
        return copy(results["Cap_Elec_H2"][m][end])
    end
    if haskey(results, "Cap_EP_Green") && !isempty(get(results["Cap_EP_Green"], m, []))
        return copy(results["Cap_EP_Green"][m][end])
    end
    return Float64[]
end

"""Pad or truncate a capacity/dual vector to length `n` without broadcasting a scalar."""
function _aligned_cap_target(z_vec, n::Int)
    n <= 0 && return Float64[]
    v = z_vec isa AbstractVector ? Float64.(collect(z_vec)) : [_cap_scalar(z_vec)]
    length(v) == n && return v
    length(v) > n && return v[1:n]
    return vcat(v, zeros(n - length(v)))
end

"""Primal capacity residual (L2 for multi-cap coalitions)."""
function _cap_primal_residual(cap_vec::AbstractVector{<:Real}, z_vec::AbstractVector{<:Real})
    isempty(cap_vec) && return 0.0
    z = _aligned_cap_target(z_vec, length(cap_vec))
    d = Float64.(cap_vec) .- z
    return length(d) == 1 ? abs(d[1]) : sqrt(sum(abs2, d))
end

"""Dual ascent update for scalar or vector capacity split."""
function _cap_dual_ascent(λ_prev, ρ_m::Real, cap_vec::AbstractVector{<:Real}, z_vec::AbstractVector{<:Real})
    isempty(cap_vec) && return _cap_z_vec(λ_prev)
    n = length(cap_vec)
    λ0 = _aligned_cap_target(λ_prev, n)
    z  = _aligned_cap_target(z_vec, n)
    return [_clip_λ_cap(λ0[i] + ρ_m * (Float64(cap_vec[i]) - z[i])) for i in 1:n]
end

"""Boyd abs+rel tolerance for a scalar capacity split (`x = z`).

Uses `EpsilonCap` (MW), not the flow-market `EpsilonAbs`. Capacity is one
decision, so there is no `sqrt(n_slots)` factor. `relax` is an optional
multiplier (1.0 by default; unused by the current entry points).
"""
function _cap_boyd_eps(ADMM_state::Dict, m::String; relax::Float64 = 1.0)
    eps_cap = Float64(get(ADMM_state, "EpsilonCap", get(ADMM_state, "EpsilonAbs", 5.0)))
    eps_rel = Float64(get(ADMM_state, "EpsilonRel", 0.0))
    cap_state = ADMM_state["Capacity"]
    sp_m = max(get(cap_state["ResidualScale_Primal"], m, 1.0), 1.0)
    sd_m = max(get(cap_state["ResidualScale_Dual"], m, 1.0), 1.0)
    return relax * (eps_cap + eps_rel * sp_m), relax * (eps_cap + eps_rel * sd_m)
end

"""Read one scalar installed capacity from SP_Capacities.csv rows for an agent."""
function _sp_cap_scalar(rows::DataFrame)
    nrow(rows) == 0 && return nothing
    v = rows.cap[1]
    v isa Real && return Float64(v)
    v isa AbstractVector && !isempty(v) && return Float64(v[1])
    return Float64(v)
end

"""Vector z warm-start for merged agents: one SP capacity per member slot."""
function merged_cap_z_from_sp(cap_df::DataFrame, member_ids, n_cap::Int)
    z = zeros(n_cap)
    for (i, mid) in enumerate(member_ids)
        i > n_cap && break
        rows = cap_df[cap_df.AgentID .== String(mid), :]
        val = _sp_cap_scalar(rows)
        val !== nothing && (z[i] = val)
    end
    return z
end
