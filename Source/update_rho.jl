# ==============================================================================
# update_rho.jl — Minimal residual-balancing rho update
# ==============================================================================
#
# PURPOSE:
#   Apply a single residual-balancing rule after residuals are computed:
#     if rp > μ*rd -> increase ρ
#     if rd > μ*rp -> decrease ρ
#     else         -> keep ρ
#
#   The same logic is applied:
#     - market-wise for flow markets in ADMM_state["ρ"]
#     - agent-wise for capacity split penalties in ADMM_state["Capacity"]["ρ"]
#
# ARGUMENTS:
#   ADMM_state — Must contain Residuals["Primal"] and ["Dual"] per market, and
#     ρ[key] as a list (we use the last element and push a new one).
#     For capacity, ADMM_state["Capacity"] holds per-agent state.
#   iter — Current iteration index.
#
# ==============================================================================

function update_rho!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return
    # ------------------------------------------------------------------
    # REFORMED CONTROLLER (minimal, canonical residual balancing)
    # ------------------------------------------------------------------
    μ = get(ADMM_state, "rho_balance_threshold", 1.2)

    # Standard flow markets
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        isempty(ADMM_state["Residuals"]["Primal"][key]) && continue
        isempty(ADMM_state["Residuals"]["Dual"][key]) && continue
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        ρ  = ADMM_state["ρ"][key][end]

        if !isfinite(rp) || !isfinite(rd)
            push!(ADMM_state["ρ"][key], ρ)
            continue
        end

        if key in ("elec", "elec_GC", "H2_GC")
            τ = 1.05
            ρ_max = key == "H2_GC" ? 100.0 : 5_000.0
        else
            τ = 1.01
            ρ_max = 100.0
        end
        ρ_min = 1e-4

        if rp > μ * rd
            push!(ADMM_state["ρ"][key], min(ρ_max, τ * ρ))
        elseif rd > μ * rp
            push!(ADMM_state["ρ"][key], max(ρ_min, ρ / τ))
        else
            push!(ADMM_state["ρ"][key], ρ)
        end
    end

    # Capacity per-agent equality split (x_cap = z_cap)
    cap_state = get(ADMM_state, "Capacity", nothing)
    if cap_state !== nothing
        τ_cap = get(ADMM_state, "rho_cap_inc_factor", 1.05)
        ρ_max_cap = get(ADMM_state, "rho_cap_max", 30.0)
        ρ_min_cap = 0.10
        for m in get(cap_state, "agents", String[])
            isempty(cap_state["Primal"][m]) && continue
            isempty(cap_state["Dual"][m]) && continue
            rp = cap_state["Primal"][m][end]
            rd = cap_state["Dual"][m][end]
            ρ  = cap_state["ρ"][m][end]

            if !isfinite(rp) || !isfinite(rd)
                push!(cap_state["ρ"][m], ρ)
                continue
            end

            if rp > μ * rd
                push!(cap_state["ρ"][m], min(ρ_max_cap, τ_cap * ρ))
            elseif rd > μ * rp
                push!(cap_state["ρ"][m], max(ρ_min_cap, ρ / τ_cap))
            else
                push!(cap_state["ρ"][m], ρ)
            end
        end
    end

    return nothing
end
