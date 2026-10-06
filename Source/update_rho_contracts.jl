# ==============================================================================
# update_rho_contracts.jl — Residual-balancing ρ update incl. contract links
# ==============================================================================
#
# PURPOSE:
#   Calls update_rho! for the five spot markets (and the leftover capacity
#   bookkeeping) and then applies the same canonical residual-balancing rule to
#   every bilateral contract link in ADMM_state["ppa"] and ADMM_state["hpa"]:
#
#     if rp > μ·rd  → ρ ← min(ρ_max, τ·ρ)
#     if rd > μ·rp  → ρ ← max(ρ_min, ρ/τ)
#     else          → keep ρ
#
#   ρ for a contract link has units €/MWh per MW: K moves by η·ρ·imbalance per
#   iteration and the QP penalty is (ρ/2)·A·(C − C̄)². ρ_max keeps a large
#   early volume gap from producing absurd price jumps; ρ_min keeps the link
#   responsive.
#
# ==============================================================================

function update_rho_contracts!(ADMM_state::Dict, iter::Int)
    update_rho!(ADMM_state, iter)

    μ     = get(ADMM_state, "rho_balance_threshold", 1.2)
    τ     = 1.05
    ρ_max = Float64(get(ADMM_state, "rho_contract_max", 0.5))
    ρ_min = 1e-4

    for ck in ("ppa", "hpa")
        haskey(ADMM_state, ck) || continue
        S = ADMM_state[ck]
        for id in S["ids"]
            isempty(S["Primal"][id]) && continue
            isempty(S["Dual"][id]) && continue
            rp = S["Primal"][id][end]
            rd = S["Dual"][id][end]
            ρ  = S["ρ"][id][end]
            if !isfinite(rp) || !isfinite(rd)
                push!(S["ρ"][id], ρ)
                continue
            end
            if rp > μ * rd
                push!(S["ρ"][id], min(ρ_max, τ * ρ))
            elseif rd > μ * rp
                push!(S["ρ"][id], max(ρ_min, ρ / τ))
            else
                push!(S["ρ"][id], ρ)
            end
        end
    end
    return nothing
end
