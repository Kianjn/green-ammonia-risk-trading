# ==============================================================================
# define_results_contracts.jl — Results / ADMM state for the contracts case
# ==============================================================================
#
# PURPOSE:
#   Calls define_results! (spot markets, capacity buffers, warm starts) and then
#   adds the state of the bilateral contract markets used by me_contracts.jl:
#
#   Contract PRICES (the ADMM duals of the contract-volume clearing conditions):
#     results["K_ppa"][v]  Vector{Float64}   K of the PPA sold by VRES v, per iteration
#     results["K_hpa"][h]  Vector{Float64}   K of the HPA sold by electrolyzer h
#
#   Contract VOLUMES (scalar MW, one entry per iteration, pushed by
#   ADMM_subroutine_contracts!):
#     results["C_ppa_sell"][v]       C offered by VRES v
#     results["C_ppa_buy"][b][v]     C demanded by electrolyzer b from VRES v
#     results["C_hpa_sell"][h]       C offered by electrolyzer h
#     results["C_hpa_buy"][b][h]     C demanded by offtaker b from electrolyzer h
#
#   Contract ADMM state, ADMM["ppa"] and ADMM["hpa"], each a Dict with
#     "ids"                    seller ids (one link per seller)
#     "n"                      consensus denominator (1 seller + #buyers)
#     "Imbalance"[id]          C_sell − Σ C_buy per iteration (MW)
#     "Primal"[id]             |imbalance| per iteration
#     "Dual"[id]               sharing-ADMM dual residual per iteration
#     "ρ"[id]                  penalty history (first = rho_initial)
#     "ResidualScale_Primal/Dual"[id]  Boyd relative-tolerance scales
#     "A"[id]                  expected contracted MWh per MW-year (from link_contract_parties!)
#
#   K WARM START. With `initial_price: spot` (default) K starts at the
#   risk-neutral fair value: the volume-weighted expected bundled spot price on
#   the contract profile, evaluated at the λ warm start (contract_fair_price).
#   In a risk-neutral run this is already the equilibrium contract price; in a
#   risk-averse run it is the natural starting point from which the risk
#   premium emerges (K falls below fair value when the seller values the hedge
#   more, rises above it when the buyer does).
#
# ARGUMENTS:
#   Same as define_results!, plus ppa_market, hpa_market (party lists filled by
#   link_contract_parties!) and mdict (to read W, P and AF for the warm start).
#
# ==============================================================================

function define_results_contracts!(admm_data::Dict, results::Dict, ADMM::Dict, agents::Dict,
                                   elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                   H2_GC_market::Dict, EP_market::Dict,
                                   ppa_market::Dict, hpa_market::Dict, mdict::Dict;
                                   sp_prices_file::String = "", sp_primal_file::String = "",
                                   sp_cap_file::String = "", use_primal_warmstart::Bool = true)
    define_results!(admm_data, results, ADMM, agents,
                    elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market;
                    sp_prices_file = sp_prices_file, sp_primal_file = sp_primal_file,
                    sp_cap_file = sp_cap_file, use_primal_warmstart = use_primal_warmstart)

    results["markets"]["ppa"] = ppa_market
    results["markets"]["hpa"] = hpa_market

    ppa_vres   = get(ppa_market, "ppa_vres", String[])
    ppa_buyers = get(ppa_market, "ppa_buyers", String[])
    hpa_h2     = get(hpa_market, "hpa_h2", String[])
    hpa_buyers = get(hpa_market, "hpa_buyers", String[])

    λ_elec0    = results["λ"]["elec"][1]
    λ_elec_GC0 = results["λ"]["elec_GC"][1]
    λ_H20      = results["λ"]["H2"][1]
    λ_H2_GC0   = results["λ"]["H2_GC"][1]

    # ── PPA links (one per VRES) ───────────────────────────────────────────
    results["K_ppa"]      = Dict{String, Vector{Float64}}()
    results["C_ppa_sell"] = Dict{String, Vector{Float64}}(v => Float64[] for v in ppa_vres)
    results["C_ppa_buy"]  = Dict{String, Dict{String, Vector{Float64}}}(
        b => Dict{String, Vector{Float64}}(v => Float64[] for v in ppa_vres) for b in ppa_buyers)
    A_ppa = Dict{String, Float64}()
    for v in ppa_vres
        mv = mdict[v]
        pv = mv.ext[:parameters]
        JH, JD, JY = mv.ext[:sets][:JH], mv.ext[:sets][:JD], mv.ext[:sets][:JY]
        AF = Float64.(mv.ext[:timeseries][:AF])
        A_ppa[v] = get(pv, :A_ppa, contract_annual_volume_per_MW(pv[:W], pv[:P], AF, JH, JD, JY))
        setting = contract_initial_price_setting(ppa_market, v)
        K0 = setting isa Real ? Float64(setting) :
             contract_fair_price(pv[:W], pv[:P], ppa_bundle_spot(λ_elec0, λ_elec_GC0), AF, JH, JD, JY)
        results["K_ppa"][v] = [K0]
    end
    ADMM["ppa"] = _new_contract_state(ppa_vres, ppa_market["rho_initial"],
                                      get(ppa_market, "nAgents", 1 + length(ppa_buyers)), A_ppa)

    # ── HPA links (one per electrolyzer) ──────────────────────────────────
    results["K_hpa"]      = Dict{String, Vector{Float64}}()
    results["C_hpa_sell"] = Dict{String, Vector{Float64}}(h => Float64[] for h in hpa_h2)
    results["C_hpa_buy"]  = Dict{String, Dict{String, Vector{Float64}}}(
        b => Dict{String, Vector{Float64}}(h => Float64[] for h in hpa_h2) for b in hpa_buyers)
    A_hpa = Dict{String, Float64}()
    for h in hpa_h2
        mh = mdict[h]
        ph = mh.ext[:parameters]
        JH, JD, JY = mh.ext[:sets][:JH], mh.ext[:sets][:JD], mh.ext[:sets][:JY]
        prof = flat_contract_profile((length(JH), length(JD), length(JY)))
        A_hpa[h] = get(ph, :A_hpa, contract_annual_volume_per_MW(ph[:W], ph[:P], prof, JH, JD, JY))
        setting = contract_initial_price_setting(hpa_market, h)
        K0 = setting isa Real ? Float64(setting) :
             contract_fair_price(ph[:W], ph[:P], hpa_bundle_spot(λ_H20, λ_H2_GC0), prof, JH, JD, JY)
        results["K_hpa"][h] = [K0]
    end
    ADMM["hpa"] = _new_contract_state(hpa_h2, hpa_market["rho_initial"],
                                      get(hpa_market, "nAgents", 1 + length(hpa_buyers)), A_hpa)

    return results, ADMM
end

"""Allocate the per-link ADMM state of one contract market (PPA or HPA)."""
function _new_contract_state(ids::Vector{String}, rho0::Real, n::Int, A::Dict{String, Float64})
    return Dict{String, Any}(
        "ids"       => copy(ids),
        "n"         => n,
        "Imbalance" => Dict{String, Vector{Float64}}(id => Float64[] for id in ids),
        "Primal"    => Dict{String, Vector{Float64}}(id => Float64[] for id in ids),
        "Dual"      => Dict{String, Vector{Float64}}(id => Float64[] for id in ids),
        "ρ"         => Dict{String, Vector{Float64}}(id => [Float64(rho0)] for id in ids),
        "ResidualScale_Primal" => Dict{String, Float64}(id => 0.0 for id in ids),
        "ResidualScale_Dual"   => Dict{String, Float64}(id => 0.0 for id in ids),
        "A"         => Dict{String, Float64}(id => get(A, id, 0.0) for id in ids),
    )
end
