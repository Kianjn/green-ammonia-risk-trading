# ==============================================================================
# build_H2_agent_contracts.jl — Electrolyzer in the contracts case (ME+C)
# ==============================================================================
#
# PURPOSE:
#   Builds the same physical electrolyzer model as build_H2_agent! (e_in,
#   h2_out, GCs, cap_H2_y, CVaR) and adds both contract positions of the
#   GreenProducer:
#
#     C_ppa_buy[v]  — capacity (MW_e) bought under the fixed-price, as-produced
#                     PPA from VRES v. In slot (h,d,y) the contracted volume is
#                     AF_v·C_ppa_buy[v]. The electrolyzer pays VRES v the fixed
#                     price K_ppa,v on that volume. Together with pool purchases
#                     at λ, the contracted green MWh cost K_ppa,v.
#     C_hpa         — capacity (MW_H2) sold under a fixed-price, baseload HPA.
#                     The offtaker pays K_hpa on C_hpa every hour. Together with
#                     pool sales at λ, that MW_H2 of output earns K_hpa.
#
#   Physical dispatch is untouched: electricity is bought in the pool and
#   hydrogen is sold in the pool exactly as in market_exposure.jl. The contract
#   terms are transfers between the two firms (revenue in one loss, cost in the
#   other) and therefore sit inside both CVaR terms (solve_H2_agent_contracts.jl).
#
#   Bounds tie contract size to the plant (hedging, not speculation):
#     C_ppa_buy[v] ≤ cap_H2_y / η   contracted power ≤ maximum electrical intake
#     C_hpa        ≤ cap_H2_y       contracted H₂ ≤ output capacity
#
# ==============================================================================

function build_H2_agent_contracts!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict,
                                   ppa_market::Dict)
    build_H2_agent!(m, mod, H2_market, H2_GC_market)

    p = mod.ext[:parameters]
    String(get(p, :Type, "")) == "GreenProducer" || return mod
    cap_H2_y = mod.ext[:variables][:cap_H2_y]
    η = p[:η_elec_H2]

    if get(p, :in_ppa_market, false)
        vres_ids = get(ppa_market, "ppa_vres", String[])
        C_buy = Dict{String, VariableRef}()
        lim = Dict{String, ConstraintRef}()
        for v in vres_ids
            C_buy[v] = @variable(mod, lower_bound = 0, base_name = "C_ppa_buy_$(m)_$(v)")
            # Contracted MW_e cannot exceed what the electrolyzer can absorb.
            lim[v] = @constraint(mod, C_buy[v] <= cap_H2_y / η)
        end
        mod.ext[:variables][:C_ppa_buy] = C_buy
        mod.ext[:constraints][:ppa_within_intake] = lim
    end

    if get(p, :in_hpa_market, false)
        C_hpa = mod.ext[:variables][:C_hpa] = @variable(mod, lower_bound = 0, base_name = "C_hpa_$(m)")
        # A baseload HPA on C_hpa MW_H2 must be deliverable every hour.
        mod.ext[:constraints][:hpa_within_plant] = @constraint(mod, C_hpa <= cap_H2_y)
    end
    return mod
end
