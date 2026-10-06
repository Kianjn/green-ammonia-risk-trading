# ==============================================================================
# build_offtaker_agent_contracts.jl — Offtakers in the contracts case (ME+C)
# ==============================================================================
#
# PURPOSE:
#   Builds the same physical offtaker models as build_offtaker_agent! and, for
#   the GreenOfftaker, adds its side of the bilateral HPA:
#
#     C_hpa_buy[h]  — capacity (MW_H2) bought under a fixed-price, baseload HPA
#                     from electrolyzer h. In every slot the contracted volume is
#                     C_hpa_buy[h]. The offtaker pays electrolyzer h the fixed
#                     price K_hpa,h on that volume every hour. Together with pool
#                     purchases at λ, the contracted green H₂ costs K_hpa,h.
#
#   Physical H₂ and GC purchases and EP sales stay in the pools as in
#   market_exposure.jl. The payment is a cash flow between the two firms in
#   every scenario and enters the offtaker's CVaR (solve_offtaker_agent_contracts.jl).
#
#   Bound: C_hpa_buy[h] ≤ cap_EP_y / α — contracted H₂ cannot exceed the plant's
#   maximum H₂ intake (α is the H₂→EP conversion ratio).
#
#   Grey offtaker and importer are unchanged.
#
# ==============================================================================

function build_offtaker_agent_contracts!(m::String, mod::Model, EP_market::Dict, H2_market::Dict,
                                         H2_GC_market::Dict, hpa_market::Dict)
    build_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)

    p = mod.ext[:parameters]
    (String(get(p, :Type, "")) == "GreenOfftaker" && get(p, :in_hpa_market, false)) || return mod

    cap_EP_y = mod.ext[:variables][:cap_EP_y]
    alpha = get(p, :Alpha, 1.0)
    h2_ids = get(hpa_market, "hpa_h2", String[])
    C_buy = Dict{String, VariableRef}()
    lim = Dict{String, ConstraintRef}()
    for h in h2_ids
        C_buy[h] = @variable(mod, lower_bound = 0, base_name = "C_hpa_buy_$(m)_$(h)")
        # Contracted MW_H2 cannot exceed the plant's maximum H₂ intake.
        lim[h] = @constraint(mod, C_buy[h] <= cap_EP_y / alpha)
    end
    mod.ext[:variables][:C_hpa_buy] = C_buy
    mod.ext[:constraints][:hpa_within_intake] = lim
    return mod
end
