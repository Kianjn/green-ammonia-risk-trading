# ==============================================================================
# define_contract_parameters.jl — Per-agent contract flags and ADMM placeholders
# ==============================================================================
#
# PURPOSE:
#   Called once per agent after define_common_parameters!. It (1) decides which
#   bilateral contract markets the agent takes part in from its Type,
#   (2) registers the agent in the party lists, and (3) allocates the
#   per-link ADMM placeholders that ADMM_subroutine_contracts! refreshes every
#   iteration:
#
#     :K_<ppa|hpa>      contract price K (€/MWh) — the ADMM dual of the
#                       contract-volume clearing condition C_sell = C_buy
#     :C_bar_<ppa|hpa>  sharing-ADMM consensus target for the agent's NET
#                       contract position (+C for the seller, −C for the buyer)
#     :ρ_<ppa|hpa>      penalty weight on (net position − C_bar)²
#     :A_<ppa|hpa>      expected contracted MWh per MW-year (penalty scale)
#     :AF_ppa           buyer only: the seller's availability profile, needed to
#                       value an as-produced PPA
#
#   Sellers hold scalars keyed implicitly by their own id; buyers hold Dicts
#   keyed by the seller's id (one PPA per VRES, one HPA per electrolyzer).
#
# PARTIES (by Type):
#   PPA  seller VRES            ↔ buyer GreenProducer (electrolyzer)
#   HPA  seller GreenProducer   ↔ buyer GreenOfftaker
#
# ==============================================================================

function define_contract_parameters!(m::String, mod::Model, data::Dict, agents::Dict)
    p = mod.ext[:parameters]
    agent_type = String(get(p, :Type, ""))

    for key in (:ppa_market, :hpa_market, :ppa_vres, :ppa_buyers, :hpa_h2, :hpa_buyers)
        haskey(agents, key) || (agents[key] = String[])
    end

    in_ppa = agent_type in ("VRES", "GreenProducer")
    in_hpa = agent_type in ("GreenProducer", "GreenOfftaker")
    p[:in_ppa_market] = in_ppa
    p[:in_hpa_market] = in_hpa
    in_ppa && push!(agents[:ppa_market], m)
    in_hpa && push!(agents[:hpa_market], m)

    if agent_type == "VRES"
        # PPA seller: one contract on its own plant.
        push!(agents[:ppa_vres], m)
        p[:K_ppa]     = 0.0
        p[:C_bar_ppa] = 0.0
        p[:ρ_ppa]     = 0.0
        p[:A_ppa]     = 0.0
    elseif agent_type == "GreenProducer"
        # PPA buyer (one contract per VRES) and HPA seller (its own H₂ output).
        push!(agents[:ppa_buyers], m)
        push!(agents[:hpa_h2], m)
        p[:K_ppa]     = Dict{String, Float64}()
        p[:C_bar_ppa] = Dict{String, Float64}()
        p[:ρ_ppa]     = Dict{String, Float64}()
        p[:A_ppa]     = Dict{String, Float64}()
        p[:AF_ppa]    = Dict{String, Array{Float64, 3}}()
        p[:K_hpa]     = 0.0
        p[:C_bar_hpa] = 0.0
        p[:ρ_hpa]     = 0.0
        p[:A_hpa]     = 0.0
    elseif agent_type == "GreenOfftaker"
        # HPA buyer (one contract per electrolyzer).
        push!(agents[:hpa_buyers], m)
        p[:K_hpa]     = Dict{String, Float64}()
        p[:C_bar_hpa] = Dict{String, Float64}()
        p[:ρ_hpa]     = Dict{String, Float64}()
        p[:A_hpa]     = Dict{String, Float64}()
    end
    return mod, agents
end

"""
    link_contract_parties!(mdict, agents, ppa_market, hpa_market)

Called once after every agent has been parametrised and BEFORE the models are
built. Copies the party lists into the market dicts, computes the expected
contracted MWh per MW-year `A` of every link, and hands each PPA buyer the
availability profile of every VRES it can contract with (the buyer must know the
seller's production profile to value an as-produced PPA).
"""
function link_contract_parties!(mdict::Dict, agents::Dict, ppa_market::Dict, hpa_market::Dict)
    ppa_vres   = copy(get(agents, :ppa_vres, String[]))
    ppa_buyers = copy(get(agents, :ppa_buyers, String[]))
    hpa_h2     = copy(get(agents, :hpa_h2, String[]))
    hpa_buyers = copy(get(agents, :hpa_buyers, String[]))
    ppa_market["ppa_vres"]   = ppa_vres
    ppa_market["ppa_buyers"] = ppa_buyers
    hpa_market["hpa_h2"]     = hpa_h2
    hpa_market["hpa_buyers"] = hpa_buyers
    # Consensus denominator per link: one seller plus every buyer.
    ppa_market["nAgents"] = 1 + length(ppa_buyers)
    hpa_market["nAgents"] = 1 + length(hpa_buyers)

    # PPA links — as-produced profile AF_v of the selling plant.
    for v in ppa_vres
        mv = mdict[v]
        pv = mv.ext[:parameters]
        JH, JD, JY = mv.ext[:sets][:JH], mv.ext[:sets][:JD], mv.ext[:sets][:JY]
        AF = Float64.(mv.ext[:timeseries][:AF])
        A_v = contract_annual_volume_per_MW(pv[:W], pv[:P], AF, JH, JD, JY)
        pv[:A_ppa] = A_v
        for b in ppa_buyers
            pb = mdict[b].ext[:parameters]
            pb[:AF_ppa][v]    = AF
            pb[:A_ppa][v]     = A_v
            pb[:K_ppa][v]     = 0.0
            pb[:C_bar_ppa][v] = 0.0
            pb[:ρ_ppa][v]     = 0.0
        end
    end

    # HPA links — baseload profile (1 in every slot).
    for h in hpa_h2
        mh = mdict[h]
        ph = mh.ext[:parameters]
        JH, JD, JY = mh.ext[:sets][:JH], mh.ext[:sets][:JD], mh.ext[:sets][:JY]
        shp = (length(JH), length(JD), length(JY))
        A_h = contract_annual_volume_per_MW(ph[:W], ph[:P], flat_contract_profile(shp), JH, JD, JY)
        ph[:A_hpa] = A_h
        for b in hpa_buyers
            pb = mdict[b].ext[:parameters]
            pb[:A_hpa][h]     = A_h
            pb[:K_hpa][h]     = 0.0
            pb[:C_bar_hpa][h] = 0.0
            pb[:ρ_hpa][h]     = 0.0
        end
    end
    return nothing
end
