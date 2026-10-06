# ==============================================================================
# define_contract_market_parameters.jl — PPA / HPA contract-market dicts
# ==============================================================================
#
# PURPOSE:
#   Fills `ppa_market` and `hpa_market` from the `PPAs:` and `HPAs:` blocks of
#   data.yaml. Each block has two numerical knobs only — there are no volume
#   modes or price indexations in the standard contract (see contract_settlement.jl):
#
#     initial_price  — ADMM warm start for the contract price K (€/MWh).
#                      `spot` (default): production-weighted expected bundled
#                      spot price on the contract profile at the λ warm start
#                      (the risk-neutral fair value). A number overrides it.
#     rho_initial    — ADMM penalty on the contract-volume consensus
#                      (ρ/2)·A·(C − C̄)², with A the expected MWh per MW-year.
#
#   Optional per-link overrides: `PPAs.<VRES id>.initial_price` and
#   `HPAs.<electrolyzer id>.initial_price`.
#
#   The lists of contract parties (`ppa_vres`, `ppa_buyers`, `hpa_h2`,
#   `hpa_buyers`) are filled later by `link_contract_parties!` once every
#   agent's Type is known.
#
# ==============================================================================

function _parse_contract_cfg(cfg)
    cfg isa Dict || return Dict{String, Any}()
    return Dict{String, Any}(String(k) => v for (k, v) in cfg)
end

function _contract_initial_price_setting(val)
    val === nothing && return "spot"
    val isa Real && return Float64(val)
    s = lowercase(String(val))
    s in ("spot", "fair", "auto") && return "spot"
    parsed = tryparse(Float64, s)
    parsed === nothing && error("Contract initial_price must be a number or `spot`; got $(val).")
    return parsed
end

function _fill_contract_market!(market::Dict, cfg::Dict, name::String, default_rho::Float64)
    market["name"] = name
    market["initial_price"] = _contract_initial_price_setting(get(cfg, "initial_price", "spot"))
    market["rho_initial"] = Float64(get(cfg, "rho_initial", default_rho))
    # Per-link overrides live under keys that are agent ids (sub-dicts).
    per_link = Dict{String, Any}()
    for (k, v) in cfg
        v isa Dict || continue
        per_link[String(k)] = Dict{String, Any}(
            "initial_price" => _contract_initial_price_setting(get(v, "initial_price", market["initial_price"])),
        )
    end
    market["per_link"] = per_link
    return market
end

function define_contract_market_parameters!(ppa_market::Dict, hpa_market::Dict, data::Dict, agents::Dict)
    ppa_cfg = _parse_contract_cfg(get(data, "PPAs", Dict{String, Any}()))
    hpa_cfg = _parse_contract_cfg(get(data, "HPAs", Dict{String, Any}()))
    # 0.02 €/MWh per MW: a 1,000 MW seller/buyer gap moves K by ≈ 20 €/MWh per
    # iteration before η damping; residual balancing adapts it from there.
    _fill_contract_market!(ppa_market, ppa_cfg, "ppa", 0.02)
    _fill_contract_market!(hpa_market, hpa_cfg, "hpa", 0.02)
    for key in (:ppa_market, :hpa_market, :ppa_vres, :ppa_buyers, :hpa_h2, :hpa_buyers)
        haskey(agents, key) || (agents[key] = String[])
    end
    return ppa_market, hpa_market
end

"""Initial K for one link: numeric override, else `"spot"` (resolved in define_results_contracts!)."""
function contract_initial_price_setting(market::Dict, link_id::String)
    per_link = get(market, "per_link", Dict{String, Any}())
    if haskey(per_link, link_id)
        return per_link[link_id]["initial_price"]
    end
    return get(market, "initial_price", "spot")
end
