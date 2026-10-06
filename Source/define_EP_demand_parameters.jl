# ==============================================================================
# define_EP_demand_parameters.jl — Optional elastic EP demand agent parameters
# ==============================================================================
#
# PURPOSE:
#   Placeholder for an optional elastic end-product demand agent. In the current
#   setup, EP demand is fixed via EP_market["D_EP"], so the EP_Demand block in
#   data.yaml is typically empty. If present, this function copies all keys into
#   mod.ext[:parameters] so a future build_EP_demand_agent could use them.
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — data = merge(General, EP_Demand[agent]) or empty.
#
# ==============================================================================

function define_EP_demand_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]

    # Placeholder: currently EP demand is inelastic and fully defined by the
    # EP_market block (Total_Demand × normalized profile) in define_EP_market_parameters!.
    # This function exists so that a future elastic EP demand agent can be added
    # by populating the EP_Demand section in data.yaml (e.g. with A_EP, B_EP for
    # a quadratic utility). Any keys present in data are copied generically.
    for (k, v) in data
        params[Symbol(k)] = v
    end
    return mod
end
