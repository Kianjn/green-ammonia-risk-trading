# ==============================================================================
# define_electricity_GC_market_parameters.jl — Electricity GC market initialization
# ==============================================================================
#
# PURPOSE:
#   Initialize the electricity Guarantees of Origin market: name, initial price,
#   rho_initial. Prices in the ADMM loop are stored in results["λ"]["elec_GC"].
#
# ==============================================================================

function define_electricity_GC_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    market["name"]          = "Electricity_GC"        # Human-readable label for logging and CSV output
    # Scalar seed for the 3D ADMM price array λ[jh,jd,jy] for electricity GCs.
    # ADMM updates it each iteration; this scalar seeds the uniform initial grid.
    market["initial_price"] = data["initial_price"]
    # Initial ADMM penalty ρ for the electricity GC market. Set lower than the
    # electricity market because GC volumes and price ranges are smaller, so a
    # gentler penalty avoids oscillation in this thinner market.
    market["rho_initial"]   = data["rho_initial"]
    # Legacy/placeholder; actual price history is in results["λ"]["elec_GC"].
    market["prices"]        = [data["initial_price"]]
    return market
end
