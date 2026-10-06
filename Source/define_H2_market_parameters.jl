# ==============================================================================
# define_H2_market_parameters.jl — Hydrogen market initialization
# ==============================================================================
#
# PURPOSE:
#   Same as electricity market: set name, initial_price, rho_initial, and a
#   trivial prices list. Actual price evolution is in results["λ"]["H2"].
#
# ==============================================================================

function define_H2_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    market["name"]          = "Hydrogen"              # Human-readable label for logging and CSV output
    # Scalar seed for the 3D ADMM price array λ[jh,jd,jy] for hydrogen.
    # ADMM will discover the shadow price of H2; the initial value is just a
    # starting guess (0 is fine because H2 is an internal transfer good).
    market["initial_price"] = data["initial_price"]
    # Initial ADMM penalty ρ for H2; balances convergence speed vs. oscillation
    # in the tightly coupled electrolyzer–offtaker market.
    market["rho_initial"]   = data["rho_initial"]
    # Legacy/placeholder; actual price history is in results["λ"]["H2"].
    market["prices"]        = [data["initial_price"]]
    return market
end
