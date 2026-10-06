# ==============================================================================
# define_electricity_market_parameters.jl — Electricity market initialization
# ==============================================================================
#
# PURPOSE:
#   Fills the electricity market dictionary with a display name, initial price
#   (scalar used to seed results["λ"]["elec"] in define_results!), and
#   rho_initial. The "prices" list is optional legacy; actual price history is
#   stored in results["λ"]["elec"] as 3D arrays per iteration.
#
# ARGUMENTS:
#   market — Dict to be filled (elec_market in the main script).
#   data   — merge(General, ADMM, elec_market) from data.yaml.
#   ts, repr_days — Not used for this market; kept for uniform signature.
#
# ==============================================================================

function define_electricity_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    market["name"]          = "Electricity"           # Human-readable label for logging and CSV output
    # Scalar seed for the 3D ADMM price array λ[jh,jd,jy]. define_results! will
    # call fill(initial_price, shp) to create a uniform starting price grid.
    # ADMM updates this grid every iteration; the scalar here is just the seed.
    market["initial_price"] = data["initial_price"]
    # Initial ADMM penalty parameter ρ for this market. Balances convergence
    # speed (higher ρ → faster primal convergence) against oscillation risk
    # (higher ρ → larger dual steps → potential price swings).
    market["rho_initial"]   = data["rho_initial"]
    # Legacy/placeholder price list; the actual iteration-by-iteration price
    # history is stored in results["λ"]["elec"] (a list of 3D arrays).
    market["prices"]        = [data["initial_price"]]
    return market
end
