# ==============================================================================
# define_H2_GC_market_parameters.jl — Hydrogen GC market initialization
# ==============================================================================
#
# PURPOSE:
#   Initialize the hydrogen guarantees market (H₂ GC): name, initial price,
#   rho_initial. Used by the electrolyzer (supply) and green/grey offtakers (demand).
#
# ==============================================================================

function define_H2_GC_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    market["name"]          = "H2_GC"                 # Human-readable label for logging and CSV output
    # Scalar seed for the 3D ADMM price array λ[jh,jd,jy] for hydrogen GCs.
    # Higher initial guess reflects the renewable premium embedded in H₂
    # certificates (green H₂ commands a premium over grey).
    market["initial_price"] = data["initial_price"]
    # Initial ADMM penalty ρ for H₂ GC market. Lower value (like elec_GC)
    # because the certificate market is thin; gentle updates prevent price
    # whipsawing between the electrolyzer and the green/grey offtakers.
    market["rho_initial"]   = data["rho_initial"]
    # Legacy/placeholder; actual price history is in results["λ"]["H2_GC"].
    market["prices"]        = [data["initial_price"]]
    return market
end
