# ==============================================================================
# define_EP_market_parameters.jl — End-product market and fixed demand profile
# ==============================================================================
#
# PURPOSE:
#   Initialize the end-product coordination "market": name, initial price,
#   rho_initial, and the fixed demand profile D_EP[jh, jd, jy]. The profile is
#   built from Total_Demand and the normalized column Demand_Column (e.g.
#   LOAD_EP) in the time series: at each (jh, jd, jy) the demand is
#   Total_Demand * profile_value, so the demand is inelastic and shaped by the
#   profile. The ADMM imbalance for EP is (sum of offtaker supplies) - D_EP.
#
# ARGUMENTS:
#   market, data, ts, repr_days — data contains Demand_Column, Total_Demand,
#     initial_price, rho_initial; ts and repr_days are used to build the 3D grid.
#
# ==============================================================================

function define_EP_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    market["name"]          = "End_Product"           # Human-readable label for logging and CSV output
    # Scalar seed for the 3D ADMM price array λ[jh,jd,jy] for end product.
    # ADMM updates it each iteration; this value seeds the uniform initial grid.
    market["initial_price"] = data["initial_price"]
    # Initial ADMM penalty ρ for the EP market. Larger than other markets because
    # the EP market is stiff (few participants, large price swings) and needs
    # stronger coupling to converge without excessive iterations.
    market["rho_initial"]   = data["rho_initial"]
    market["Demand_Column"] = data["Demand_Column"]   # Timeseries column with the normalized EP demand shape
    market["Total_Demand"]  = data["Total_Demand"]    # Annual total EP demand (MWh_EP); scaling factor for the profile
    # Legacy/placeholder; actual price history is in results["λ"]["EP"].
    market["prices"]        = [data["initial_price"]]

    n_ts = data["nTimesteps"]
    n_rd = data["nReprDays"]
    n_yr = data["nYears"]
    # Scenario labels are 1..nYears (file keys). Fallback if Main.years missing.
    _years    = isdefined(Main, :years) ? Main.years : Dict(1 => 1)

    # Construct the fixed inelastic demand profile D_EP[jh, jd, jy].
    # At each (jh, jd, jy), demand = Total_Demand × normalized timeseries value.
    # This means EP demand is exogenous and does not respond to price; the ADMM
    # imbalance for the EP market is (sum of offtaker supplies) − D_EP.
    # Row mapping: the timeseries CSV stores representative-day data sequentially
    # in its first (nReprDays * nTimesteps) rows, so we index directly:
    # row = (jd-1)*n_ts + jh.
    D_EP = Array{Float64}(undef, n_ts, n_rd, n_yr)
    col  = Symbol(data["Demand_Column"])   # Convert column name to Symbol for DataFrame indexing
    tot  = data["Total_Demand"]            # Scalar annual total (MWh_EP)
    for jy in 1:n_yr
        yr = _years[jy]
        for jd in 1:n_rd, jh in 1:n_ts
            row = (jd - 1) * n_ts + jh
            D_EP[jh, jd, jy] = tot * ts[yr][!, col][row]
        end
    end
    market["D_EP"] = D_EP   # Full 3D demand tensor used by ADMM_subroutine to compute EP imbalance
    # Per-year 2D slice of D_EP for social-planner compatibility: social_planner
    # indexes demand by year key (integer), so we provide demand[y] = D_EP[:,:,y].
    market["demand"] = Dict(y => D_EP[:, :, y] for y in 1:n_yr)
    return market
end
