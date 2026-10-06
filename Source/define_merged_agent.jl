# ==============================================================================
# define_merged_agent.jl — Parameters for merged partial-planner agents
# ==============================================================================

function define_merged_agent!(merged_id::String, mod::Model, merged_type::String,
                              data_run::Dict, ts::Dict, repr_days::Dict,
                              agents::Dict, member_data::AbstractDict{Symbol, <:AbstractDict})
    mod.ext[:sets]        = Dict{Symbol, Any}()
    mod.ext[:parameters]  = Dict{Symbol, Any}()
    mod.ext[:timeseries]  = Dict{Symbol, Any}()
    mod.ext[:variables]   = Dict{Symbol, Any}()
    mod.ext[:constraints] = Dict{Symbol, Any}()
    mod.ext[:expressions] = Dict{Symbol, Any}()

    gen = data_run["General"]
    admm = data_run["ADMM"]
    n_years = gen["nYears"]
    n_repr_days = gen["nReprDays"]
    n_timesteps = gen["nTimesteps"]
    JY = 1:n_years
    JD = 1:n_repr_days
    JH = 1:n_timesteps
    mod.ext[:sets][:JY] = JY
    mod.ext[:sets][:JD] = JD
    mod.ext[:sets][:JH] = JH

    # Scenario labels are 1..nYears (file keys). Fallback if Main.years missing.
    _years = isdefined(Main, :years) ? Main.years : Dict(1 => 1)
    W = [repr_days[_years[jy]][!, :weights][jd] for jd in JD, jy in JY]
    params = mod.ext[:parameters]
    params[:W] = W
    params[:P] = ones(n_years) ./ n_years
    params[:γ] = get(admm, "gamma", 1.0)
    params[:β] = get(admm, "beta", 0.95)
    params[:Type] = merged_type
    params[:MergedMembers] = member_data

    push!(agents[:elec_market], merged_id)
    push!(agents[:elec_GC_market], merged_id)
    push!(agents[:H2_GC_market], merged_id)
    push!(agents[:EP_market], merged_id)
    params[:in_elec_market] = true
    params[:in_H2_market] = false
    params[:in_elec_GC_market] = true
    params[:in_H2_GC_market] = true
    params[:in_EP_market] = true

    shp = (n_timesteps, n_repr_days, n_years)
    params[:λ_elec] = zeros(shp)
    params[:g_bar_elec] = zeros(shp)
    params[:ρ_elec] = 1.0
    params[:λ_elec_GC] = zeros(shp)
    params[:g_bar_elec_GC] = zeros(shp)
    params[:ρ_elec_GC] = 1.0
    params[:λ_H2_GC] = zeros(shp)
    params[:g_bar_H2_GC] = zeros(shp)
    params[:ρ_H2_GC] = 1.0
    params[:λ_EP] = zeros(shp)
    params[:g_bar_EP] = zeros(shp)
    params[:ρ_EP] = 1.0

    h2_data = member_data[:electrolyzer]
    off_data = member_data[:green_offtaker]
    derive_electrolyzer_capacities!(params, h2_data)
    params[:OperationalCost] = get(h2_data, "OperationalCost", 0.0)

    for (k, v) in off_data
        k == "Type" && continue
        params[Symbol(k)] = v
    end
    params[:gamma_GC] = get(off_data, "gamma_GC", 0.42)
    derive_green_offtaker_capacities!(params, off_data)
    params[:ProcessingCost] = get(off_data, "ProcessingCost", 0.0)
    params[:FixedCost_per_MW_EP_Out] = get(off_data, "FixedCost_per_MW_EP_Out", 0.0)
    params[:Type] = merged_type

    cap_slots = String["H2", "EP"]
    if merged_type == "GreenCoalition"
        params[:vres_units] = [
            _merged_vres_unit(:solar, member_data[:solar], ts, repr_days, JH, JD, JY, n_timesteps, _years),
            _merged_vres_unit(:wind, member_data[:wind], ts, repr_days, JH, JD, JY, n_timesteps, _years),
        ]
        cap_slots = String["VRES_solar", "VRES_wind", "H2", "EP"]
    end

    params[:cap_slots] = cap_slots
    n_cap = length(cap_slots)
    params[:z_cap] = zeros(n_cap)
    params[:λ_cap] = zeros(n_cap)
    params[:ρ_cap] = get(admm, "rho_cap_initial", 0.1)
    return mod
end

function _merged_vres_unit(label::Symbol, data::Dict, ts::Dict, repr_days::Dict,
                           JH, JD, JY, n_ts::Int, _years::Dict)
    col = String(data["Profile_Column"])
    AF = Array{Float64}(undef, length(JH), length(JD), length(JY))
    for jy in JY
        yr = _years[jy]
        for jd in JD, jh in JH
            row = (jd - 1) * n_ts + jh
            AF[jh, jd, jy] = ts[yr][!, Symbol(col)][row]
        end
    end
    return (
        label = label,
        id = String(get(data, "agent_id", string(label))),
        Capacity = data["Capacity"],
        MarginalCost = data["MarginalCost"],
        FixedCost_per_MW = get(data, "FixedCost_per_MW", 0.0),
        AF = AF,
    )
end
