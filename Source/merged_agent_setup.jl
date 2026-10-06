# ==============================================================================
# merged_agent_setup.jl — Agent lists for partial-planner (merged green) cases
# ==============================================================================

"""
    setup_merged_agents!(data, planner_key, mdict, optimizer_factory)

Same agent/market structure as `market_exposure.jl`, but replace configured
member IDs with one merged agent. Returns `(agents, merged_id, merged_type, member_ids)`.
"""
function setup_merged_agents!(data::Dict, planner_key::String, mdict::Dict, optimizer_factory)
    cfg = data["PartialPlanners"][planner_key]
    merged_id = String(cfg["coalition_id"])
    members = cfg["members"]
    merged_type = planner_key == "GreenH2" ? "GreenH2Coalition" :
                  planner_key == "Green" ? "GreenCoalition" :
                  error("Unknown PartialPlanners key: $planner_key")

    member_ids = String[]
    if merged_type == "GreenH2Coalition"
        push!(member_ids, String(members["electrolyzer"]))
        push!(member_ids, String(members["green_offtaker"]))
    else
        push!(member_ids, String(members["solar"]))
        push!(member_ids, String(members["wind"]))
        push!(member_ids, String(members["electrolyzer"]))
        push!(member_ids, String(members["green_offtaker"]))
    end

    agents = Dict{Symbol, Any}()
    agents[:power] = [id for id in keys(data["Power"]) if id ∉ member_ids]
    agents[:H2] = [id for id in keys(data["Hydrogen"]) if id ∉ member_ids]
    agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"]) if id ∉ member_ids]
    agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ?
        [id for id in keys(data["Electricity_GC_Demand"])] : String[]
    agents[:merged] = [merged_id]
    agents[:merged_members] = Dict(merged_id => member_ids)
    agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker],
                         agents[:elec_GC_demand], agents[:merged])

    agents[:elec_market] = String[]
    agents[:H2_market] = String[]
    agents[:elec_GC_market] = String[]
    agents[:H2_GC_market] = String[]
    agents[:EP_market] = String[]

    for id in agents[:all]
        if !haskey(mdict, id)
            mdict[id] = Model(optimizer_factory)
        end
    end
    return agents, merged_id, merged_type, member_ids
end

function merged_member_data(data::Dict, planner_key::String)
    cfg = data["PartialPlanners"][planner_key]
    members = cfg["members"]
    if planner_key == "GreenH2"
        return Dict(
            :electrolyzer => merge(data["General"], data["Hydrogen"][String(members["electrolyzer"])]),
            :green_offtaker => merge(data["General"], data["Hydrogen_Offtaker"][String(members["green_offtaker"])]),
        )
    elseif planner_key == "Green"
        solar_id = String(members["solar"])
        wind_id = String(members["wind"])
        solar = merge(data["General"], data["Power"][solar_id])
        wind = merge(data["General"], data["Power"][wind_id])
        solar["agent_id"] = solar_id
        wind["agent_id"] = wind_id
        return Dict(
            :solar => solar,
            :wind => wind,
            :electrolyzer => merge(data["General"], data["Hydrogen"][String(members["electrolyzer"])]),
            :green_offtaker => merge(data["General"], data["Hydrogen_Offtaker"][String(members["green_offtaker"])]),
        )
    end
    error("Unknown PartialPlanners key: $planner_key")
end

"""Set JuMP start values on merged-agent flows from SP primal CSV (member decomposition)."""
function merged_operational_warmstart!(mod::Model, sp_primal::DataFrame, member_ids::Vector{String},
                                         n_ts::Int, n_rd::Int, n_yr::Int)
    t = String(mod.ext[:parameters][:Type])
    prod_id = t == "GreenCoalition" ? member_ids[3] : member_ids[1]
    off_id = t == "GreenCoalition" ? member_ids[4] : member_ids[2]
    vars = mod.ext[:variables]
    JH, JD, JY = mod.ext[:sets][:JH], mod.ext[:sets][:JD], mod.ext[:sets][:JY]
    n_total = n_ts * n_rd * n_yr
    nrow(sp_primal) == n_total || return nothing

    function _col_val(aid::String, suffix::String, jh, jd, jy)
        col = Symbol(aid * suffix)
        hasproperty(sp_primal, col) || return 0.0
        iy = jy
        id = jd
        ih = jh
        row_idx = (iy - 1) * n_rd * n_ts + (id - 1) * n_ts + ih
        return Float64(sp_primal[row_idx, col])
    end

    for (iy, jy) in enumerate(JY), (id, jd) in enumerate(JD), (ih, jh) in enumerate(JH)
        e_buy = -_col_val(prod_id, "_elec", ih, id, iy)
        gc_e = -_col_val(prod_id, "_elec_GC", ih, id, iy)
        h2_int = _col_val(prod_id, "_H2", ih, id, iy) / max(mod.ext[:parameters][:η_elec_H2], 1e-9)
        if haskey(vars, :h2)
            h2_sp = _col_val(off_id, "_H2", ih, id, iy)
            h2_int = h2_int > 0 ? h2_int : -h2_sp
        end
        gc_prod = _col_val(prod_id, "_H2_GC", ih, id, iy)
        gc_off = _col_val(off_id, "_H2_GC", ih, id, iy)
        gc_int = max(0.0, -gc_off)
        gc_ext = max(0.0, gc_prod + gc_off)
        ep_sp = _col_val(off_id, "_EP", ih, id, iy)

        e_buy > 0 && haskey(vars, :e_in) && set_start_value(vars[:e_in][jh, jd, jy], e_buy)
        h2_int > 0 && haskey(vars, :h2) && set_start_value(vars[:h2][jh, jd, jy], h2_int)
        gc_e > 0 && haskey(vars, :q_elec_gc) && set_start_value(vars[:q_elec_gc][jh, jd, jy], gc_e)
        haskey(vars, :q_h2gc_int) && set_start_value(vars[:q_h2gc_int][jh, jd, jy], gc_int)
        haskey(vars, :q_h2gc_ext) && set_start_value(vars[:q_h2gc_ext][jh, jd, jy], gc_ext)
        ep_sp > 0 && haskey(vars, :ep) && set_start_value(vars[:ep][jh, jd, jy], ep_sp)

        if haskey(vars, :g_vres)
            for u in mod.ext[:parameters][:vres_units]
                g_sp = _col_val(String(u.id), "_elec", ih, id, iy)
                g_sp > 0 && set_start_value(vars[:g_vres][u.label][jh, jd, jy], g_sp)
            end
        end
    end
    return nothing
end
