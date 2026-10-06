# ==============================================================================
# compute_social_risk_metrics.jl — Social CVaR reporting (SP and ADMM)
# ==============================================================================

using JuMP
using DataFrames
using CSV
import Printf: @sprintf, @printf
using Statistics

"""
    empirical_cvar(loss, P, beta) -> (cvar, alpha, u)

Rockafellar–Uryasev CVaR on discrete scenarios: min_alpha alpha + (1/(1-beta)) E[(L-alpha)_+].
"""
function empirical_cvar(loss::AbstractVector{<:Real}, P::AbstractVector{<:Real}, beta::Real)
    loss = Float64.(loss)
    P = Float64.(P)
    n = length(loss)
    length(P) == n || error("loss and P length mismatch")
    sP = sum(P)
    abs(sP - 1.0) > 1e-8 && (P = P ./ sP)
    one_minus_beta = max(1e-6, 1.0 - Float64(beta))

    best_cvar = Inf
    best_alpha = 0.0
    best_u = zeros(n)
    candidates = unique(vcat(loss, [minimum(loss) - 1.0, maximum(loss) + 1.0]))
    for a in candidates
        u = max.(loss .- a, 0.0)
        c = a + sum(P[i] * u[i] for i in eachindex(P)) / one_minus_beta
        if c < best_cvar
            best_cvar, best_alpha, best_u = c, a, u
        end
    end
    return best_cvar, best_alpha, best_u
end

"""Per-year planner welfare for one agent (no market transfers)."""
function _agent_welfare_per_year(m::Model, agents::Dict; JH, JD, JY, W)
    p = m.ext[:parameters]
    ts = m.ext[:timeseries]
    vars = m.ext[:variables]
    atype = String(get(p, :Type, ""))
    Wd = Dict(jy => Dict(jd => W[jd, jy] for jd in JD) for jy in JY)
    wy = Dict{Int, Float64}()

    if atype == "Consumer"
        A_E = get(p, :A_E, 500.0)
        B_E = get(p, :B_E, 0.5)
        d = vars[:d]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * (A_E * value(d[jh, jd, jy]) -
                        0.5 * B_E * value(d[jh, jd, jy])^2) for jh in JH, jd in JD)
        end

    elseif atype == "VRES"
        C = get(p, :MarginalCost, 0.0)
        F_cap = get(p, :FixedCost_per_MW, 0.0)
        cap = vars[:cap_VRES]
        gen_at = if haskey(vars, :g_EOM)
            (jh, jd, jy) -> value(vars[:g_EOM][jh, jd, jy])
        else
            g = vars[:g]
            (jh, jd, jy) -> value(g[jh, jd, jy])
        end
        for jy in JY
            op = sum(Wd[jy][jd] * C * gen_at(jh, jd, jy) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif atype == "Conventional"
        stage_cap = get(p, :ConvStageCap, [0.0, 0.0, 0.0])
        stage_base = get(p, :ConvStageBaseCost, zeros(3, length(JY)))
        stage_slope = get(p, :ConvStageSlope, zeros(3, length(JY)))
        if haskey(vars, :g_stage)
            for jy in JY
                cost = 0.0
                for jh in JH, jd in JD, s in 1:3
                    gs = value(vars[:g_stage][s, jh, jd, jy])
                    cost += Wd[jy][jd] * (stage_base[s, jy] * gs + 0.5 * stage_slope[s, jy] * gs^2)
                end
                wy[jy] = -cost
            end
        else
            MC_y = get(p, :MarginalCostByYear, nothing)
            C = get(p, :MarginalCost, 0.0)
            g = vars[:g]
            for jy in JY
                mc_jy = MC_y === nothing ? C : MC_y[jy]
                wy[jy] = -sum(Wd[jy][jd] * mc_jy * value(g[jh, jd, jy]) for jh in JH, jd in JD)
            end
        end

    elseif atype == "GreenProducer"
        C_H = get(p, :MarginalCost, get(p, :OperationalCost, 0.0))
        F_cap = electrolyzer_h2_annuity(p)
        cap = vars[:cap_H2_y]
        h = vars[:h2_out]
        for jy in JY
            op = sum(Wd[jy][jd] * C_H * value(h[jh, jd, jy]) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif haskey(vars, :d_H)
        utility_val = get(p, :Utility, 0.0)
        d_H = vars[:d_H]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * utility_val * value(d_H[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif atype == "GreenOfftaker"
        C_proc = get(p, :MarginalCost, get(p, :ProcessingCost, 0.0))
        F_cap = get(p, :FixedCost_per_MW_EP_Out, 0.0)
        cap = vars[:cap_EP_y]
        ep = vars[:ep]
        for jy in JY
            op = sum(Wd[jy][jd] * C_proc * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif atype == "GreyOfftaker"
        C_proc = get(p, :MarginalCostByYear, fill(get(p, :MarginalCost, 0.0), length(JY)))
        ep = vars[:ep]
        for jy in JY
            wy[jy] = -sum(Wd[jy][jd] * C_proc[jy] * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif atype == "EPImporter"
        C_proc = get(p, :ImportCost, 0.0)
        ep = vars[:ep]
        for jy in JY
            wy[jy] = -sum(Wd[jy][jd] * C_proc * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif atype in ("GreenH2Coalition", "GreenCoalition")
        # Real-resource surplus of the merged chain (no λ transfers): −OPEX − CAPEX.
        op_cost = get(p, :OperationalCost, 0.0)
        proc_cost = get(p, :ProcessingCost, 0.0)
        F_h2 = electrolyzer_h2_annuity(p)
        F_ep = get(p, :FixedCost_per_MW_EP_Out, 0.0)
        h2 = vars[:h2]
        ep = vars[:ep]
        cap_h2 = vars[:cap_H2_y]
        cap_ep = vars[:cap_EP_y]
        g_vres = get(vars, :g_vres, Dict())
        cap_vres = get(vars, :cap_vres, Dict())
        units = get(p, :vres_units, ())
        for jy in JY
            cost = sum(Wd[jy][jd] * (op_cost * value(h2[jh, jd, jy]) +
                                     proc_cost * value(ep[jh, jd, jy])) for jh in JH, jd in JD)
            cost += F_h2 * value(cap_h2) + F_ep * value(cap_ep)
            for u in units
                g = get(g_vres, u.label, nothing)
                cap = get(cap_vres, u.label, nothing)
                g === nothing && continue
                cost += sum(Wd[jy][jd] * u.MarginalCost * value(g[jh, jd, jy]) for jh in JH, jd in JD)
                cap !== nothing && (cost += u.FixedCost_per_MW * value(cap))
            end
            wy[jy] = -cost
        end

    elseif haskey(vars, :d_gc)
        A_GC = get(p, :A_GC, 0.0)
        B_GC = get(p, :B_GC, 0.5)
        d_gc = vars[:d_gc]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * (A_GC * value(d_gc[jh, jd, jy]) -
                        0.5 * B_GC * value(d_gc[jh, jd, jy])^2) for jh in JH, jd in JD)
        end
    end

    return wy
end

"""Planner-welfare role: demand utility vs real-resource surplus (costs)."""
function _welfare_role(atype::AbstractString, vars)
    if atype == "Consumer"
        return (group = "elec_demand", is_demand = true)
    elseif atype == "GC_Demand" || haskey(vars, :d_gc)
        return (group = "gc_demand", is_demand = true)
    elseif atype == "EP_Demand"
        return (group = "ep_demand", is_demand = true)
    elseif haskey(vars, :d_H) && atype != "GreenProducer"
        return (group = "h2_demand", is_demand = true)
    elseif atype == "VRES"
        return (group = "vres", is_demand = false)
    elseif atype == "Conventional"
        return (group = "conventional", is_demand = false)
    elseif atype == "GreenProducer"
        return (group = "h2_producer", is_demand = false)
    elseif atype == "GreenOfftaker"
        return (group = "offtaker_green", is_demand = false)
    elseif atype == "GreyOfftaker"
        return (group = "offtaker_grey", is_demand = false)
    elseif atype == "EPImporter"
        return (group = "offtaker_import", is_demand = false)
    elseif atype == "GreenH2Coalition"
        return (group = "green_h2_coalition", is_demand = false)
    elseif atype == "GreenCoalition"
        return (group = "green_coalition", is_demand = false)
    else
        return (group = "other", is_demand = false)
    end
end

function _empty_wy(JY)
    Dict{Int, Float64}(jy => 0.0 for jy in JY)
end

"""
Per-agent planner welfare (no transfers). `planner_wpy` is used for the
centralised SP (mdict is not solved). ADMM fills `wy` from mdict values.
"""
function collect_agent_welfare_table(mdict::Dict, agents::Dict, JY;
                                     planner_wpy::Union{Nothing, Dict}=nothing)
    ref = mdict[agents[:all][1]]
    JH = collect(ref.ext[:sets][:JH])
    JD = collect(ref.ext[:sets][:JD])
    W = ref.ext[:parameters][:W]
    ids = planner_wpy === nothing ? collect(agents[:all]) : collect(keys(planner_wpy))
    rows = NamedTuple[]
    for id in ids
        haskey(mdict, id) || continue
        m = mdict[id]
        atype = String(get(m.ext[:parameters], :Type, ""))
        vars = m.ext[:variables]
        role = _welfare_role(atype, vars)
        wy = _empty_wy(JY)
        if planner_wpy !== nothing && haskey(planner_wpy, id)
            for jy in JY
                wy[jy] = Float64(value(planner_wpy[id][jy]))
            end
        else
            computed = _agent_welfare_per_year(m, agents; JH=JH, JD=JD, JY=JY, W=W)
            for (jy, w) in computed
                wy[jy] = Float64(w)
            end
        end
        push!(rows, (
            agent = id,
            type = atype,
            group = role.group,
            is_demand = role.is_demand,
            welfare_per_year = wy,
        ))
    end
    return rows
end

function summarize_welfare_decomposition(agent_rows, JY, P::AbstractVector{<:Real}, beta::Real)
    n = length(JY)
    demand_y = zeros(n)
    rest_y = zeros(n)
    total_y = zeros(n)
    group_y = Dict{String, Vector{Float64}}()
    group_is_demand = Dict{String, Bool}()
    for r in agent_rows
        haskey(group_y, r.group) || (group_y[r.group] = zeros(n))
        group_is_demand[r.group] = r.is_demand
        for (i, jy) in enumerate(JY)
            w = get(r.welfare_per_year, jy, 0.0)
            total_y[i] += w
            group_y[r.group][i] += w
            if r.is_demand
                demand_y[i] += w
            else
                rest_y[i] += w
            end
        end
    end
    expected_total = sum(P[i] * total_y[i] for i in 1:n)
    expected_demand = sum(P[i] * demand_y[i] for i in 1:n)
    expected_rest = sum(P[i] * rest_y[i] for i in 1:n)
    share = abs(expected_total) > 1e-12 ? expected_demand / expected_total : NaN
    cvar_rest, _, _ = empirical_cvar(-rest_y, P, beta)
    group_expected = Dict(g => sum(P[i] * v[i] for i in 1:n) for (g, v) in group_y)
    return (
        welfare_demand_per_year = demand_y,
        welfare_ex_demand_per_year = rest_y,
        expected_welfare_demand = expected_demand,
        expected_welfare_ex_demand = expected_rest,
        share_demand_of_E_SW = share,
        welfare_ex_demand_CVaR = cvar_rest,
        welfare_ex_demand_min = minimum(rest_y),
        welfare_ex_demand_spread = maximum(rest_y) - minimum(rest_y),
        group_welfare_per_year = group_y,
        group_is_demand = group_is_demand,
        group_expected = group_expected,
        agent_rows = agent_rows,
    )
end

function aggregate_social_welfare_per_year(mdict::Dict, agents::Dict)
    ref = mdict[agents[:all][1]]
    JH = collect(ref.ext[:sets][:JH])
    JD = collect(ref.ext[:sets][:JD])
    JY = collect(ref.ext[:sets][:JY])
    W = ref.ext[:parameters][:W]
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    social = Dict{Int, Float64}(jy => 0.0 for jy in JY)
    for id in agents[:all]
        haskey(mdict, id) || continue
        wy = _agent_welfare_per_year(mdict[id], agents; JH=JH, JD=JD, JY=JY, W=W)
        for (jy, w) in wy
            social[jy] += w
        end
    end
    return JY, P, [social[jy] for jy in JY]
end

function extract_private_cvar_by_agent(mdict::Dict, agents::Dict)
    rows = NamedTuple[]
    for id in agents[:all]
        m = mdict[id]
        t = String(get(m.ext[:parameters], :Type, ""))
        vars = m.ext[:variables]
        cvar_val, alpha_val = NaN, NaN
        if t == "VRES" && haskey(vars, :CVaR_VRES)
            cvar_val = value(vars[:CVaR_VRES])
            alpha_val = haskey(vars, :alpha_VRES) ? value(vars[:alpha_VRES]) : NaN
        elseif t == "GreenProducer" && haskey(vars, :CVaR_H2)
            cvar_val = value(vars[:CVaR_H2])
            alpha_val = haskey(vars, :alpha_H2) ? value(vars[:alpha_H2]) : NaN
        elseif t == "GreenOfftaker" && haskey(vars, :CVaR_GreenOfftaker)
            cvar_val = value(vars[:CVaR_GreenOfftaker])
            alpha_val = haskey(vars, :alpha_GreenOfftaker) ? value(vars[:alpha_GreenOfftaker]) : NaN
        elseif t in ("GreenH2Coalition", "GreenCoalition") && haskey(vars, :CVaR_coalition)
            cvar_val = value(vars[:CVaR_coalition])
            alpha_val = haskey(vars, :alpha_coalition) ? value(vars[:alpha_coalition]) : NaN
        end
        isfinite(cvar_val) || continue
        push!(rows, (agent=id, type=t, cvar_private=cvar_val, alpha_private=alpha_val))
    end
    return rows
end

"""Parse a numeric metric cell from Risk_Metrics.csv (may be String or Real)."""
function _parse_metric_value(v)
    v isa Real && return Float64(v)
    v isa Missing && return nothing
    s = strip(string(v))
    (isempty(s) || s == "-" || lowercase(s) == "nan") && return nothing
    return tryparse(Float64, s)
end

function load_sp_social_cvar_benchmark(project_root::String)
    path = joinpath(project_root, "social_planner_results", "Risk_Metrics.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    # Prefer the ex-post recomputed CVaR. The stored `social_CVaR` is the planner's
    # cvar_social variable, which carries zero objective weight at gamma = 1 and is
    # therefore left arbitrarily loose; comparing an ADMM ex-post CVaR against it
    # would report a large spurious gap. Fall back to it only if the recomputed row
    # is missing (older result files).
    for key in ("social_CVaR_recomputed", "social_CVaR")
        for i in 1:nrow(df)
            String(df.Metric[i]) == key || continue
            parsed = _parse_metric_value(df.Value[i])
            parsed === nothing || return parsed
        end
    end
    return nothing
end

function extract_sp_risk_metrics(planner::Model, planner_state::Dict, mdict::Dict, agents::Dict)
    JY = collect(planner_state[:JY])
    ref = mdict[agents[:all][1]]
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    gamma = Float64(planner_state[:gamma])
    beta = Float64(planner_state[:beta])

    sw_aux = planner_state[:sw_aux]
    social_welfare = planner_state[:social_welfare]
    sw_y = Float64[value(sw_aux[jy]) for jy in JY]
    swelfare_y = Float64[value(social_welfare[jy]) for jy in JY]
    loss_y = -sw_y

    cvar_model = value(planner_state[:cvar_social])
    alpha_model = value(planner_state[:alpha_social])
    cvar_check, alpha_check, _ = empirical_cvar(loss_y, P, beta)

    expected_welfare = sum(P[jy] * sw_y[jy] for jy in JY)
    # Must match build_social_planner! objective: γ·Σ P[y]·sw_aux[y] − (1−γ)·CVaR.
    # Read from the solved model (not sum(sw_y), which ignores P and inflates by nYears).
    risk_adj_obj = objective_value(planner)
    if !isfinite(risk_adj_obj)
        risk_adj_obj = gamma * expected_welfare - (1 - gamma) * cvar_model
    end

    agent_rows = collect_agent_welfare_table(mdict, agents, JY;
        planner_wpy = get(planner_state, :agent_welfare_per_year, nothing))
    decomp = summarize_welfare_decomposition(agent_rows, JY, P, beta)

    return (
        case = "social_planner",
        gamma = gamma,
        beta = beta,
        expected_social_welfare = expected_welfare,
        social_welfare_per_year = swelfare_y,
        social_loss_per_year = loss_y,
        social_CVaR = cvar_model,
        alpha_social = alpha_model,
        social_CVaR_recomputed = cvar_check,
        sum_private_CVaR = NaN,
        risk_adjusted_objective = risk_adj_obj,
        social_CVaR_gap_vs_SP = 0.0,
        private_cvar_rows = NamedTuple[],
        decomp...,
    )
end

"""Read γ and β from agent parameters (stored as :γ/:β in define_common_parameters!)."""
function _read_risk_params(p::Dict)
    gamma = Float64(get(p, :γ, get(p, :gamma, 1.0)))
    beta  = Float64(get(p, :β, get(p, :beta, 0.95)))
    return gamma, beta
end

function extract_admm_risk_metrics(mdict::Dict, agents::Dict, case_label::String; project_root::String)
    ref = mdict[agents[:all][1]]
    gamma, beta = _read_risk_params(ref.ext[:parameters])
    JY, P, swelfare_y = aggregate_social_welfare_per_year(mdict, agents)
    loss_y = -swelfare_y
    cvar_expost, alpha_expost, _ = empirical_cvar(loss_y, P, beta)
    expected_welfare = sum(P[jy] * swelfare_y[jy] for jy in JY)
    priv = extract_private_cvar_by_agent(mdict, agents)
    sum_priv = isempty(priv) ? 0.0 : sum(r.cvar_private for r in priv)
    sp_cvar = load_sp_social_cvar_benchmark(project_root)
    gap = sp_cvar === nothing ? NaN : (cvar_expost - sp_cvar)

    agent_rows = collect_agent_welfare_table(mdict, agents, JY)
    decomp = summarize_welfare_decomposition(agent_rows, JY, P, beta)

    return (
        case = case_label,
        gamma = gamma,
        beta = beta,
        expected_social_welfare = expected_welfare,
        social_welfare_per_year = swelfare_y,
        social_loss_per_year = loss_y,
        social_CVaR = cvar_expost,
        alpha_social = alpha_expost,
        social_CVaR_recomputed = cvar_expost,
        sum_private_CVaR = sum_priv,
        risk_adjusted_objective = NaN,
        social_CVaR_gap_vs_SP = gap,
        private_cvar_rows = priv,
        decomp...,
    )
end

function _risk_metrics_table(metrics::NamedTuple)
    sp_bench = metrics.case == "social_planner" ? metrics.social_CVaR :
        (isfinite(metrics.social_CVaR_gap_vs_SP) ? metrics.social_CVaR - metrics.social_CVaR_gap_vs_SP : NaN)
    names = String[
        "case", "gamma", "beta", "expected_social_welfare", "social_CVaR", "alpha_social",
        "social_CVaR_recomputed", "sum_private_CVaR", "risk_adjusted_objective",
        "social_CVaR_gap_vs_SP", "SP_social_CVaR_benchmark",
        "expected_welfare_demand", "expected_welfare_ex_demand", "share_demand_of_E_SW",
        "welfare_ex_demand_CVaR", "welfare_ex_demand_min", "welfare_ex_demand_spread",
    ]
    vals = Any[
        metrics.case, metrics.gamma, metrics.beta, metrics.expected_social_welfare,
        metrics.social_CVaR, metrics.alpha_social, metrics.social_CVaR_recomputed,
        metrics.sum_private_CVaR, metrics.risk_adjusted_objective,
        metrics.social_CVaR_gap_vs_SP, sp_bench,
        metrics.expected_welfare_demand, metrics.expected_welfare_ex_demand,
        metrics.share_demand_of_E_SW, metrics.welfare_ex_demand_CVaR,
        metrics.welfare_ex_demand_min, metrics.welfare_ex_demand_spread,
    ]
    units = String["-", "-", "-", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR",
                   "EUR", "EUR", "-", "EUR", "EUR", "EUR"]
    if hasproperty(metrics, :group_expected)
        for g in sort(collect(keys(metrics.group_expected)))
            push!(names, "expected_welfare_group_" * g)
            push!(vals, metrics.group_expected[g])
            push!(units, "EUR")
        end
    end
    DataFrame(Metric = names, Value = vals, Unit = units)
end

function _write_welfare_decomposition_csvs!(metrics::NamedTuple, JY, P, results_dir::String)
    n = length(JY)
    CSV.write(joinpath(results_dir, "Social_Welfare_Per_Year.csv"),
        DataFrame(case=fill(metrics.case, n), scenario_year=JY, probability=P,
                  social_welfare=metrics.social_welfare_per_year,
                  social_loss=metrics.social_loss_per_year,
                  welfare_demand=metrics.welfare_demand_per_year,
                  welfare_ex_demand=metrics.welfare_ex_demand_per_year))

    agent_year_rows = NamedTuple[]
    for r in metrics.agent_rows
        for (i, jy) in enumerate(JY)
            push!(agent_year_rows, (
                case = metrics.case,
                agent = r.agent,
                type = r.type,
                group = r.group,
                is_demand = r.is_demand,
                scenario_year = jy,
                probability = P[i],
                welfare = get(r.welfare_per_year, jy, 0.0),
            ))
        end
    end
    CSV.write(joinpath(results_dir, "Welfare_By_Agent_Per_Year.csv"), DataFrame(agent_year_rows))

    group_year_rows = NamedTuple[]
    for g in sort(collect(keys(metrics.group_welfare_per_year)))
        vec = metrics.group_welfare_per_year[g]
        is_d = get(metrics.group_is_demand, g, false)
        for (i, jy) in enumerate(JY)
            push!(group_year_rows, (
                case = metrics.case,
                group = g,
                is_demand = is_d,
                scenario_year = jy,
                probability = P[i],
                welfare = vec[i],
            ))
        end
    end
    CSV.write(joinpath(results_dir, "Welfare_By_Group_Per_Year.csv"), DataFrame(group_year_rows))
    return nothing
end

function write_sp_risk_outputs!(planner::Model, planner_state::Dict, mdict::Dict,
                                agents::Dict, results_folder::String)
    metrics = extract_sp_risk_metrics(planner, planner_state, mdict, agents)
    CSV.write(joinpath(results_folder, "Risk_Metrics.csv"), _risk_metrics_table(metrics))
    JY = collect(planner_state[:JY])
    P = [mdict[agents[:all][1]].ext[:parameters][:P][jy] for jy in JY]
    _write_welfare_decomposition_csvs!(metrics, JY, P, results_folder)
    return metrics
end

function write_admm_risk_outputs!(mdict::Dict, agents::Dict, results_dir::String;
                                  case_label::String = "market_exposure")
    project_root = dirname(results_dir)
    metrics = extract_admm_risk_metrics(mdict, agents, case_label; project_root=project_root)
    CSV.write(joinpath(results_dir, "Risk_Metrics.csv"), _risk_metrics_table(metrics))
    ref = mdict[agents[:all][1]]
    JY = collect(ref.ext[:sets][:JY])
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    _write_welfare_decomposition_csvs!(metrics, JY, P, results_dir)
    if !isempty(metrics.private_cvar_rows)
        CSV.write(joinpath(results_dir, "Private_CVaR_By_Agent.csv"),
            DataFrame(
                Agent=[r.agent for r in metrics.private_cvar_rows],
                Type=[r.type for r in metrics.private_cvar_rows],
                CVaR_private=[r.cvar_private for r in metrics.private_cvar_rows],
                alpha_private=[r.alpha_private for r in metrics.private_cvar_rows],
            ))
    end
    return metrics
end

function print_risk_metrics_summary!(metrics::NamedTuple; title::String = "Risk metrics")
    # Internal CVaR is on loss L = -welfare; report tail welfare = -CVaR(L) (higher is better).
    # Always report the EX-POST empirical CVaR of the realised welfare vector rather
    # than the planner's cvar_social variable. At gamma = 1 the objective puts zero
    # weight on that variable, so the solver leaves it anywhere above its true value
    # and the printed tail would otherwise come out below the worst scenario, which
    # is impossible. At gamma < 1 the two agree because the objective drives it down.
    tail_welfare = -metrics.social_CVaR_recomputed
    min_sw = minimum(metrics.social_welfare_per_year)
    spread = maximum(metrics.social_welfare_per_year) - min_sw
    tail_pct = round(100 * (1 - metrics.beta); digits=0)
    risk_neutral = metrics.gamma >= 1.0 - 1e-12
    n_scen = length(metrics.social_welfare_per_year)

    println()
    println("-" ^ 72)
    println("  ", title)
    println("-" ^ 72)
    @printf("  Case:                    %s\n", metrics.case)
    @printf("  gamma:                   %.4f\n", metrics.gamma)
    @printf("  beta:                    %.4f  (tail = worst %.0f%% of scenario years)\n",
            metrics.beta, tail_pct)
    @printf("  E[social welfare]:            %10.3f bn EUR\n", metrics.expected_social_welfare / 1e9)
    @printf("    demand utility:             %10.3f bn EUR  (%.1f%% of net SW)\n",
            metrics.expected_welfare_demand / 1e9, 100 * metrics.share_demand_of_E_SW)
    @printf("    ex-demand (rest):           %10.3f bn EUR\n", metrics.expected_welfare_ex_demand / 1e9)
    @printf("  Tail welfare (CVaR):          %10.3f bn EUR  (higher = safer tail)\n", tail_welfare / 1e9)
    @printf("  Min scenario welfare:         %10.3f bn EUR\n", min_sw / 1e9)
    @printf("  Welfare spread (max−min):     %10.3f bn EUR\n", spread / 1e9)
    @printf("  E[ex-demand] spread:          %10.3f bn EUR  (max−min of rest)\n",
            metrics.welfare_ex_demand_spread / 1e9)
    @printf("  Tail of ex-demand:            %10.3f bn EUR  (ex-post; not in the objective)\n",
            -metrics.welfare_ex_demand_CVaR / 1e9)

    if metrics.case == "social_planner"
        @printf("  Risk-adjusted objective:        %10.3f bn EUR\n", metrics.risk_adjusted_objective / 1e9)
    elseif isfinite(metrics.social_CVaR_gap_vs_SP)
        sp_tail = tail_welfare + metrics.social_CVaR_gap_vs_SP  # gap is on loss: ME − SP
        tail_gap = -metrics.social_CVaR_gap_vs_SP               # ME tail − SP tail
        @printf("  SP tail welfare (benchmark):    %10.3f bn EUR\n", sp_tail / 1e9)
        @printf("  Tail welfare gap vs SP:        %+10.1f M EUR  (negative ⇒ ME worse)\n", tail_gap / 1e6)
    end

    if risk_neutral
        println("  (gamma = 1: risk-neutral; tail lines are ex-post at the stated beta.)")
    end
    # With n equiprobable scenarios a tail of (1-beta) narrower than 1/n collapses
    # onto the single worst scenario, so beta loses all resolution. Flag it, because
    # a risk-aversion sweep over such betas would show no variation at all.
    if (1 - metrics.beta) < 1 / n_scen - 1e-9
        @printf("  (NOTE: beta = %.2f implies a %.1f%% tail, narrower than one of the %d\n",
                metrics.beta, 100 * (1 - metrics.beta), n_scen)
        @printf("   equiprobable scenarios (%.1f%%), so CVaR collapses to the worst scenario.\n",
                100 / n_scen)
        @printf("   For a tail spanning k scenarios use beta = 1 - k/%d, e.g. %.3f for k=3.)\n",
                n_scen, 1 - 3 / n_scen)
    end
    return nothing
end

# ==============================================================================
# Cost metrics for the console log (system cost + consumer €/MWh)
# ==============================================================================

function _slot_value(x, jh, jd, jy)
    v = x[jh, jd, jy]
    return Float64(v isa Number ? v : value(v))
end

function _add_price_qty!(exp_y::Vector{Float64}, qty_y::Vector{Float64},
                         λ, q, W, JH, JD, JY)
    q === nothing && return
    for (iy, jy) in enumerate(JY), jd in JD, jh in JH
        qq = _slot_value(q, jh, jd, jy)
        pp = _slot_value(λ, jh, jd, jy)
        w = W[jd, jy]
        exp_y[iy] += w * pp * qq
        qty_y[iy] += w * qq
    end
    return nothing
end

function _prob_expect(yvec::Vector{Float64}, P::Vector{Float64})
    return sum(P[i] * yvec[i] for i in eachindex(yvec))
end

"""γ·E[x] + (1−γ)·CVaR_β(x). CVaR is on the high-x tail (x = cost or expenditure)."""
function _risk_adjust_cost(yvec::Vector{Float64}, P::Vector{Float64}, gamma::Float64, beta::Float64)
    e = _prob_expect(yvec, P)
    gamma >= 1.0 - 1e-12 && return e
    cv, _, _ = empirical_cvar(yvec, P, beta)
    return gamma * e + (1.0 - gamma) * cv
end

function _safe_unit_price(exp::Float64, qty::Float64)
    return qty > 1.0e-12 ? exp / qty : NaN
end

"""
Real-resource system cost per scenario year (€/year): every CAPEX + OPEX / fuel
expense, excluding consumer utility and excluding market transfers.
"""
function _system_cost_per_year(mdict::Dict, agents::Dict, JY;
                               planner_wpy::Union{Nothing, Dict}=nothing)
    rows = collect_agent_welfare_table(mdict, agents, JY; planner_wpy=planner_wpy)
    cost_y = zeros(length(JY))
    for r in rows
        r.is_demand && continue
        for (i, jy) in enumerate(JY)
            cost_y[i] -= r.welfare_per_year[jy]
        end
    end
    return cost_y
end

function _final_h2_qty(id::String, atype::AbstractString, vars, vd)
    # Standalone H₂ demand only. Offtaker / merged-chain H₂ is ammonia feedstock.
    atype in ("GreenOfftaker", "GreenProducer", "GreenH2Coalition", "GreenCoalition") && return nothing
    if vd !== nothing
        h2d = get(vd, :H2_d_H, Dict())
        return get(h2d, id, nothing)
    end
    return haskey(vars, :d_H) ? vars[:d_H] : nothing
end

function _final_ep_qty(id::String, atype::AbstractString, vars, vd)
    atype in ("GreenOfftaker", "GreyOfftaker", "EPImporter",
              "GreenH2Coalition", "GreenCoalition") || return nothing
    if vd !== nothing
        haskey(get(vd, :offtaker_ep_sell, Dict()), id) && return vd[:offtaker_ep_sell][id]
        haskey(get(vd, :offtaker_ep_sell_import, Dict()), id) && return vd[:offtaker_ep_sell_import][id]
        return nothing
    end
    return get(vars, :ep, nothing)
end

function _elec_demand_qty(id::String, atype::AbstractString, vars, vd)
    atype == "Consumer" || return nothing
    if vd !== nothing
        return get(get(vd, :power_d_E, Dict()), id, nothing)
    end
    return get(vars, :d, nothing)
end

"""
Collect expected social welfare, resource cost, and consumer €/MWh metrics.

Resource cost is CAPEX + fuel/OPEX (no transfers). Final energy = electricity
consumed by power consumers + any standalone H₂ demand + ammonia (EP).
Hydrogen that is converted to ammonia is excluded.
"""
function collect_cost_metrics(mdict::Dict, agents::Dict;
                              planner_state::Union{Nothing, Dict}=nothing,
                              λ_elec=nothing, λ_H2=nothing, λ_EP=nothing)
    ref = mdict[agents[:all][1]]
    JH = collect(ref.ext[:sets][:JH])
    JD = collect(ref.ext[:sets][:JD])
    JY = collect(ref.ext[:sets][:JY])
    W = ref.ext[:parameters][:W]
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    gamma, beta = _read_risk_params(ref.ext[:parameters])
    planner_wpy = nothing
    vd = nothing
    if planner_state !== nothing
        JH = collect(planner_state[:JH])
        JD = collect(planner_state[:JD])
        JY = collect(planner_state[:JY])
        W = planner_state[:W]
        gamma = Float64(get(planner_state, :gamma, gamma))
        beta = Float64(get(planner_state, :beta, beta))
        planner_wpy = get(planner_state, :agent_welfare_per_year, nothing)
        vd = get(planner_state, :var_dict, nothing)
        P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    end

    λ_elec === nothing && (λ_elec = ref.ext[:parameters][:λ_elec])
    λ_H2 === nothing && (λ_H2 = ref.ext[:parameters][:λ_H2])
    λ_EP === nothing && (λ_EP = ref.ext[:parameters][:λ_EP])

    n = length(JY)
    elec_exp = zeros(n); elec_qty = zeros(n)
    h2_exp = zeros(n);   h2_qty = zeros(n)
    ep_exp = zeros(n);   ep_qty = zeros(n)

    for id in agents[:all]
        haskey(mdict, id) || continue
        atype = String(get(mdict[id].ext[:parameters], :Type, ""))
        vars = mdict[id].ext[:variables]
        q_e = _elec_demand_qty(id, atype, vars, vd)
        q_e !== nothing && _add_price_qty!(elec_exp, elec_qty, λ_elec, q_e, W, JH, JD, JY)
        q_h = _final_h2_qty(id, atype, vars, vd)
        q_h !== nothing && _add_price_qty!(h2_exp, h2_qty, λ_H2, q_h, W, JH, JD, JY)
        q_p = _final_ep_qty(id, atype, vars, vd)
        q_p !== nothing && _add_price_qty!(ep_exp, ep_qty, λ_EP, q_p, W, JH, JD, JY)
    end

    rows = collect_agent_welfare_table(mdict, agents, JY; planner_wpy=planner_wpy)
    cost_y = zeros(n)
    sw_y = zeros(n)
    for r in rows
        for (i, jy) in enumerate(JY)
            w = r.welfare_per_year[jy]
            sw_y[i] += w
            r.is_demand || (cost_y[i] -= w)
        end
    end
    total_exp_y = elec_exp .+ h2_exp .+ ep_exp
    total_qty_y = elec_qty .+ h2_qty .+ ep_qty

    e_sw = _prob_expect(sw_y, P)
    e_cost = _prob_expect(cost_y, P)
    ra_cost = _risk_adjust_cost(cost_y, P, gamma, beta)

    e_elec_exp = _prob_expect(elec_exp, P)
    e_elec_qty = _prob_expect(elec_qty, P)
    ra_elec_exp = _risk_adjust_cost(elec_exp, P, gamma, beta)
    elec_cwap = _safe_unit_price(e_elec_exp, e_elec_qty)
    elec_ra_cwap = _safe_unit_price(ra_elec_exp, e_elec_qty)

    e_h2_exp = _prob_expect(h2_exp, P)
    e_h2_qty = _prob_expect(h2_qty, P)
    e_ep_exp = _prob_expect(ep_exp, P)
    e_ep_qty = _prob_expect(ep_qty, P)

    e_tot_exp = _prob_expect(total_exp_y, P)
    e_tot_qty = _prob_expect(total_qty_y, P)
    ra_tot_exp = _risk_adjust_cost(total_exp_y, P, gamma, beta)
    blended = _safe_unit_price(e_tot_exp, e_tot_qty)
    blended_ra = _safe_unit_price(ra_tot_exp, e_tot_qty)

    return (
        gamma = gamma,
        beta = beta,
        expected_social_welfare = e_sw,
        resource_cost = e_cost,
        system_cost_expected = e_cost,
        system_cost_risk_adjusted = ra_cost,
        elec_cwap = elec_cwap,
        elec_ra_cwap = elec_ra_cwap,
        elec_qty = e_elec_qty,
        elec_exp = e_elec_exp,
        h2_cwap = _safe_unit_price(e_h2_exp, e_h2_qty),
        h2_qty = e_h2_qty,
        h2_exp = e_h2_exp,
        ep_cwap = _safe_unit_price(e_ep_exp, e_ep_qty),
        ep_qty = e_ep_qty,
        ep_exp = e_ep_exp,
        blended_cwap = blended,
        blended_ra_cwap = blended_ra,
        total_exp = e_tot_exp,
        total_qty = e_tot_qty,
        total_exp_risk_adjusted = ra_tot_exp,
    )
end

function print_cost_metrics_summary!(metrics::NamedTuple; title::String = "Welfare and cost metrics")
    nz(x) = isfinite(x) ? x : 0.0
    resource_cost = hasproperty(metrics, :resource_cost) ?
        metrics.resource_cost : metrics.system_cost_expected
    e_sw = hasproperty(metrics, :expected_social_welfare) ?
        metrics.expected_social_welfare : NaN
    println()
    println("-" ^ 72)
    println("  ", title)
    println("-" ^ 72)
    @printf("  Expected social welfare:                %10.3f bn EUR\n", e_sw / 1e9)
    @printf("  Total resource cost:                    %10.3f bn EUR/year\n",
            resource_cost / 1e9)
    println("    (CAPEX + fuel/OPEX; no market transfers. Equals −E[ex-demand welfare].)")
    @printf("  Consumer CWAP (electricity):            %10.2f €/MWh\n",
            nz(metrics.elec_cwap))
    @printf("  Expected consumer cost (final energy):  %10.2f €/MWh\n",
            nz(metrics.blended_cwap))
    @printf("    expenditure / energy:                 %10.3f bn EUR  /  %.2f TWh\n",
            metrics.total_exp / 1e9, metrics.total_qty / 1e6)
    @printf("    electricity                           %10.2f €/MWh   (%.2f TWh)\n",
            nz(metrics.elec_cwap), metrics.elec_qty / 1e6)
    @printf("    ammonia (EP)                          %10.2f €/MWh   (%.2f TWh)\n",
            nz(metrics.ep_cwap), metrics.ep_qty / 1e6)
    @printf("    hydrogen (final only)                 %10.2f €/MWh   (%.2f TWh)\n",
            nz(metrics.h2_cwap), metrics.h2_qty / 1e6)
    println("    (H₂ converted to ammonia is excluded from final energy.)")
    println("    Hours weighted by representative-day weights W and scenario probabilities.")
    println("    Consumer €/MWh is expected expenditure / expected MWh, not a CVaR mix.")
    return nothing
end

function write_cost_metrics_csv(metrics::NamedTuple, results_dir::String)
    resource_cost = hasproperty(metrics, :resource_cost) ?
        metrics.resource_cost : metrics.system_cost_expected
    e_sw = hasproperty(metrics, :expected_social_welfare) ?
        metrics.expected_social_welfare : NaN
    CSV.write(joinpath(results_dir, "Cost_Metrics.csv"), DataFrame(
        Metric = [
            "gamma", "beta", "expected_social_welfare", "resource_cost",
            "elec_cwap", "ep_cwap", "h2_cwap", "blended_cwap",
            "elec_exp", "ep_exp", "h2_exp", "total_exp",
            "elec_qty", "ep_qty", "h2_qty", "total_qty",
        ],
        Value = [
            metrics.gamma, metrics.beta, e_sw, resource_cost,
            metrics.elec_cwap, metrics.ep_cwap, metrics.h2_cwap, metrics.blended_cwap,
            metrics.elec_exp, metrics.ep_exp, metrics.h2_exp, metrics.total_exp,
            metrics.elec_qty, metrics.ep_qty, metrics.h2_qty, metrics.total_qty,
        ],
        Unit = [
            "-", "-", "EUR", "EUR/year",
            "EUR/MWh", "EUR/MWh", "EUR/MWh", "EUR/MWh",
            "EUR", "EUR", "EUR", "EUR",
            "MWh", "MWh", "MWh", "MWh",
        ],
    ))
    return nothing
end
