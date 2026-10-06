# ==============================================================================
# define_results.jl — Initialize result and ADMM state structures
# ==============================================================================

if !isdefined(@__MODULE__, :repeat_last_agent_quantities!)
    include(joinpath(@__DIR__, "cap_admm_helpers.jl"))
end

# PURPOSE:
#   Allocates all dictionaries and arrays used during and after the ADMM loop:
#   (1) results: agents reference, markets reference, initial price 3D arrays
#       per market, and per-agent lists for quantities (g, h2, elec_GC, H2_GC, EP)
#       that will each hold one 3D array per iteration. (2) ADMM: initial ρ per
#       market, empty lists for full imbalance tensors, scalar PriceHistory and
#       ImbalanceMean per market (filled each iteration in ADMM.jl), Primal/Dual
#       residual lists, Tolerance per market, n_iter and walltime.
#
# ARGUMENTS:
#   admm_data — merge(General, ADMM) for nTimesteps, nReprDays, nYears, epsilon.
#   results, ADMM — Dicts to be filled (passed by reference).
#   agents — Dict of agent lists (stored in results for save_results).
#   elec_market, H2_market, ... — Market dicts (stored in results["markets"] and
#     used to read initial_price and rho_initial).
#
# ==============================================================================

"""Load one agent's SP primal 3D arrays; coalition agents sum member columns."""
function _fill_sp_primal_agent!(g_arr, h2_arr, gc_arr, h2gc_arr, ep_arr,
                                sp_primal::DataFrame, agent_ids,
                                n_ts::Int, n_rd::Int, n_yr::Int)
    cols = (
        ("_elec", g_arr),
        ("_H2", h2_arr),
        ("_elec_GC", gc_arr),
        ("_H2_GC", h2gc_arr),
        ("_EP", ep_arr),
    )
    for (iy, jy) in enumerate(1:n_yr), (id, jd) in enumerate(1:n_rd), (ih, jh) in enumerate(1:n_ts)
        row_idx = (iy - 1) * n_rd * n_ts + (id - 1) * n_ts + ih
        for aid in agent_ids
            for (suffix, arr) in cols
                col = Symbol(string(aid) * suffix)
                hasproperty(sp_primal, col) || continue
                arr[ih, id, iy] += sp_primal[row_idx, col]
            end
        end
    end
    return nothing
end

function define_results!(admm_data::Dict, results::Dict, ADMM::Dict, agents::Dict,
                        elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                        H2_GC_market::Dict, EP_market::Dict;
                        sp_prices_file::String = "", sp_primal_file::String = "",
                        sp_cap_file::String = "", use_primal_warmstart::Bool = true)
    n_ts = admm_data["nTimesteps"]
    n_rd = admm_data["nReprDays"]
    n_yr = admm_data["nYears"]
    shp = (n_ts, n_rd, n_yr)

    # If ADMM is run with multiple scenario years while SP is saved for one year,
    # replicate the single-year SP block across all scenario years.
    function _expand_sp_rows(df::DataFrame, label::String)
        n_expected = n_ts * n_rd * n_yr
        n_one_year = n_ts * n_rd
        if nrow(df) == n_expected
            return df, false
        elseif n_yr > 1 && nrow(df) == n_one_year
            @info "$label has one-year rows ($(nrow(df))); replicating across $n_yr scenario years for warm-start."
            return vcat([copy(df) for _ in 1:n_yr]...), true
        else
            return df, false
        end
    end

    # Store references so save_results and post-processing can access agent/market info.
    results["agents"] = agents
    results["markets"] = Dict("elec" => elec_market, "H2" => H2_market, "elec_GC" => elec_GC_market, "H2_GC" => H2_GC_market, "EP" => EP_market)

    # results["λ"] — Per-market list of 3D price arrays, one per ADMM iteration.
    # The list grows by push! in ADMM.jl each iteration. The first element
    # seeds iteration-0 prices for the very first agent solves.
    #
    # Warm-start strategy:
    #   If a social-planner Market_Prices.csv exists, load its HOURLY prices
    #   as the initial λ. This gives each (jh,jd,jy) slot the true equilibrium
    #   price — dramatically speeding convergence for hours whose prices differ
    #   from the market-wide mean (e.g. peak hours at 483 vs. off-peak at 3).
    #   Falls back to uniform scalar fill when the file is missing or incompatible.
    sp_loaded = false
    if !isempty(sp_prices_file) && isfile(sp_prices_file)
        try
            sp_df = CSV.read(sp_prices_file, DataFrame)
            sp_df, _ = _expand_sp_rows(sp_df, "Social planner prices file")
            n_total = n_ts * n_rd * n_yr
            if nrow(sp_df) == n_total
                # Use SP prices as-is when warm-starting. Do NOT clamp H2_GC to 0 here:
                # clamping would change the equilibrium and cause agents to produce wrong
                # quantities → large imbalance → cascading price updates (e.g. negative elec).
                # The H2_GC floor is applied in ADMM.jl after each price update; for iter 1
                # we trust the SP solution.
                results["λ"] = Dict(
                    "elec"    => [reshape(Float64.(sp_df.Elec_Price),    shp)],
                    "H2"      => [reshape(Float64.(sp_df.H2_Price),     shp)],
                    "elec_GC" => [reshape(Float64.(sp_df.Elec_GC_Price), shp)],
                    "H2_GC"   => [reshape(Float64.(sp_df.H2_GC_Price), shp)],
                    "EP"      => [reshape(Float64.(sp_df.EP_Price),     shp)],
                )
                sp_loaded = true
            else
                @warn "Social planner prices file has $(nrow(sp_df)) rows, expected $n_total. " *
                      "Falling back to scalar warm-start."
            end
        catch e
            @warn "Could not read social planner prices ($sp_prices_file): $e. " *
                  "Falling back to scalar warm-start."
        end
    end

    if !sp_loaded
        results["λ"] = Dict(
            "elec"    => [fill(elec_market["initial_price"], shp...)],
            "H2"      => [fill(H2_market["initial_price"], shp...)],
            "elec_GC" => [fill(elec_GC_market["initial_price"], shp...)],
            "H2_GC"   => [fill(H2_GC_market["initial_price"], shp...)],
            "EP"      => [fill(EP_market["initial_price"], shp...)],
        )
    end

    # Per-agent quantity buffers: empty lists for ALL agents (even non-participants).
    # Only agents that actually participate in a given market will have 3D arrays
    # pushed (by ADMM_subroutine); non-participants keep empty lists. Initialising
    # all agents avoids key-missing errors in generic post-processing loops.
    # Key = agent ID (String), value = list of 3D arrays (one per ADMM iteration).
    #
    # Warm-start: If SP_Primal_Quantities.csv exists (and we loaded SP prices),
    # pre-populate with SP solution so iteration 1 has g_bar = SP (prev = SP, imb = 0).
    # Agents then solve with (λ=SP, g_bar=SP) and reproduce SP → zero imbalance → immediate convergence.
    results["g"]       = Dict(m => [] for m in agents[:all])   # Electricity net position (MW; + = sell, − = buy)
    results["h2"]      = Dict(m => [] for m in agents[:all])   # Hydrogen net position (MW_H2)
    results["elec_GC"] = Dict(m => [] for m in agents[:all])   # Electricity GC net position (MW_GC)
    results["H2_GC"]   = Dict(m => [] for m in agents[:all])   # Hydrogen GC net position (MW_GC)
    results["EP"]      = Dict(m => [] for m in agents[:all])   # End-product net position (MW_EP)

    primal_loaded = false
    if sp_loaded && use_primal_warmstart && !isempty(sp_primal_file) && isfile(sp_primal_file)
        try
            sp_primal = CSV.read(sp_primal_file, DataFrame)
            n_total = n_ts * n_rd * n_yr
            if nrow(sp_primal) == n_total
                # CSV rows are (jy, jd, jh) order. Build 3D [jh, jd, jy] arrays.
                for m in agents[:all]
                    g_arr = zeros(n_ts, n_rd, n_yr)
                    h2_arr = zeros(n_ts, n_rd, n_yr)
                    gc_arr = zeros(n_ts, n_rd, n_yr)
                    h2gc_arr = zeros(n_ts, n_rd, n_yr)
                    ep_arr = zeros(n_ts, n_rd, n_yr)
                    member_src = if haskey(agents, :merged_members) &&
                                    haskey(agents[:merged_members], m)
                        agents[:merged_members][m]
                    else
                        [m]
                    end
                    _fill_sp_primal_agent!(g_arr, h2_arr, gc_arr, h2gc_arr, ep_arr,
                        sp_primal, member_src, n_ts, n_rd, n_yr)
                    push!(results["g"][m], g_arr)
                    push!(results["h2"][m], h2_arr)
                    push!(results["elec_GC"][m], gc_arr)
                    push!(results["H2_GC"][m], h2gc_arr)
                    push!(results["EP"][m], ep_arr)
                end
                primal_loaded = true
                # Sanity check: elec and H2-GC markets should clear at warm-start
                elec_sum = sum(sum(results["g"][m][end]) for m in agents[:elec_market])
                if abs(elec_sum) > 0.01
                    @warn "SP primal warm-start: elec market sum = $elec_sum (expected ≈0). " *
                          "Index/ordering or g_net mapping may be wrong."
                end
                if haskey(agents, :H2_GC_market)
                    h2gc_sum = sum(sum(results["H2_GC"][m][end]) for m in agents[:H2_GC_market])
                    if abs(h2gc_sum) > 0.01
                        @warn "SP primal warm-start: H2-GC market sum = $h2gc_sum (expected ≈0). " *
                              "Merged-agent external GC exposure may be misaligned."
                    end
                end
            elseif n_yr > 1 && nrow(sp_primal) == n_ts * n_rd
                @warn "SP primal file has one-year rows ($(nrow(sp_primal))) but ADMM is running $n_yr years. " *
                      "Skipping primal warm-start to avoid injecting year-mismatched consensus targets."
            else
                @warn "SP primal file has $(nrow(sp_primal)) rows, expected $n_total. Skipping primal warm-start."
            end
        catch e
            @warn "Could not read SP primal quantities ($sp_primal_file): $e. Skipping primal warm-start."
        end
    end

    # Per-agent capacity and investment history for green agents (one vector per ADMM iteration).
    # These are populated only for agents that actually own these variables (VRES, electrolyzer, green offtaker).
    results["Cap_VRES"]        = Dict(m => [] for m in agents[:all])   # VRES capacity per year (MW)
    results["Inv_VRES"]        = Dict(m => [] for m in agents[:all])   # VRES investment per year (MW)
    results["Cap_Elec_H2"]     = Dict(m => [] for m in agents[:all])   # Electrolyzer elec capacity per year (MW)
    results["Inv_Elec_H2"]     = Dict(m => [] for m in agents[:all])   # Electrolyzer elec investment per year (MW)
    results["Cap_EP_Green"]    = Dict(m => [] for m in agents[:all])   # Green offtaker EP capacity per year (MW)
    results["Inv_EP_Green"]    = Dict(m => [] for m in agents[:all])   # Green offtaker EP investment per year (MW)
    results["Cap_Merged"]      = Dict(m => [] for m in get(agents, :merged, String[]))
    results["Inv_Merged"]      = Dict(m => [] for m in get(agents, :merged, String[]))
    results["Merged_z_flow"]   = Dict(m => [] for m in get(agents, :merged, String[]))
    results["Cap_Merged_slots"]  = Dict{String, Vector{String}}()
    results["Cap_Merged_floors"] = Dict{String, Vector{Float64}}()

    # ADMM["ρ"] — Per-MARKET list of scalar penalty weights, one entry per
    # ADMM iteration. Updated by update_rho! (which may increase/decrease ρ
    # based on the ratio of primal to dual residuals). The first element =
    # rho_initial.
    #
    # NOTE: The capacity penalty `ρ_cap` is NO LONGER stored here. After the
    # capacity-ADMM refactor, capacity uses a per-agent equality split
    # (x_cap_m = z_cap_m) with its own per-agent state in ADMM["Capacity"]
    # below. See docs/TECHNICAL.md (ADMM).
    rho_cap_init = get(admm_data, "rho_cap_initial", 0.1)
    ADMM["ρ"] = Dict(
        "elec"    => [elec_market["rho_initial"]],
        "H2"      => [H2_market["rho_initial"]],
        "elec_GC" => [elec_GC_market["rho_initial"]],
        "H2_GC"   => [H2_GC_market["rho_initial"]],
        "EP"      => [EP_market["rho_initial"]],
    )

    # Full 3D imbalance tensor per iteration: sum of all agents' net positions in
    # each market (for EP: minus D_EP). Used in ADMM_subroutine to compute g_bar
    # (consensus centre) for the next iteration. Any[] typed for flexibility:
    # each element is a full 3D array whose exact numeric type may vary.
    ADMM["Imbalances"] = Dict(
        "elec"    => Any[],
        "H2"      => Any[],
        "elec_GC" => Any[],
        "H2_GC"   => Any[],
        "EP"      => Any[],
    )

    # Scalar summary statistics per market per iteration, written to CSV for
    # diagnostics and convergence monitoring:
    #   PriceHistory  — mean(λ) across all (jh,jd,jy) entries each iteration.
    #   ImbalanceMean — mean(|imbalance|) across all entries each iteration.
    ADMM["PriceHistory"] = Dict(
        "elec"    => Float64[],
        "H2"      => Float64[],
        "elec_GC" => Float64[],
        "H2_GC"   => Float64[],
        "EP"      => Float64[],
    )
    ADMM["ImbalanceMean"] = Dict(
        "elec"    => Float64[],
        "H2"      => Float64[],
        "elec_GC" => Float64[],
        "H2_GC"   => Float64[],
        "EP"      => Float64[],
    )

    # Primal and Dual residuals per market per iteration:
    #   Primal = L2 norm of market imbalance (how far supply ≠ demand).
    #   Dual   = L2 norm of the change in consensus variable g_bar between
    #            consecutive iterations (how much the "agreed" allocation shifted).
    # Convergence is checked using Boyd-style absolute + relative tolerances
    # (see ADMM.jl and docs/TECHNICAL.md, ADMM).
    ADMM["Residuals"] = Dict(
        "Primal" => Dict("elec" => Float64[], "H2" => Float64[], "elec_GC" => Float64[], "H2_GC" => Float64[], "EP" => Float64[], "cap" => Float64[]),
        "Dual"   => Dict("elec" => Float64[], "H2" => Float64[], "elec_GC" => Float64[], "H2_GC" => Float64[], "EP" => Float64[], "cap" => Float64[]),
    )

    # Best (smallest) primal/dual residual seen so far per market; used by
    # update_rho! to implement hysteresis and freeze ρ once the algorithm has
    # entered a near-solution region.
    # Capacity best-residuals are per-agent and live in ADMM["Capacity"].
    ADMM["BestResidual"] = Dict(
        "Primal" => Dict("elec" => Inf, "H2" => Inf, "elec_GC" => Inf, "H2_GC" => Inf, "EP" => Inf),
        "Dual"   => Dict("elec" => Inf, "H2" => Inf, "elec_GC" => Inf, "H2_GC" => Inf, "EP" => Inf),
    )

    # Per-MARKET flag indicating that ρ has been frozen permanently; once set
    # to true, update_rho! stops adapting ρ for that market and ADMM behaves
    # like fixed-ρ ADMM in the local neighbourhood of the solution.
    # Capacity freeze flags live per-agent in ADMM["Capacity"]["ρ_frozen"].
    ADMM["ρ_frozen"] = Dict(
        "elec" => false,
        "H2" => false,
        "elec_GC" => false,
        "H2_GC" => false,
        "EP" => false,
    )

    # Short history of residual metrics R = rp + rd per market; update_rho!
    # uses this to decide whether increasing ρ has been beneficial over the
    # recent window, and skips harmful increases that would worsen residuals.
    # Capacity R-histories live per-agent in ADMM["Capacity"]["R_hist"].
    ADMM["R_hist"] = Dict(
        "elec"    => Float64[],
        "H2"      => Float64[],
        "elec_GC" => Float64[],
        "H2_GC"   => Float64[],
        "EP"      => Float64[],
    )

    # ResidualScale: reference magnitude for primal and dual residuals used in
    # the Boyd-style absolute + relative stopping criteria. These are set from
    # the first non-zero residual observed per market and kept fixed for the
    # rest of the run. The "cap" entries here are AGGREGATE scales used for
    # the one-line convergence summary; per-agent scales live inside
    # ADMM["Capacity"].
    ADMM["ResidualScale"] = Dict(
        "Primal" => Dict("elec" => 0.0, "H2" => 0.0, "elec_GC" => 0.0, "H2_GC" => 0.0, "EP" => 0.0, "cap" => 0.0),
        "Dual"   => Dict("elec" => 0.0, "H2" => 0.0, "elec_GC" => 0.0, "H2_GC" => 0.0, "EP" => 0.0, "cap" => 0.0),
    )

    # --------------------------------------------------------------------
    # ADMM["Capacity"] — Per-agent capacity consensus state (equality split)
    #
    # Capacity consensus is solved as a proper ADMM equality split per
    # capacity-owning agent m and year y:
    #
    #     x_{m,y} = cap variable in agent m's model (cap_VRES, cap_H2_y, cap_EP_y)
    #     z_{m,y} = capacity target derived from flow consensus
    #     λ_{m,y} = Lagrange multiplier for x_{m,y} = z_{m,y}
    #     ρ_m    = per-agent penalty (each agent has its own controller)
    #
    # Agent objective adds:
    #     Σ_y [ λ_{m,y}·(x_{m,y} - z_{m,y}) + (ρ_m/2)·(x_{m,y} - z_{m,y})² ].
    #
    # Per-agent residuals (Boyd):
    #     r_m^k = ||x_m^k - z_m^k||_2
    #     s_m^k = ||ρ_m·(z_m^k - z_m^{k-1})||_2
    #
    # WHY per-agent: a single global ρ_cap was the previous bottleneck — one
    # stiff agent type (e.g. electrolyzer) would push the global controller
    # into a regime that destabilised the others. Per-agent ρ_m lets each
    # capacity type adapt independently, which is structurally closer to the
    # ADMM theory.
    #
    # WHY include λ_cap (vs. a pure quadratic penalty): without the linear
    # dual term, ADMM has no exact mechanism to drive x → z; only the
    # quadratic penalty pushes them together. Once ρ_cap saturates, CAPEX
    # dominates and x ≠ z is the local optimum. The dual ascent
    # λ ← λ + ρ·(x - z) restores the missing first-order mechanism.
    # See docs/TECHNICAL.md (ADMM tolerances).
    # --------------------------------------------------------------------
    cap_agents = get(agents, :cap_agents, String[])
    ADMM["Capacity"] = Dict{String, Any}(
        # Ordered list of cap-owning agents (preserves CSV column order)
        "agents" => copy(cap_agents),
        # Per-agent penalty ρ_m history (one push per iteration; first entry = rho_cap_initial)
        "ρ" => Dict{String, Vector{Float64}}(m => [rho_cap_init] for m in cap_agents),
        # Per-agent dual λ_m history; each entry is a length-nYears vector
        "λ" => Dict{String, Vector{Vector{Float64}}}(
            m => [zeros(haskey(get(agents, :merged_members, Dict()), m) ?
                         length(agents[:merged_members][m]) : 1)]
            for m in cap_agents),
        # Per-agent auxiliary z_m history; filled by ADMM_subroutine each iter
        "z" => Dict{String, Vector{Vector{Float64}}}(m => Vector{Float64}[] for m in cap_agents),
        # Per-agent residuals (one scalar per iteration). Primal = ||x_m-z_m||,
        # Dual = ||ρ_m·(z_m^k - z_m^{k-1})||. Initial dual is Inf on iter 1.
        "Primal" => Dict{String, Vector{Float64}}(m => Float64[] for m in cap_agents),
        "Dual"   => Dict{String, Vector{Float64}}(m => Float64[] for m in cap_agents),
        # Hysteresis anchors for the per-agent ρ controller
        "BestPrimal" => Dict{String, Float64}(m => Inf for m in cap_agents),
        "BestDual"   => Dict{String, Float64}(m => Inf for m in cap_agents),
        # Recent residual-sum window used by the controller's monotonicity guards
        "R_hist" => Dict{String, Vector{Float64}}(m => Float64[] for m in cap_agents),
        # Per-agent freeze flag: once close to convergence ρ_m is locked
        "ρ_frozen" => Dict{String, Bool}(m => false for m in cap_agents),
        # Per-agent residual scales (initialised from first non-zero observation)
        "ResidualScale_Primal" => Dict{String, Float64}(m => 0.0 for m in cap_agents),
        "ResidualScale_Dual"   => Dict{String, Float64}(m => 0.0 for m in cap_agents),
    )

    # Capacity ADMM state warm-start from social-planner capacities:
    # if SP capacities are available, preload z_cap history with the SP vector
    # so the very first ADMM iteration can start from a capacity target that is
    # already economically consistent with the SP solution.
    #
    # WHY this choice:
    # set_start_value(cap) initializes variable starts, but ADMM penalties act
    # on (x - z). If z starts far from SP while x starts at SP, iter-1 can
    # immediately create artificial capacity mismatch. Preloading z with SP
    # aligns the split state at startup.
    cap_state_warm_loaded = false
    if !isempty(sp_cap_file) && isfile(sp_cap_file) && !isempty(cap_agents)
        try
            cap_df = CSV.read(sp_cap_file, DataFrame)
            # Expected columns from save_social_planner_results.jl:
            # AgentID, jy, cap
            if all(sym -> hasproperty(cap_df, sym), (:AgentID, :jy, :cap))
                n_cap_z_loaded = 0
                for m in cap_agents
                    if haskey(agents, :merged_members) && haskey(agents[:merged_members], m)
                        member_ids = agents[:merged_members][m]
                        n_cap = length(member_ids)
                        z_vec = merged_cap_z_from_sp(cap_df, member_ids, n_cap)
                        _cap_z_push!(ADMM["Capacity"]["z"][m], z_vec)
                        ADMM["Capacity"]["λ"][m] = [zeros(length(z_vec))]
                        n_cap_z_loaded += 1
                        continue
                    end
                    rows = cap_df[cap_df.AgentID .== m, :]
                    z_init = _sp_cap_scalar(rows)
                    z_init === nothing && continue
                    _cap_z_push!(ADMM["Capacity"]["z"][m], z_init)
                    n_cap_z_loaded += 1
                    # λ history is already seeded as [[0.0]] in ADMM["Capacity"]["λ"].
                end
                cap_state_warm_loaded = n_cap_z_loaded > 0
            end
        catch e
            @warn "Could not read SP capacities for capacity-state warm-start ($sp_cap_file): $e"
        end
    end

    # Absolute and relative tolerances for the ADMM stopping rule:
    #   ε_abs: per-slot flow-market absolute tolerance (MW), from epsilon_abs
    #          if present, otherwise epsilon. me_contracts uses the same epsilon;
    #          contract links use epsilon_cap (scalar MW, no √n).
    #   ε_cap: scalar MW tolerance for capacity splits (not √n scaled).
    #   ε_rel: relative tolerance (dimensionless), optional; defaults to 0.
    # Combined per Boyd et al. as:
    #   ε_pri_flow = ε_abs * sqrt(n) + ε_rel * Scale_primal
    #   ε_pri_cap  = ε_cap + ε_rel * Scale_primal_m
    # where n is the number of time slots and Scale_* are the entries of
    # ResidualScale above. See docs/TECHNICAL.md (ADMM tolerances).
    eps_abs = get(admm_data, "epsilon_abs", get(admm_data, "epsilon", 1.0))
    eps_rel = get(admm_data, "epsilon_rel", 0.0)
    # Capacity is a scalar MW split, not a flow tensor: use a dedicated MW
    # tolerance (default 1.0) rather than reusing the per-slot flow ε_abs.
    eps_cap = get(admm_data, "epsilon_cap", 5.0)
    ADMM["EpsilonAbs"] = eps_abs
    ADMM["EpsilonRel"] = eps_rel
    ADMM["EpsilonCap"] = eps_cap

    # Legacy per-market convergence tolerances (kept for diagnostics; the
    # actual stopping rule uses EpsilonAbs/EpsilonRel and ResidualScale).
    # Capacity tolerance is now computed per agent inside ADMM.jl and not
    # stored here; the per-agent stopping rule replaces the single "cap" entry.
    base_tol = get(admm_data, "epsilon", 1.0)
    ADMM["Tolerance"] = Dict(
        "elec"    => base_tol,
        "elec_GC" => base_tol,
        "EP"      => base_tol,
        "H2"      => base_tol,
        "H2_GC"   => base_tol,
    )
    ADMM["n_iter"]   = 0     # Iteration counter; incremented each ADMM loop
    ADMM["walltime"] = 0.0   # Cumulative wall-clock time (seconds); measured in ADMM.jl
    # Number of market-clearing slots used by Boyd-style scaled tolerances.
    ADMM["n_slots"]  = n_ts * n_rd * n_yr

    # Warm-start flags for consolidated logging (read by market_exposure.jl)
    results["warmstart"] = Dict("λ" => sp_loaded, "primal" => primal_loaded, "capacity_state" => cap_state_warm_loaded)
    return results, ADMM
end
