# ==============================================================================
# ADMM.jl — Main ADMM coordination loop
# ==============================================================================

if !isdefined(@__MODULE__, :repeat_last_agent_quantities!)
    include(joinpath(@__DIR__, "cap_admm_helpers.jl"))
end

# PURPOSE:
#   Runs the ADMM loop until convergence or max_iter. Each iteration:
#   1. For each agent: ADMM_subroutine! updates λ, g_bar, ρ on the model, solves
#      the agent, and appends the solution quantities to results.
#   2. Compute market imbalances (sum of net positions; for EP, subtract D_EP).
#   3. Append mean imbalance per market to ImbalanceMean (for CSV).
#   4. Primal residual per market = L2 norm of imbalance.
#   5. Dual residual per market = norm of (ρ * change in consensus deviation);
#      first iteration uses Inf (no previous deviation).
#   6. Track normalized residual merit (best-so-far checkpoint).
#   7. Price update: λ_new = λ_old - η * ρ * imbalance (elementwise), with
#      scale-aware η damping and per-market step-scale adaptation.
#   8. Append mean price per market to PriceHistory (for CSV).
#   9. update_rho! adapts ρ per market (Boyd rule with hysteresis/freeze).
#   10. Anti-stall logic: if merit worsens for a long window, restart from best
#       checkpoint and continue with smaller dual steps.
#   11. Installed VRES/H2/EP capacity is a private QP variable (Alessio):
#       ADMM coordinates energy, not an x = peak(g/AF) split.
#   12. If all flow primal and dual residuals are below tolerance, set convergence=1.
#   Progress bar is left clean (no per-iteration print).
#
# ARGUMENTS:
#   results, ADMM_state — Filled by define_results!; updated in place.
#   elec_market, ... — Market dicts (nAgents, D_EP for EP).
#   mdict — Dict of JuMP models (one per agent).
#   agents — Dict of agent lists (all, elec_market, H2_market, ...).
#   data — Contains General (nTimesteps, nReprDays, nYears) and ADMM (max_iter).
#   TO — TimerOutput for profiling each section.
#
# ==============================================================================

function ADMM!(results::Dict, ADMM_state::Dict, elec_market::Dict, H2_market::Dict,
               elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
               mdict::Dict, agents::Dict, data::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    # Shape of every 3D quantity / price tensor: (hours × representative days × years).
    # All market imbalances, prices λ, and consensus targets g_bar share this shape.
    shp = (n_ts, n_rd, n_yr)
    max_iter = data["ADMM"]["max_iter"]
    convergence = 0
    iterations = ProgressBar(1:max_iter)
    # market_keys drives the merit, rollback and aggregate display loops.
    # "cap" is included here for the aggregate one-line summary, but the
    # actual stopping rule is checked PER AGENT (see within_tol_cap below).
    market_keys = ("elec", "H2", "elec_GC", "H2_GC", "EP", "cap")

    # Capacity-owning agents (VRES, GreenProducer, GreenOfftaker). Their
    # capacity consensus uses a per-agent equality split with state stored in
    # ADMM_state["Capacity"]. See docs/TECHNICAL.md (ADMM: capacity is a private variable).
    cap_agents = get(agents, :cap_agents, String[])

    # Expose horizon sizes to update_rho! (used by both market and per-agent
    # capacity controllers to compute Boyd-style absolute tolerances).
    ADMM_state["n_slots"] = n_ts * n_rd * n_yr
    ADMM_state["n_yr"]    = n_yr
    # Optional controller knobs (read by the per-agent cap controller).
    ADMM_state["rho_cap_max"]        = get(get(data, "ADMM", Dict()), "rho_cap_max", 30.0)
    ADMM_state["rho_cap_inc_factor"] = get(get(data, "ADMM", Dict()), "rho_cap_inc_factor", 1.05)

    # Per-market dual-update step scaling (in addition to rho and eta damping).
    # This is adapted online from residual progress to prevent late-iteration
    # oscillations in tightly coupled markets.
    η_scale = Dict("elec" => 1.0, "H2" => 1.0, "elec_GC" => 1.0, "H2_GC" => 1.0, "EP" => 1.0)

    # Best-iterate checkpoint (normalized merit score). If progress stalls for a
    # long window and current merit is much worse than best, we apply a
    # rate-limited blended rollback toward this checkpoint and continue with
    # smaller steps (instead of hard rewinds that can create cycles).
    best_iter = 0
    best_score = Inf
    stall_count = 0
    restart_patience = 40
    restart_factor = 1.15
    # Reform: disable checkpoint/recovery steering and rely on plain ADMM updates.
    enable_recovery_steering = false
    rollback_count = 0
    # Keep this large: in strongly coupled systems, we may need several
    # best-region re-entry attempts over long runs.
    max_rollbacks = 999
    rollback_cooldown = 80
    last_rollback_iter = -rollback_cooldown
    rollback_blend = 0.35
    # Coordinated rho exploration factors around global-best region.
    explore_idx = 0
    explore_factors = (0.90, 1.00, 1.10)
    best_λ = Dict{String,Array{Float64,3}}()
    best_ρ = Dict{String,Float64}()
    # Per-agent best ρ_cap for the rollback blend; populated alongside best_ρ.
    best_ρ_cap = Dict{String,Float64}()
    # Per-agent best λ_cap snapshot for global best-region re-entry.
    best_λ_cap = Dict{String,Vector{Float64}}()
    # ------------------------------------------------------------------
    # Per-market / per-agent basin guards
    #
    # Global best_score tracks only the worst market. To avoid the pattern
    # where some markets enter a good basin and then drift away while waiting
    # for others, we keep market-local and capacity-agent-local "best basin"
    # anchors and pull states back when they drift too far.
    #
    # WHY this choice:
    # - Addresses the observed "best iter ~100 then diverge for 100+ iters"
    #   behaviour by preserving local progress.
    # - Implements "stay in best area while others catch up" in a controlled,
    #   blended way (no hard resets).
    # ------------------------------------------------------------------
    flow_markets = ("elec", "H2", "elec_GC", "H2_GC", "EP")
    market_best_merit = Dict{String,Float64}(m => Inf for m in flow_markets)
    market_best_λ = Dict{String,Array{Float64,3}}(m => copy(results["λ"][m][end]) for m in flow_markets)
    market_best_ρ = Dict{String,Float64}(m => ADMM_state["ρ"][m][end] for m in flow_markets)
    market_rho_hold_until = Dict{String,Int}(m => 0 for m in flow_markets)
    # Persistent-drift detector per market; used by rescue mode.
    market_bad_streak = Dict{String,Int}(m => 0 for m in flow_markets)
    cap_best_merit = Dict{String,Float64}(m => Inf for m in cap_agents)
    cap_best_λ = Dict{String,Vector{Float64}}(m => copy(ADMM_state["Capacity"]["λ"][m][end]) for m in cap_agents)
    cap_rho_hold_until = Dict{String,Int}(m => 0 for m in cap_agents)
    # Basin guards should only activate once a market/agent has reached a
    # reasonably good neighborhood. Activating too early (far from convergence)
    # can lock the run near a poor early iterate.
    # Experimental local guard/re-entry logic is disabled by default because
    # it can over-steer tightly coupled runs. Keep the value above max_iter
    # so those branches are inactive unless explicitly re-enabled in code.
    guard_min_iter = max_iter + 1

    function _market_eps(key::String, n_slots::Int)
        eps_rel = ADMM_state["EpsilonRel"]
        # Capacity consensus is a scalar MW split, not a (hour,day,year) flow
        # tensor. Using flow ε_abs·√n_slots here would either (a) make cap far
        # too loose if √n_slots is applied, or (b) make cap far too tight if
        # the per-slot flow ε_abs is reused unscaled. Dedicated ε_cap (MW).
        if key == "cap"
            eps_abs = Float64(get(ADMM_state, "EpsilonCap", ADMM_state["EpsilonAbs"]))
            sqrt_n = 1.0
        else
            eps_abs = ADMM_state["EpsilonAbs"]
            sqrt_n = sqrt(max(1, n_slots))
        end
        sp = max(ADMM_state["ResidualScale"]["Primal"][key], 1.0)
        sd = max(ADMM_state["ResidualScale"]["Dual"][key], 1.0)
        eps_pr = eps_abs * sqrt_n + eps_rel * sp
        eps_du = eps_abs * sqrt_n + eps_rel * sd
        return eps_pr, eps_du
    end

    # Log initial mean prices (before any iteration) so diagnostics show warm-start.
    for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        push!(ADMM_state["PriceHistory"][mkt], mean(results["λ"][mkt][end]))
    end

    for iter in iterations
        # Early exit once convergence has been achieved (flagged at the end of
        # the previous iteration). Breaking here—rather than using a while-loop—
        # keeps the ProgressBar display frozen at the converged iteration number
        # instead of jumping ahead to max_iter.
        convergence == 1 && break

        # Solve all agents (single-threaded for deterministic order and result indexing)
        for m in agents[:all]
            Base.invokelatest(ADMM_subroutine!, m, data, results, ADMM_state, elec_market, H2_market,
                            elec_GC_market, H2_GC_market, EP_market, mdict[m], agents, TO)
        end

        # ------------------------------------------------------------------
        # Imbalances (full 3D tensors, one entry per (jh, jd, jy))
        # For each market, sum the net positions of ALL participants.
        # Sign convention: generators are positive, consumers negative.
        # A positive imbalance means excess supply; negative means excess demand.
        # For the EP market, we additionally subtract D_EP (the fixed, inelastic
        # end-product demand) so that imbalance = supply − demand.
        # ------------------------------------------------------------------
        @timeit TO "Compute imbalances" begin
            imb_elec = sum(results["g"][m][end] for m in agents[:elec_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec"], imb_elec)

            imb_H2 = sum(results["h2"][m][end] for m in agents[:H2_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["H2"], imb_H2)

            imb_elec_GC = sum(results["elec_GC"][m][end] for m in agents[:elec_GC_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec_GC"], imb_elec_GC)

            imb_H2_GC = sum(results["H2_GC"][m][end] for m in agents[:H2_GC_market]; init=zeros(shp...))
            # H2_GC is a proper hourly market (no annual aggregation).
            # Offtakers have temporal flexibility to buy GCs whenever prices
            # are low, accumulating toward their annual mandate internally.
            push!(ADMM_state["Imbalances"]["H2_GC"], imb_H2_GC)

            # EP is the only market with fixed inelastic demand D_EP;
            # subtract it so imbalance = Σ agent_supply − D_EP.
            imb_EP = sum(results["EP"][m][end] for m in agents[:EP_market]; init=zeros(shp...)) .- EP_market["D_EP"]
            push!(ADMM_state["Imbalances"]["EP"], imb_EP)
        end

        # ------------------------------------------------------------------
        # Imbalance means — scalar diagnostic (mean over all (jh,jd,jy)
        # entries of the 3D imbalance tensor). These scalars are appended
        # per iteration for CSV logging / plotting convergence curves.
        # ------------------------------------------------------------------
        @timeit TO "Imbalance means" begin
            push!(ADMM_state["ImbalanceMean"]["elec"],    mean(ADMM_state["Imbalances"]["elec"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2"],      mean(ADMM_state["Imbalances"]["H2"][end]))
            push!(ADMM_state["ImbalanceMean"]["elec_GC"], mean(ADMM_state["Imbalances"]["elec_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2_GC"],   mean(ADMM_state["Imbalances"]["H2_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["EP"],      mean(ADMM_state["Imbalances"]["EP"][end]))
        end

        # ------------------------------------------------------------------
        # Primal residuals — L2 norm of the imbalance vector: √Σ imbalance².
        # Measures how far each market is from clearing (supply = demand).
        # Smaller values ⇒ closer to feasibility. This is one half of the
        # standard ADMM stopping criterion (Boyd et al., 2011).
        # ------------------------------------------------------------------
        @timeit TO "Primal residuals" begin
            rp_elec    = sqrt(sum(ADMM_state["Imbalances"]["elec"][end].^2))
            rp_H2      = sqrt(sum(ADMM_state["Imbalances"]["H2"][end].^2))
            rp_elec_GC = sqrt(sum(ADMM_state["Imbalances"]["elec_GC"][end].^2))
            rp_H2_GC   = sqrt(sum(ADMM_state["Imbalances"]["H2_GC"][end].^2))
            rp_EP      = sqrt(sum(ADMM_state["Imbalances"]["EP"][end].^2))
            push!(ADMM_state["Residuals"]["Primal"]["elec"],    rp_elec)
            push!(ADMM_state["Residuals"]["Primal"]["H2"],      rp_H2)
            push!(ADMM_state["Residuals"]["Primal"]["elec_GC"], rp_elec_GC)
            push!(ADMM_state["Residuals"]["Primal"]["H2_GC"],   rp_H2_GC)
            push!(ADMM_state["Residuals"]["Primal"]["EP"],      rp_EP)

            # Initialise per-market primal residual scales on first non-zero
            # observation; used in Boyd-style absolute + relative tolerances.
            scales_pr = ADMM_state["ResidualScale"]["Primal"]
            if scales_pr["elec"] == 0.0 && rp_elec > 0.0
                scales_pr["elec"] = rp_elec
            end
            if scales_pr["H2"] == 0.0 && rp_H2 > 0.0
                scales_pr["H2"] = rp_H2
            end
            if scales_pr["elec_GC"] == 0.0 && rp_elec_GC > 0.0
                scales_pr["elec_GC"] = rp_elec_GC
            end
            if scales_pr["H2_GC"] == 0.0 && rp_H2_GC > 0.0
                scales_pr["H2_GC"] = rp_H2_GC
            end
            if scales_pr["EP"] == 0.0 && rp_EP > 0.0
                scales_pr["EP"] = rp_EP
            end

            # ------------------------------------------------------------
            # Capacity primal residuals — per-agent split (and aggregate)
            #
            # For each capacity-owning agent m we measure
            #     r_m^k = || x_m^k - z_m^k ||_2  (over years)
            # where x_m is the agent's actual cap variable (cap_VRES,
            # cap_H2_y, cap_EP_y) and z_m is the auxiliary target that
            # ADMM_subroutine pushed into ADMM_state["Capacity"]["z"][m]
            # this iteration. The per-agent residuals drive the per-agent
            # ρ controller and the stopping rule; the aggregate (Σ_m r_m²)^½
            # is kept only as a one-line summary.
            # ------------------------------------------------------------
            cap_state = ADMM_state["Capacity"]
            rp_cap_sq = 0.0
            for m in cap_agents
                cap_vec = _agent_cap_vec_from_results(m, results)
                # z_m^k pushed by ADMM_subroutine this iteration
                z_hist = cap_state["z"][m]
                z_vec  = isempty(z_hist) ? cap_vec : z_hist[end]
                local_r = 0.0
                if !isempty(cap_vec)
                    z_use = _aligned_cap_target(z_vec, length(cap_vec))
                    local_r = _cap_primal_residual(cap_vec, z_use)
                end
                push!(cap_state["Primal"][m], local_r)
                # Initialise per-agent primal scale from first non-zero observation
                if cap_state["ResidualScale_Primal"][m] == 0.0 && local_r > 0.0
                    cap_state["ResidualScale_Primal"][m] = local_r
                end
                rp_cap_sq += local_r^2
            end
            rp_cap = sqrt(rp_cap_sq)
            push!(ADMM_state["Residuals"]["Primal"]["cap"], rp_cap)
            if ADMM_state["ResidualScale"]["Primal"]["cap"] == 0.0 && rp_cap > 0.0
                ADMM_state["ResidualScale"]["Primal"]["cap"] = rp_cap
            end
        end

        # ------------------------------------------------------------------
        # Capacity dual ASCENT (per-agent λ_cap update)
        #
        # Standard ADMM equality-split dual update:
        #     λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k)   per year y
        # This is the first-order force that drives x → z in the limit; the
        # quadratic penalty alone (without λ) leaves a persistent residual
        # whenever the CAPEX gradient exceeds the quadratic term — which is
        # exactly the failure mode the old single-rho design suffered from.
        #
        # Push λ_m^k to history so the NEXT iteration's subroutine reads it.
        # ρ_m^{k-1} is the per-agent penalty BEFORE update_rho! runs at the
        # end of this iteration; update_rho! produces ρ_m^k for the next.
        # ------------------------------------------------------------------
        @timeit TO "Capacity dual update" begin
            cap_state = ADMM_state["Capacity"]
            for m in cap_agents
                cap_vec = _agent_cap_vec_from_results(m, results)
                z_hist = cap_state["z"][m]
                z_vec  = isempty(z_hist) ? cap_vec : z_hist[end]
                ρ_m    = cap_state["ρ"][m][end]
                λ_prev = cap_state["λ"][m][end]
                if isempty(cap_vec)
                    push!(cap_state["λ"][m], copy(λ_prev))
                else
                    z_use = _aligned_cap_target(z_vec, length(cap_vec))
                    λ_new = _cap_dual_ascent(λ_prev, ρ_m, cap_vec, z_use)
                    push!(cap_state["λ"][m], λ_new)
                end
            end
        end

        # ------------------------------------------------------------------
        # Dual residuals — measure how much each agent's position changed
        # relative to its consensus target between successive iterations.
        #
        # For each agent m in a market with n participants, compute:
        #   diff_m = (q_m^k − q̄^k) − (q_m^{k−1} − q̄^{k−1})
        # where q̄^k = (1/(n+1)) · Σ_m q_m^k is the consensus average.
        # The (n+1) denominator comes from the sharing ADMM formulation,
        # which introduces one "market copy" alongside the n agent copies,
        # so the consensus variable is the mean of (n+1) copies.
        #
        # The dual residual = √ Σ_m (ρ · diff_m)² (L2 norm over all
        # agents and all (jh,jd,jy) timesteps). Smaller ⇒ agents are
        # settling on consistent positions. Together with the primal
        # residual, this forms the ADMM stopping criterion (Boyd et al.).
        #
        # On the first iteration there is no previous iterate to compare
        # against, so we set dual residuals to Inf (cannot be satisfied,
        # forcing at least two iterations before convergence is declared).
        # ------------------------------------------------------------------
        @timeit TO "Dual residuals" begin
            nE  = elec_market["nAgents"]
            nH  = H2_market["nAgents"]
            nEG = elec_GC_market["nAgents"]
            nHG = H2_GC_market["nAgents"]
            nEP = EP_market["nAgents"]
            if iter > 1
                dual_elec = 0.0
                for m in agents[:elec_market]
                    # Change in (own_quantity − consensus_average) from iter k−1 to k
                    diff = (results["g"][m][end] .- sum(results["g"][mstar][end] for mstar in agents[:elec_market]) ./ (nE + 1)) .-
                           (results["g"][m][end-1] .- sum(results["g"][mstar][end-1] for mstar in agents[:elec_market]) ./ (nE + 1))
                    dual_elec += sum((ADMM_state["ρ"]["elec"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["elec"], sqrt(dual_elec))
                dual_H2 = 0.0
                for m in agents[:H2_market]
                    diff = (results["h2"][m][end] .- sum(results["h2"][mstar][end] for mstar in agents[:H2_market]) ./ (nH + 1)) .-
                           (results["h2"][m][end-1] .- sum(results["h2"][mstar][end-1] for mstar in agents[:H2_market]) ./ (nH + 1))
                    dual_H2 += sum((ADMM_state["ρ"]["H2"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["H2"], sqrt(dual_H2))
                dual_elec_GC = 0.0
                for m in agents[:elec_GC_market]
                    diff = (results["elec_GC"][m][end] .- sum(results["elec_GC"][mstar][end] for mstar in agents[:elec_GC_market]) ./ (nEG + 1)) .-
                           (results["elec_GC"][m][end-1] .- sum(results["elec_GC"][mstar][end-1] for mstar in agents[:elec_GC_market]) ./ (nEG + 1))
                    dual_elec_GC += sum((ADMM_state["ρ"]["elec_GC"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["elec_GC"], sqrt(dual_elec_GC))
                dual_H2_GC = 0.0
                for m in agents[:H2_GC_market]
                    diff = (results["H2_GC"][m][end] .- sum(results["H2_GC"][mstar][end] for mstar in agents[:H2_GC_market]) ./ (nHG + 1)) .-
                           (results["H2_GC"][m][end-1] .- sum(results["H2_GC"][mstar][end-1] for mstar in agents[:H2_GC_market]) ./ (nHG + 1))
                    dual_H2_GC += sum((ADMM_state["ρ"]["H2_GC"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["H2_GC"], sqrt(dual_H2_GC))
                dual_EP = 0.0
                for m in agents[:EP_market]
                    diff = (results["EP"][m][end] .- sum(results["EP"][mstar][end] for mstar in agents[:EP_market]) ./ (nEP + 1)) .-
                           (results["EP"][m][end-1] .- sum(results["EP"][mstar][end-1] for mstar in agents[:EP_market]) ./ (nEP + 1))
                    dual_EP += sum((ADMM_state["ρ"]["EP"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["EP"], sqrt(dual_EP))

                # Initialise per-market dual residual scales on first non-zero
                # observation; used in Boyd-style absolute + relative tolerances.
                scales_du = ADMM_state["ResidualScale"]["Dual"]
                rd_elec    = ADMM_state["Residuals"]["Dual"]["elec"][end]
                rd_H2      = ADMM_state["Residuals"]["Dual"]["H2"][end]
                rd_elec_GC = ADMM_state["Residuals"]["Dual"]["elec_GC"][end]
                rd_H2_GC   = ADMM_state["Residuals"]["Dual"]["H2_GC"][end]
                rd_EP      = ADMM_state["Residuals"]["Dual"]["EP"][end]
                if scales_du["elec"] == 0.0 && rd_elec < Inf
                    scales_du["elec"] = rd_elec
                end
                if scales_du["H2"] == 0.0 && rd_H2 < Inf
                    scales_du["H2"] = rd_H2
                end
                if scales_du["elec_GC"] == 0.0 && rd_elec_GC < Inf
                    scales_du["elec_GC"] = rd_elec_GC
                end
                if scales_du["H2_GC"] == 0.0 && rd_H2_GC < Inf
                    scales_du["H2_GC"] = rd_H2_GC
                end
                if scales_du["EP"] == 0.0 && rd_EP < Inf
                    scales_du["EP"] = rd_EP
                end
                # ----------------------------------------------------------
                # Capacity dual residuals — per-agent split
                #
                # For the equality split x_m = z_m, the ADMM-correct dual
                # residual is the change in the AUXILIARY z (not in x):
                #     s_m^k = || ρ_m^{k-1} · (z_m^k - z_m^{k-1}) ||_2
                # (Boyd et al. 2011, Eq. 3.12).
                #
                # Using Δz (not Δx) is essential: if x has converged but z
                # is still drifting, a Δx-based residual would falsely
                # declare convergence; Δz captures the true ADMM dual
                # progress.
                # ----------------------------------------------------------
                cap_state = ADMM_state["Capacity"]
                dual_cap_sq = 0.0
                for m in cap_agents
                    z_hist = cap_state["z"][m]
                    ρ_m    = cap_state["ρ"][m][end]
                    if length(z_hist) >= 2
                        z_new = _cap_scalar(z_hist[end])
                        z_old = _cap_scalar(z_hist[end - 1])
                        local_s = abs(ρ_m * (z_new - z_old))
                        push!(cap_state["Dual"][m], local_s)
                        if cap_state["ResidualScale_Dual"][m] == 0.0 && local_s > 0.0 && isfinite(local_s)
                            cap_state["ResidualScale_Dual"][m] = local_s
                        end
                        dual_cap_sq += local_s^2
                    else
                        # First iteration: z^{k-1} does not exist → dual undefined
                        push!(cap_state["Dual"][m], Inf)
                    end
                end
                dual_cap = isempty(cap_agents) || any(isinf, [cap_state["Dual"][m][end] for m in cap_agents]) ? Inf : sqrt(dual_cap_sq)
                push!(ADMM_state["Residuals"]["Dual"]["cap"], dual_cap)
                if ADMM_state["ResidualScale"]["Dual"]["cap"] == 0.0 && isfinite(dual_cap) && dual_cap > 0.0
                    ADMM_state["ResidualScale"]["Dual"]["cap"] = dual_cap
                end
            else
                # First iteration: no previous iterate exists, so dual residuals
                # are undefined. Set to Inf to prevent premature convergence.
                push!(ADMM_state["Residuals"]["Dual"]["elec"],    Inf)
                push!(ADMM_state["Residuals"]["Dual"]["H2"],      Inf)
                push!(ADMM_state["Residuals"]["Dual"]["elec_GC"], Inf)
                push!(ADMM_state["Residuals"]["Dual"]["H2_GC"],  Inf)
                push!(ADMM_state["Residuals"]["Dual"]["EP"],      Inf)
                push!(ADMM_state["Residuals"]["Dual"]["cap"],     Inf)
                # Keep per-agent capacity dual history aligned with iterations
                # (iter 1 has undefined dual because z^{k-1} does not exist).
                cap_state = ADMM_state["Capacity"]
                for m in cap_agents
                    push!(cap_state["Dual"][m], Inf)
                end
            end
        end

        # ------------------------------------------------------------------
        # Merit tracking (normalized residuals) and anti-stall restart
        # ------------------------------------------------------------------
        n_slots = n_ts * n_rd * n_yr
        merit = Dict{String,Float64}()
        for key in market_keys
            rp = ADMM_state["Residuals"]["Primal"][key][end]
            rd = ADMM_state["Residuals"]["Dual"][key][end]
            eps_pr, eps_du = _market_eps(key, n_slots)
            m = max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))
            merit[key] = isfinite(m) ? m : 1e12
        end
        score = maximum(values(merit))
        # Track local best basin per flow market (independent of global score).
        for mkt in flow_markets
            if merit[mkt] < market_best_merit[mkt]
                market_best_merit[mkt] = merit[mkt]
                market_best_λ[mkt] = copy(results["λ"][mkt][end])
                market_best_ρ[mkt] = ADMM_state["ρ"][mkt][end]
            end
        end
        # Track local best basin per capacity-owning agent.
        cap_state_merit = ADMM_state["Capacity"]
        for m in cap_agents
            rp_m = cap_state_merit["Primal"][m][end]
            rd_m = cap_state_merit["Dual"][m][end]
            eps_pr_m, eps_du_m = _cap_boyd_eps(ADMM_state, m)
            mm = max(rp_m / max(eps_pr_m, 1e-9), isfinite(rd_m) ? rd_m / max(eps_du_m, 1e-9) : 1e12)
            if isfinite(mm) && mm < cap_best_merit[m]
                cap_best_merit[m] = mm
                cap_best_λ[m] = copy(cap_state_merit["λ"][m][end])
                best_ρ_cap[m] = cap_state_merit["ρ"][m][end]
            end
        end
        if score < best_score
            best_score = score
            best_iter = iter
            stall_count = 0
            for mkt in flow_markets
                best_λ[mkt] = copy(results["λ"][mkt][end])
                best_ρ[mkt] = ADMM_state["ρ"][mkt][end]
            end
            # Capacity is per-agent: snapshot every agent's current ρ_m.
            for m in cap_agents
                best_ρ_cap[m] = ADMM_state["Capacity"]["ρ"][m][end]
                best_λ_cap[m] = copy(ADMM_state["Capacity"]["λ"][m][end])
            end
        else
            stall_count += 1
        end

        # Adapt per-market step scales from one-step merit movement.
        if iter > 1
            for mkt in flow_markets
                rp_prev = ADMM_state["Residuals"]["Primal"][mkt][end-1]
                rd_prev = ADMM_state["Residuals"]["Dual"][mkt][end-1]
                eps_pr_prev, eps_du_prev = _market_eps(mkt, n_slots)
                merit_prev = max(rp_prev / max(eps_pr_prev, 1e-9), rd_prev / max(eps_du_prev, 1e-9))
                merit_now = merit[mkt]
                if merit_now > 1.02 * merit_prev
                    η_scale[mkt] = max(0.15, 0.85 * η_scale[mkt])
                elseif merit_now < 0.98 * merit_prev
                    η_scale[mkt] = min(1.0, 1.03 * η_scale[mkt])
                end
            end
        end

        # ------------------------------------------------------------------
        # Price (dual variable) update: λ_new = λ_old − η · ρ · imbalance.
        # Standard ADMM dual variable update with a per-market, iteration-
        # dependent step-size factor η ∈ (0,1], derived from current residuals.
        #   • When supply > demand (positive imbalance) → price decreases.
        #   • When demand > supply (negative imbalance) → price increases.
        # Far from convergence, η = 1 so we recover the usual update. Near
        # convergence (small primal/dual residuals), η is reduced to damp
        # oscillations in tightly coupled markets while ρ remains fixed.
        # ------------------------------------------------------------------
        @timeit TO "Update prices" begin
            # Scale-aware damping of dual updates (λ update):
            #   λ^{k+1} = λ^k - η_k * ρ_k * imbalance
            # with η_k in [η_min, 1]. We compute η_k from the ratio of the
            # market's current residual level to its Boyd-style tolerance scale
            # (same scale used in the convergence check). This makes damping
            # horizon-robust (e.g. 1 year vs 10 years) and prevents thin markets
            # from oscillating with full-step updates near the stopping region.
            η_min = 0.25
            eps_abs = ADMM_state["EpsilonAbs"]
            eps_rel = ADMM_state["EpsilonRel"]
            n_slots = max(1, get(ADMM_state, "n_slots", 1))
            sqrt_n = sqrt(n_slots)
            scale_pr = ADMM_state["ResidualScale"]["Primal"]
            scale_du = ADMM_state["ResidualScale"]["Dual"]
            for mkt in flow_markets
                rp = ADMM_state["Residuals"]["Primal"][mkt][end]
                rd = ADMM_state["Residuals"]["Dual"][mkt][end]
                base = max(rp, rd)
                sp = max(scale_pr[mkt], 1.0)
                sd = max(scale_du[mkt], 1.0)
                eps_pr = eps_abs * sqrt_n + eps_rel * sp
                eps_du = eps_abs * sqrt_n + eps_rel * sd
                eps_m = max(eps_pr, eps_du)
                # Full step when comfortably above tolerance; smoothly damp near it.
                # NOTE: Damping must decrease the step when residuals are SMALL,
                # not when they are large. Reversed damping stalls progress.
                η_raw = base >= 1.5 * eps_m ? 1.0 : max(η_min, base / max(1.5 * eps_m, 1e-9))
                η = η_scale[mkt] * η_raw
                push!(results["λ"][mkt],
                      results["λ"][mkt][end] .- η .* ADMM_state["ρ"][mkt][end] .* ADMM_state["Imbalances"][mkt][end])
            end
            # H2_GC price floor: the electrolyzer VOLUNTARILY issues green
            # certificates — at price < 0 no rational producer would issue,
            # so supply is identically 0 and the equilibrium price is ≥ 0.
            # Without this projection, negative prices attract unbounded
            # demand from offtakers (who profit from buying at negative
            # prices), creating a persistent limit-cycle oscillation that
            # ADMM cannot resolve.  Clamping to [0,∞) is the standard
            # "projected ADMM" technique and preserves convergence theory.
            results["λ"]["H2_GC"][end] .= max.(results["λ"]["H2_GC"][end], 0.0)
        end

        # Store scalar price diagnostics (mean price per market and iteration)
        @timeit TO "Price means" begin
            push!(ADMM_state["PriceHistory"]["elec"],    mean(results["λ"]["elec"][end]))
            push!(ADMM_state["PriceHistory"]["H2"],      mean(results["λ"]["H2"][end]))
            push!(ADMM_state["PriceHistory"]["elec_GC"], mean(results["λ"]["elec_GC"][end]))
            push!(ADMM_state["PriceHistory"]["H2_GC"],   mean(results["λ"]["H2_GC"][end]))
            push!(ADMM_state["PriceHistory"]["EP"],      mean(results["λ"]["EP"][end]))
        end

        @timeit TO "Update ρ" begin
            update_rho!(ADMM_state, iter)
        end
        # Temporary rho hold windows (guard against immediate post-update
        # overreaction after a basin rollback).
        for mkt in flow_markets
            if iter <= market_rho_hold_until[mkt] && length(ADMM_state["ρ"][mkt]) >= 2
                ADMM_state["ρ"][mkt][end] = ADMM_state["ρ"][mkt][end-1]
            end
        end
        cap_state_hold = ADMM_state["Capacity"]
        for m in cap_agents
            if iter <= cap_rho_hold_until[m] && length(cap_state_hold["ρ"][m]) >= 2
                cap_state_hold["ρ"][m][end] = cap_state_hold["ρ"][m][end-1]
            end
        end

        # ------------------------------------------------------------------
        # Per-market rescue mode (persistent drift recovery)
        #
        # If a market's normalized merit stays significantly above its local
        # best for many consecutive iterations, local smooth updates can be too
        # weak (observed in H2/EP drifts). Apply a decisive but bounded jump:
        #   1) boost rho for that market,
        #   2) pull λ back toward market-local best,
        #   3) unfreeze rho so adaptation can continue from the new state.
        # ------------------------------------------------------------------
        if enable_recovery_steering
            rescue_trigger = 1.20
            rescue_streak_iters = 10
            rescue_rho_boost = 1.35
            rescue_blend = 0.60
            for mkt in flow_markets
                b = market_best_merit[mkt]
                if b < Inf && merit[mkt] > rescue_trigger * b
                    market_bad_streak[mkt] += 1
                else
                    market_bad_streak[mkt] = 0
                end
                if market_bad_streak[mkt] >= rescue_streak_iters
                    # rho jump (bounded by market-specific maxima)
                    ρ_cur = ADMM_state["ρ"][mkt][end]
                    ρ_max = mkt in ("elec", "elec_GC") ? 5_000.0 : 100.0
                    ADMM_state["ρ"][mkt][end] = min(ρ_max, rescue_rho_boost * ρ_cur)
                    # λ pullback to local-best basin
                    results["λ"][mkt][end] .= (1.0 - rescue_blend) .* results["λ"][mkt][end] .+ rescue_blend .* market_best_λ[mkt]
                    # ensure controller can react after rescue
                    if haskey(ADMM_state["ρ_frozen"], mkt)
                        ADMM_state["ρ_frozen"][mkt] = false
                    end
                    η_scale[mkt] = max(0.20, 0.85 * η_scale[mkt])
                    market_bad_streak[mkt] = 0
                end
            end
        end

        # ------------------------------------------------------------------
        # Local basin guard (per flow market + per capacity agent)
        #
        # If a market/agent drifts sufficiently away from its own best merit
        # basin, blend λ and ρ back toward that local best and hold rho for a
        # few iterations. This is a soft "stay near best area" mechanism.
        # ------------------------------------------------------------------
        guard_trigger = 1.25   # relative drift from local best merit
        guard_blend = 0.15
        guard_hold_iters = 4
        for mkt in flow_markets
            b = market_best_merit[mkt]
            (b < Inf && iter >= guard_min_iter) || continue
            if merit[mkt] > guard_trigger * b
                results["λ"][mkt][end] .= (1.0 - guard_blend) .* results["λ"][mkt][end] .+ guard_blend .* market_best_λ[mkt]
                ADMM_state["ρ"][mkt][end] = (1.0 - guard_blend) * ADMM_state["ρ"][mkt][end] + guard_blend * market_best_ρ[mkt]
                η_scale[mkt] = max(0.10, 0.80 * η_scale[mkt])
                market_rho_hold_until[mkt] = max(market_rho_hold_until[mkt], iter + guard_hold_iters)
            end
        end
        cap_state_guard = ADMM_state["Capacity"]
        for m in cap_agents
            rp_m = cap_state_guard["Primal"][m][end]
            rd_m = cap_state_guard["Dual"][m][end]
            eps_pr_m, eps_du_m = _cap_boyd_eps(ADMM_state, m)
            mm = max(rp_m / max(eps_pr_m, 1e-9), isfinite(rd_m) ? rd_m / max(eps_du_m, 1e-9) : 1e12)
            b = cap_best_merit[m]
            if b < Inf && iter >= guard_min_iter && isfinite(mm) && mm > guard_trigger * b
                cap_state_guard["λ"][m][end] .= (1.0 - guard_blend) .* cap_state_guard["λ"][m][end] .+ guard_blend .* cap_best_λ[m]
                if haskey(best_ρ_cap, m)
                    cap_state_guard["ρ"][m][end] = (1.0 - guard_blend) * cap_state_guard["ρ"][m][end] + guard_blend * best_ρ_cap[m]
                end
                cap_rho_hold_until[m] = max(cap_rho_hold_until[m], iter + guard_hold_iters)
            end
        end

        # Global checkpoint recovery (active):
        # If we drift well above the best score for a sustained window,
        # HARD-restore λ and ρ to the best iterate and continue from there.
        # This is intentionally simple and robust in tightly coupled systems.
        stalled_and_worse = enable_recovery_steering &&
                            stall_count >= restart_patience && score > restart_factor * best_score
        if stalled_and_worse && !isempty(best_λ)
            can_rollback = rollback_count < max_rollbacks &&
                           (iter - last_rollback_iter) >= max(15, Int(0.35 * restart_patience))
            if can_rollback
                α = 1.0
                for mkt in flow_markets
                    results["λ"][mkt][end] .= best_λ[mkt]
                    ADMM_state["ρ"][mkt][end] = best_ρ[mkt]
                    η_scale[mkt] = max(0.20, 0.90 * η_scale[mkt])
                    if haskey(ADMM_state["ρ_frozen"], mkt)
                        ADMM_state["ρ_frozen"][mkt] = false
                    end
                end
                cap_state_rb = ADMM_state["Capacity"]
                for m in cap_agents
                    if haskey(best_ρ_cap, m)
                        cap_state_rb["ρ"][m][end] = best_ρ_cap[m]
                    end
                    if haskey(best_λ_cap, m)
                        cap_state_rb["λ"][m][end] .= best_λ_cap[m]
                    end
                    cap_state_rb["ρ_frozen"][m] = false
                end
                rollback_count += 1
                last_rollback_iter = iter
            end
            stall_count = 0
        end

        # Clean progress bar: show only iteration and max; no extra printing
        set_description(iterations, "")

        # ------------------------------------------------------------------
        # Convergence check: Boyd-style absolute + relative criteria.
        #
        # For each market k we compute primal and dual tolerances:
        #
        #   ε_pri_k  = ε_abs * sqrt(n) + ε_rel * Scale_primal_k
        #   ε_dual_k = ε_abs * sqrt(n) + ε_rel * Scale_dual_k
        #
        # where n is the number of time slots in the horizon and Scale_*_k are
        # fixed reference magnitudes captured from the first non-zero residual
        # (see define_results.jl). This mirrors the stopping rule in Boyd et
        # al. (2011), making the criteria scale-aware and robust across
        # markets with very different quantity ranges.
        #
        # ALL five markets must have BOTH primal AND dual residuals below
        # their respective tolerance. If any single residual exceeds its
        # tolerance, the algorithm continues iterating. This ensures that
        # every market has simultaneously cleared (primal) and that agent
        # positions have stabilised (dual) before we declare convergence.
        # ------------------------------------------------------------------
        eps_abs = ADMM_state["EpsilonAbs"]
        eps_rel = ADMM_state["EpsilonRel"]
        n_slots = n_ts * n_rd * n_yr
        sqrt_n = sqrt(n_slots)
        scale_pr = ADMM_state["ResidualScale"]["Primal"]
        scale_du = ADMM_state["ResidualScale"]["Dual"]

        function within_tol(key::String)
            rp = ADMM_state["Residuals"]["Primal"][key][end]
            rd = ADMM_state["Residuals"]["Dual"][key][end]
            sp = max(scale_pr[key], 1.0)
            sd = max(scale_du[key], 1.0)
            eps_pr = eps_abs * sqrt_n + eps_rel * sp
            eps_du = eps_abs * sqrt_n + eps_rel * sd
            return (rp <= eps_pr) && (rd <= eps_du)
        end

        # Installed capacity is private (Alessio). Stopping is flow-market Boyd only.
        function within_tol_cap()
            return true
        end

        if (within_tol("elec") &&
            within_tol("H2") &&
            within_tol("elec_GC") &&
            within_tol("H2_GC") &&
            within_tol("EP") &&
            within_tol_cap())
            convergence = 1
        end

        ADMM_state["n_iter"] = iter
    end

    println()  # clean line after progress bar
    ADMM_state["converged"] = (convergence == 1)
    if !ADMM_state["converged"]
        @printf("ADMM reached max_iter without convergence (best score %.4f at iter %d).\n",
                best_score, best_iter)
    end

    return nothing
end
