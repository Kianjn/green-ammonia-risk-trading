# ==============================================================================
# save_results_contracts.jl — Write results for me_contracts.jl (ME+C)
# ==============================================================================
#
# PURPOSE:
#   Writes everything market_exposure writes (via save_results: ADMM_Convergence,
#   ADMM_Diagnostics, 5× *_Market_History, Agent_Summary, Agent_Objectives_Per_
#   Timestep, Offtaker_GC_Diagnostics, H2_Producer_Diagnostics, Market_Prices,
#   risk and cost metrics) into me_contracts_results/, and adds the bilateral
#   contract outputs:
#
#   PPAs.csv / HPAs.csv         one row per link: seller, buyer(s), contracted
#                               capacity (seller offer, buyer demand, cleared
#                               mid-point), fixed price K, expected contracted
#                               energy, expected fixed leg, expected spot value
#                               and expected net payment to the seller, share of
#                               the seller's plant under contract, ADMM residuals.
#   Contract_Cashflows_Per_Year.csv
#                               per link and scenario year: contracted energy,
#                               fixed leg K·E, floating leg Σ λ·q, and the net
#                               settlement paid by the buyer to the seller.
#   Green_Agents_Detail.csv     long format: one row per (agent, contract, role)
#                               with plant capacity, contracted MW and share.
#   Contract_History.csv        per iteration and link: K, C_sell, Σ C_buy,
#                               imbalance, primal/dual residual, ρ.
#
#   Agent_Summary.csv is extended with two columns:
#     Contract_Cash_Expected      expected net contract cash RECEIVED (€/yr;
#                                 negative = net payer)
#     Objective_Value_incl_Contracts = Objective_Value − Contract_Cash_Expected
#   Objective_Value itself stays the pool-only economic objective so it is
#   directly comparable with market_exposure_results/Agent_Summary.csv. Because
#   contracts are pure transfers, Σ_agents Contract_Cash_Expected = 0.
#
#   ADMM_Convergence.csv and ADMM_Diagnostics.csv gain `ppa_<v>_*` / `hpa_<h>_*`
#   columns; Market_Prices.csv gains constant `PPA_Price_<v>` / `HPA_Price_<h>`
#   columns (a fixed price has no time profile).
#
# ==============================================================================

import Printf: @sprintf, @printf

if !isdefined(@__MODULE__, :save_results)
    include(joinpath(@__DIR__, "save_results.jl"))
end
if !isdefined(@__MODULE__, :contract_cashflows)
    include(joinpath(@__DIR__, "contract_settlement.jl"))
end

"""Last element of a history vector, or `default` when empty."""
_last_or(v::AbstractVector, default) = isempty(v) ? default : v[end]

"""Align a history vector to n rows (truncate, or pad with `pad`)."""
function _align_hist(v::AbstractVector, n::Int; pad = NaN)
    length(v) >= n && return Float64.(v[1:n])
    return vcat(Float64.(v), fill(Float64(pad), n - length(v)))
end

"""Per-link contract report for one contract market (`:ppa` or `:hpa`)."""
function _contract_link_report(kind::Symbol, id::String, mdict::Dict, results::Dict, ADMM_state::Dict)
    ck       = kind == :ppa ? "ppa" : "hpa"
    K_key    = kind == :ppa ? "K_ppa" : "K_hpa"
    sell_key = kind == :ppa ? "C_ppa_sell" : "C_hpa_sell"
    buy_key  = kind == :ppa ? "C_ppa_buy" : "C_hpa_buy"
    cap_key  = kind == :ppa ? "Cap_VRES" : "Cap_Elec_H2"

    ms = mdict[id]
    p  = ms.ext[:parameters]
    JH, JD, JY = ms.ext[:sets][:JH], ms.ext[:sets][:JD], ms.ext[:sets][:JY]
    W, P = p[:W], p[:P]
    prof = kind == :ppa ? Float64.(ms.ext[:timeseries][:AF]) :
                          flat_contract_profile((length(JH), length(JD), length(JY)))
    λb = kind == :ppa ? ppa_bundle_spot(results["λ"]["elec"][end], results["λ"]["elec_GC"][end]) :
                        hpa_bundle_spot(results["λ"]["H2"][end], results["λ"]["H2_GC"][end])

    K   = _last_or(results[K_key][id], 0.0)
    C_s = _last_or(results[sell_key][id], 0.0)
    buyers = sort(collect(keys(results[buy_key])))
    C_b_each = Dict(b => _last_or(results[buy_key][b][id], 0.0) for b in buyers)
    C_b = sum(values(C_b_each); init = 0.0)
    C_mid = 0.5 * (C_s + C_b)          # equal at convergence; mid-point otherwise
    cf = contract_cashflows(W, K, λb, prof, C_mid, JH, JD, JY)
    cap_hist = get(get(results, cap_key, Dict()), id, [])
    cap = isempty(cap_hist) ? NaN : cap_hist[end][1]
    S = ADMM_state[ck]
    return (
        id = id, kind = kind, K = K, C_sell = C_s, C_buy = C_b, C_mid = C_mid,
        buyers = buyers, C_b_each = C_b_each, cap = cap, JY = JY, P = P, cf = cf,
        fair_K = contract_fair_price(W, P, λb, prof, JH, JD, JY),
        A = get(S["A"], id, NaN),
        primal = _last_or(S["Primal"][id], NaN), dual = _last_or(S["Dual"][id], NaN),
        rho = _last_or(S["ρ"][id], NaN),
    )
end

function save_results_contracts!(mdict::Dict, elec_market::Dict, H2_market::Dict,
                                 elec_GC_market::Dict, H2_GC_market::Dict,
                                 ppa_market::Dict, hpa_market::Dict,
                                 ADMM_state::Dict, results::Dict, agents::Dict;
                                 results_dir::String = joinpath(@__DIR__, "..", "me_contracts_results"),
                                 case_label::String = "me_contracts")
    isdir(results_dir) || mkdir(results_dir)

    # ── 1. Everything the ME case writes (summary printed later, see below) ──
    cost_metrics = save_results(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market,
                                ADMM_state, results, agents;
                                results_dir = results_dir, case_label = case_label,
                                ppa_market = ppa_market, hpa_market = hpa_market,
                                print_summary = false)

    n_it = length(ADMM_state["Imbalances"]["elec"])
    ppa_ids = get(ppa_market, "ppa_vres", String[])
    hpa_ids = get(hpa_market, "hpa_h2", String[])

    # ── 2. Contract columns in the ADMM convergence / diagnostics tables ──
    conv_path = joinpath(results_dir, "ADMM_Convergence.csv")
    diag_path = joinpath(results_dir, "ADMM_Diagnostics.csv")
    conv_df = CSV.read(conv_path, DataFrame)
    diag_df = CSV.read(diag_path, DataFrame)
    hist_rows = NamedTuple[]
    for (ck, ids, K_key, sell_key, buy_key) in (("ppa", ppa_ids, "K_ppa", "C_ppa_sell", "C_ppa_buy"),
                                                ("hpa", hpa_ids, "K_hpa", "C_hpa_sell", "C_hpa_buy"))
        S = ADMM_state[ck]
        for id in ids
            rp = _align_hist(S["Primal"][id], n_it)
            rd = _align_hist(S["Dual"][id], n_it)
            ρ  = _align_hist(S["ρ"][id], n_it)
            imb = _align_hist(S["Imbalance"][id], n_it)
            # K has n_it+1 entries (initial + one per iteration): report "K after iteration i".
            Kh = results[K_key][id]
            Kh = length(Kh) == n_it + 1 ? Kh[2:end] : _align_hist(Kh, n_it)
            Cs = _align_hist(results[sell_key][id], n_it)
            Cb = zeros(n_it)
            for b in keys(results[buy_key])
                Cb .+= _align_hist(results[buy_key][b][id], n_it; pad = 0.0)
            end
            conv_df[!, Symbol("$(ck)_$(id)_primal")] = rp
            conv_df[!, Symbol("$(ck)_$(id)_dual")]   = rd
            diag_df[!, Symbol("$(ck)_$(id)_rho")]    = ρ
            diag_df[!, Symbol("$(ck)_$(id)_price")]  = Float64.(Kh)
            diag_df[!, Symbol("$(ck)_$(id)_imb")]    = imb
            diag_df[!, Symbol("$(ck)_$(id)_C_sell")] = Cs
            diag_df[!, Symbol("$(ck)_$(id)_C_buy")]  = Cb
            for i in 1:n_it
                push!(hist_rows, (iter = i, Contract = uppercase(ck), Seller = id, K = Kh[i],
                                  C_sell = Cs[i], C_buy = Cb[i], imbalance = imb[i],
                                  primal_res = rp[i], dual_res = rd[i], rho = ρ[i]))
            end
        end
    end
    CSV.write(conv_path, conv_df)
    CSV.write(diag_path, diag_df)
    isempty(hist_rows) || CSV.write(joinpath(results_dir, "Contract_History.csv"), DataFrame(hist_rows))

    # ── 3. Per-link reports ────────────────────────────────────────────────
    reports = NamedTuple[]
    for v in ppa_ids
        push!(reports, _contract_link_report(:ppa, v, mdict, results, ADMM_state))
    end
    for h in hpa_ids
        push!(reports, _contract_link_report(:hpa, h, mdict, results, ADMM_state))
    end

    function _link_rows(kind::Symbol)
        rows = NamedTuple[]
        for r in reports
            r.kind == kind || continue
            E_exp   = contract_expected(r.cf.energy, r.P, r.JY)
            fix_exp = contract_expected(r.cf.fixed_leg, r.P, r.JY)
            flo_exp = contract_expected(r.cf.floating_leg, r.P, r.JY)
            net_exp = contract_expected(r.cf.net_to_seller, r.P, r.JY)
            push!(rows, (
                Seller                    = r.id,
                Buyers                    = join(r.buyers, ";"),
                Contract_Type             = kind == :ppa ? "PPA pay-as-produced, fixed price" :
                                                           "HPA baseload, fixed price",
                C_sell_MW                 = r.C_sell,
                C_buy_MW                  = r.C_buy,
                C_contract_MW             = r.C_mid,
                Seller_Capacity_MW        = r.cap,
                Contract_Share_of_Plant   = (isfinite(r.cap) && r.cap > 1e-9) ? r.C_mid / r.cap : NaN,
                K_price                   = r.K,
                Fair_Price_at_Final_Spot  = r.fair_K,
                Risk_Premium              = r.K - r.fair_K,
                Expected_MWh_per_MW       = r.A,
                Expected_Energy_MWh       = E_exp,
                Expected_Fixed_Leg        = fix_exp,
                Expected_Spot_Value       = flo_exp,
                Expected_Net_To_Seller    = net_exp,
                Primal_Residual_MW        = r.primal,
                Dual_Residual             = r.dual,
                Rho                       = r.rho,
            ))
        end
        return rows
    end
    ppa_rows = _link_rows(:ppa)
    hpa_rows = _link_rows(:hpa)
    CSV.write(joinpath(results_dir, "PPAs.csv"), isempty(ppa_rows) ? DataFrame(Seller = String[]) : DataFrame(ppa_rows))
    CSV.write(joinpath(results_dir, "HPAs.csv"), isempty(hpa_rows) ? DataFrame(Seller = String[]) : DataFrame(hpa_rows))

    # ── 4. Per-scenario cash flows ────────────────────────────────────────
    cf_rows = NamedTuple[]
    for r in reports
        for jy in r.JY
            push!(cf_rows, (
                Contract      = r.kind == :ppa ? "PPA" : "HPA",
                Seller        = r.id,
                Buyers        = join(r.buyers, ";"),
                jy            = jy,
                P             = r.P[jy],
                C_MW          = r.C_mid,
                K_price       = r.K,
                Energy_MWh    = r.cf.energy[jy],
                Fixed_Leg     = r.cf.fixed_leg[jy],
                Floating_Leg  = r.cf.floating_leg[jy],
                Net_To_Seller = r.cf.net_to_seller[jy],
            ))
        end
    end
    isempty(cf_rows) || CSV.write(joinpath(results_dir, "Contract_Cashflows_Per_Year.csv"), DataFrame(cf_rows))

    # ── 5. Green_Agents_Detail.csv (long format) and per-agent expected contract cash ──
    cash = Dict{String, Float64}(m => 0.0 for m in agents[:all])   # expected net cash RECEIVED
    detail_rows = NamedTuple[]
    function _agent_cap(id::String)
        for key in ("Cap_VRES", "Cap_Elec_H2", "Cap_EP_Green")
            hist = get(get(results, key, Dict()), id, [])
            isempty(hist) || return hist[end][1]
        end
        return NaN
    end
    for r in reports
        ctype = r.kind == :ppa ? "PPA" : "HPA"
        net_exp_total = contract_expected(r.cf.net_to_seller, r.P, r.JY)
        E_exp_total   = contract_expected(r.cf.energy, r.P, r.JY)
        cash[r.id] = get(cash, r.id, 0.0) + net_exp_total
        cap_s = _agent_cap(r.id)
        push!(detail_rows, (
            AgentID = r.id, Type = String(get(mdict[r.id].ext[:parameters], :Type, "")),
            Contract = ctype, Role = "seller", Counterparty = join(r.buyers, ";"),
            Capacity_MW = cap_s, Contract_MW = r.C_sell,
            Contract_Share = (isfinite(cap_s) && cap_s > 1e-9) ? r.C_sell / cap_s : NaN,
            Expected_Contract_Energy_MWh = E_exp_total, K_price = r.K,
            Expected_Net_Cash_Received = net_exp_total,
        ))
        for b in r.buyers
            C_b = r.C_b_each[b]
            share_b = r.C_buy > 1e-9 ? C_b / r.C_buy : 0.0
            cash_b = -share_b * net_exp_total
            cash[b] = get(cash, b, 0.0) + cash_b
            cap_b = _agent_cap(b)
            # Contracted MW relative to the buyer's plant: MW_e intake for the
            # electrolyzer (cap/η), MW_H2 intake for the offtaker (cap/α).
            pb = mdict[b].ext[:parameters]
            intake = r.kind == :ppa ? cap_b / max(get(pb, :η_elec_H2, 1.0), 1e-9) :
                                      cap_b / max(get(pb, :Alpha, 1.0), 1e-9)
            push!(detail_rows, (
                AgentID = b, Type = String(get(pb, :Type, "")),
                Contract = ctype, Role = "buyer", Counterparty = r.id,
                Capacity_MW = cap_b, Contract_MW = C_b,
                Contract_Share = (isfinite(intake) && intake > 1e-9) ? C_b / intake : NaN,
                Expected_Contract_Energy_MWh = share_b * E_exp_total, K_price = r.K,
                Expected_Net_Cash_Received = cash_b,
            ))
        end
    end
    isempty(detail_rows) || CSV.write(joinpath(results_dir, "Green_Agents_Detail.csv"), DataFrame(detail_rows))

    # ── 6. Agent_Summary.csv: add contract cash columns ───────────────────
    sum_path = joinpath(results_dir, "Agent_Summary.csv")
    if isfile(sum_path)
        sdf = CSV.read(sum_path, DataFrame)
        sdf[!, :Contract_Cash_Expected] = [get(cash, String(a), 0.0) for a in sdf.AgentID]
        sdf[!, :Objective_Value_incl_Contracts] = sdf.Objective_Value .- sdf.Contract_Cash_Expected
        CSV.write(sum_path, sdf)
    end

    # ── 7. Market_Prices.csv: constant contract-price columns ─────────────
    mp_path = joinpath(results_dir, "Market_Prices.csv")
    if isfile(mp_path)
        mp = CSV.read(mp_path, DataFrame)
        for r in reports
            col = r.kind == :ppa ? "PPA_Price_$(r.id)" : "HPA_Price_$(r.id)"
            mp[!, Symbol(col)] = fill(r.K, nrow(mp))
        end
        CSV.write(mp_path, mp)
    end

    # ── 8. Console summary (also tee'd to the run log) ────────────────────
    with_run_summary_log(results_dir) do
        print_cost_metrics_summary!(cost_metrics)
        print_admm_run_summary!(ADMM_state, results, agents; results_dir = results_dir,
                                ppa_market = ppa_market, hpa_market = hpa_market)
    end
    return nothing
end
