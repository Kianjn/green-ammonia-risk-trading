# ==============================================================================
# build_power_agent_contracts.jl — Power agents in the contracts case (ME+C)
# ==============================================================================
#
# PURPOSE:
#   Builds the same physical model as build_power_agent! and, for a VRES agent,
#   adds its side of the bilateral PPA:
#
#     C_ppa  — contracted capacity (MW) sold under a fixed-price, as-produced
#              PPA to the electrolyzer. One scalar per VRES: the same MW are
#              under contract in every hour and every scenario (that is what an
#              "agreed capacity" means).
#
#   The VRES keeps selling all of its output g into the electricity and GC
#   pools exactly as in market_exposure.jl (g ≤ AF·cap_VRES). The electrolyzer
#   pays this VRES the fixed price K_ppa on the as-produced volume AF·C_ppa.
#   Pool revenue on that slice is λ_bundle·AF·C_ppa, so the two together lock
#   C_ppa MW of the plant at K_ppa; the rest of the plant stays on the spot.
#   The QP writes that lock as the transfer the buyer pays,
#   Π_y = Σ W (K_ppa − λ_elec − λ_elec_GC)·AF·C_ppa (contract_settlement.jl).
#   There is no third party: the same Π_y is a cost in the electrolyzer's loss.
#
#   Constraint: C_ppa ≤ cap_VRES — a PPA is written on the seller's own plant,
#   which is what ties contracting to investment (a PPA can underwrite new MW).
#
#   The contract enters the OBJECTIVE only (through the per-scenario loss that
#   feeds CVaR), so it is added in solve_power_agent_contracts! each iteration
#   with the current K and λ. Conventional and Consumer agents are unchanged.
#
# ==============================================================================

function build_power_agent_contracts!(m::String, mod::Model, elec_market::Dict,
                                      elec_GC_market::Dict, ppa_market::Dict)
    build_power_agent!(m, mod, elec_market, elec_GC_market)

    p = mod.ext[:parameters]
    if String(get(p, :Type, "")) == "VRES" && get(p, :in_ppa_market, false)
        cap_VRES = mod.ext[:variables][:cap_VRES]
        # Contracted capacity (MW). Scalar: agreed once, before weather is known.
        C_ppa = mod.ext[:variables][:C_ppa] = @variable(mod, lower_bound = 0, base_name = "C_ppa_$(m)")
        # A PPA covers the output of the seller's own plant.
        mod.ext[:constraints][:ppa_within_plant] = @constraint(mod, C_ppa <= cap_VRES)
    end
    return mod
end
