# ==============================================================================
# contract_settlement.jl — Fixed-price bilateral contract terms (PPA / HPA)
# ==============================================================================
#
# PURPOSE:
#   Pure helper functions shared by the contract build/solve/ADMM/save files.
#   They encode the ONE contract type used by me_contracts.jl:
#
#     A bilateral contract between two firms: agreed capacity C (MW) at an
#     agreed fixed price K (€/MWh). The buyer pays the seller K on the
#     contracted energy. That cash flow is revenue in the seller's objective
#     and cost in the buyer's, inside both CVaR terms. No government, no
#     subsidy, and no indexed strike: K does not float with a benchmark.
#
#   * PPA (VRES → electrolyzer): pay-as-produced. The contracted volume in
#     slot (h,d,y) is AF_v[h,d,y]·C, the output of C MW of the seller's plant.
#     This is the standard renewable PPA (fixed €/MWh on the plant's profile).
#   * HPA (electrolyzer → green offtaker): baseload. The contracted volume is
#     C in every slot. This is the standard industrial hydrogen offtake
#     (a firm MW at a fixed €/MWh_H2).
#
#   Physical MWh still clear in the spot pools. On the contracted slice the
#   seller therefore earns K instead of the bundled spot price, and the buyer
#   pays K instead of that spot price:
#
#       λ·(q_phys − q) + K·q   =   λ·q_phys + (K − λ_bundle)·q
#
#   with q = prof·C. The QP uses the right-hand side, because q_phys is already
#   in the pool term and the contract adds one coefficient on the scalar C:
#
#       M_y = Σ_{h,d} W[d,y] · (K − λ_bundle[h,d,y]) · prof[h,d,y]
#       Π_y = M_y · C          paid by the buyer to the seller
#
#   Π_y is the incremental cash relative to selling/buying the slice on the
#   spot. The contractual payment itself is the fixed leg K·Σ W·prof·C
#   (contract_cashflows). The bundle is the green package the contract covers
#   (elec + elec-GC for a PPA; H₂ + H₂-GC for an HPA). Summed over the two
#   parties, Π_y cancels: it is a transfer, not a payment from outside the model.
#
#   Units: prof is dimensionless (AF ∈ [0,1] or 1.0), W in days, C in MW, so
#   W·prof·C is MWh per representative day and Π_y is €/year in scenario y.
#
# ==============================================================================

"""Bundled spot price a PPA is settled against: electricity + electricity GC."""
ppa_bundle_spot(λ_elec::AbstractArray, λ_elec_GC::AbstractArray) = λ_elec .+ λ_elec_GC

"""Bundled spot price an HPA is settled against: hydrogen + hydrogen GC."""
hpa_bundle_spot(λ_H2::AbstractArray, λ_H2_GC::AbstractArray) = λ_H2 .+ λ_H2_GC

"""Flat (baseload) delivery profile with the same shape as the price tensors."""
flat_contract_profile(shp::Tuple) = ones(Float64, shp)

"""
    contract_annual_volume_per_MW(W, P, prof, JH, JD, JY)

Expected contracted energy per MW of contract, A = Σ_y P_y Σ_{h,d} W[d,y]·prof[h,d,y]
(MWh/MW-year). For a baseload HPA this is ≈ 8760; for an as-produced PPA it is
8760 × the expected capacity factor of the contracted plant.

A is used (i) to scale the ADMM contract penalty (ρ/2)·A·(C − C̄)² so that ρ has
the same €/MWh-per-MW meaning as in the spot markets, and (ii) to convert
contract MW into expected MWh in the reports.
"""
function contract_annual_volume_per_MW(W::AbstractMatrix, P::AbstractVector,
                                       prof::AbstractArray, JH, JD, JY)
    A = 0.0
    for jy in JY
        s = 0.0
        for jd in JD, jh in JH
            s += W[jd, jy] * prof[jh, jd, jy]
        end
        A += P[jy] * s
    end
    return A
end

"""
    contract_settlement_coeffs(W, K, λ_bundle, prof, JH, JD, JY)

Per-scenario settlement coefficient M_y = Σ_{h,d} W[d,y]·(K − λ_bundle[h,d,y])·prof[h,d,y]
(€ per MW of contract, per year, in scenario y). The buyer pays M_y·C to the
seller in scenario y. Returns a Dict{Int,Float64} keyed by jy.

Rebuilt every ADMM iteration because K and λ_bundle change.
"""
function contract_settlement_coeffs(W::AbstractMatrix, K::Real, λ_bundle::AbstractArray,
                                    prof::AbstractArray, JH, JD, JY)
    M = Dict{Int, Float64}()
    for jy in JY
        s = 0.0
        for jd in JD, jh in JH
            s += W[jd, jy] * (K - λ_bundle[jh, jd, jy]) * prof[jh, jd, jy]
        end
        M[jy] = s
    end
    return M
end

"""
    contract_fair_price(W, P, λ_bundle, prof, JH, JD, JY)

Risk-neutral fair fixed price: the volume-weighted expected bundled spot price on
the contract profile, K_fair = Σ_y P_y Σ W λ_bundle prof / Σ_y P_y Σ W prof. At this
price the expected settlement Σ_y P_y Π_y is zero, so a risk-neutral agent is
indifferent to the contract. Used as the ADMM warm start for K when
`initial_price: spot` is set in `data.yaml`; at γ = 1 it is already the
equilibrium contract price.
"""
function contract_fair_price(W::AbstractMatrix, P::AbstractVector, λ_bundle::AbstractArray,
                             prof::AbstractArray, JH, JD, JY)
    num = 0.0
    den = 0.0
    for jy in JY
        for jd in JD, jh in JH
            w = P[jy] * W[jd, jy] * prof[jh, jd, jy]
            num += w * λ_bundle[jh, jd, jy]
            den += w
        end
    end
    return den > 1e-12 ? num / den : 0.0
end

"""
    contract_cashflows(W, K, λ_bundle, prof, C, JH, JD, JY)

Per-scenario cash-flow decomposition of one contract at volume C (MW), for
reporting. Returns a NamedTuple of Dict{Int,Float64} keyed by jy:

- `energy`        Σ W·prof·C                       contracted MWh in scenario y
- `fixed_leg`     K·energy                          what the buyer pays at the fixed price
- `floating_leg`  Σ W·λ_bundle·prof·C               spot value of the same volume
- `net_to_seller` fixed_leg − floating_leg          = Π_y (positive: buyer pays seller)
"""
function contract_cashflows(W::AbstractMatrix, K::Real, λ_bundle::AbstractArray,
                            prof::AbstractArray, C::Real, JH, JD, JY)
    energy = Dict{Int, Float64}()
    fixed_leg = Dict{Int, Float64}()
    floating_leg = Dict{Int, Float64}()
    net = Dict{Int, Float64}()
    for jy in JY
        e = 0.0
        f = 0.0
        for jd in JD, jh in JH
            q = W[jd, jy] * prof[jh, jd, jy] * C
            e += q
            f += λ_bundle[jh, jd, jy] * q
        end
        energy[jy] = e
        fixed_leg[jy] = K * e
        floating_leg[jy] = f
        net[jy] = K * e - f
    end
    return (energy = energy, fixed_leg = fixed_leg, floating_leg = floating_leg, net_to_seller = net)
end

"""Probability-weighted expectation of a per-scenario Dict."""
function contract_expected(x::Dict{Int, Float64}, P::AbstractVector, JY)
    return sum(P[jy] * x[jy] for jy in JY; init = 0.0)
end
