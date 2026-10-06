# Technical note

This note states what the code computes, which economic object each script solves, and why the calibration and numerical settings are the ones in `Data/data.yaml`. It is the reference for the public release. Agent-level algebra lives in `Source/`; this document is the map.

## 1. Problem

The model is a **price-taking competitive equilibrium** in five coupled commodity markets, with capacity chosen once before uncertainty resolves. It is a partial equilibrium: there is no income feedback into the rest of the economy, and there is no spatial network. Indices $(h,d,y)$ are hour, representative day, and scenario. They are not buses or pipelines.

Each agent treats prices as given and optimises its own technology. Markets clear by equality of supply and demand. At `gamma = 1` the social planner is the welfare dual of that equilibrium (first welfare theorem, under convexity). ADMM is the decentralised **solver**. The quadratic penalties are not part of the economics; they vanish when markets clear.

The same stack is solved under five risk institutions, following the complete / incomplete risk-trading distinction in d'Aertrycke, Ehrenmann, Ralph and Smeers (2018):

| Script | When `gamma < 1` |
|--------|------------------|
| `social_planner.jl` | **Complete** risk trading. One social CVaR on aggregate welfare. Balance duals are risk-adjusted social shadow prices. |
| `green_social_planner.jl` | **Complete inside the green chain.** Solar, wind, electrolyzer, and green offtaker share one coalition CVaR. Residual spot exposure remains. |
| `green_h2_social_planner.jl` | Same institution, limited to electrolyzer and green offtaker. |
| `me_contracts.jl` | **Partial.** Each firm keeps a private CVaR. A fixed-price PPA and a fixed-price HPA move cash between them. |
| `market_exposure.jl` | **None.** Private CVaR only. Full spot exposure. |

At `gamma = 1` the CVaR term drops out. The four decentralised scripts then reproduce the planner, up to ADMM tolerance. At `gamma < 1` a gap between planner and market exposure is the risk institution, not a formulation error. Do not expect quantities or prices to match.

This is not Nash–Cournot and not an EPEC. Inside each agent problem the price is a parameter. No firm sees rivals' quantities, and no first-order condition contains $\partial\lambda/\partial q$. Aggregation into one solar firm and one wind firm is a modelling choice, not market power. In the taxonomy of Gabriel, Conejo, Fuller, Hobbs and Ruiz (2013) the decentralised model is a perfect-competition mixed complementarity problem. The social planner is a separate convex program used as the benchmark, not an upper level wrapped around the market.

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $h,d,y$ | Hour, representative day, scenario. In code: `jh`, `jd`, `jy`. |
| $W_{d,y}$ | Number of calendar days represented by day $d$ in scenario $y$. Sums to 365. |
| $P_y$ | Scenario probability. Default: $1/15$. |
| $g_i^k$ | Net position of agent $i$ in market $k$. Positive is supply, negative is demand. |
| $\lambda_k$ | Price in market $k$ (€ per unit). |
| $\gamma$ | Weight on expected loss versus CVaR. `gamma = 1` is risk-neutral. |
| $\beta$ | CVaR confidence level (Rockafellar–Uryasev). Higher $\beta$ is a thinner, more extreme tail. |
| $\mathrm{cap}$ | Installed capacity (MW). One number, chosen before the scenario is known. |

Units are MWh for electricity, certificate-MWh for guarantees of origin, MWh of hydrogen (LHV), and MWh of ammonia. Ammonia uses LHV $18.6\,\mathrm{MJ/kg}$, so $5.167\,\mathrm{MWh}$ per tonne. All money is euro. A per-hour term is multiplied by $W_{d,y}$ when it is aggregated to a scenario year, so eight representative days stand in for a weighted year of 365 days.

## 3. Markets and agents

| Market | Code key | Sellers | Buyers |
|--------|----------|---------|--------|
| Electricity | `elec` | Solar, wind, CCGT, coal, biomass | Consumer, electrolyzer |
| Electricity certificates | `elec_GC` | Solar, wind | Electrolyzer, certificate demand |
| Hydrogen | `H2` | Electrolyzer | Green offtaker |
| Hydrogen certificates | `H2_GC` | Electrolyzer | Green offtaker, grey offtaker |
| End product (ammonia) | `EP` | Green offtaker, grey offtaker, importer | Fixed demand |

The electrolyzer couples the markets: hydrogen output equals electrical input divided by `SpecificConsumption` (1.5 MWh electricity per MWh hydrogen, about 67% LHV). Hydrogen certificates can be issued only against certified electricity, so the certificate markets are not a free second revenue stream.

The agent set in `data.yaml` is a Netherlands-scale stack:

- **Solar and wind**, with existing capacity as a floor and endogenous investment above it.
- **CCGT, coal, and biomass**, dispatchable, with short-run marginal cost derived from the `Fuel` block.
- **An electricity consumer** with quadratic utility, hence linear inverse demand.
- **A green electrolyzer** and a **green ammonia offtaker** (Haber–Bosch), both with endogenous capacity.
- **A grey ammonia offtaker** on steam-methane reforming, capacity fixed.
- **An importer**, present in the file with capacity zero so the option can be switched on.
- **Certificate demand** with its own linear inverse demand, standing in for the rest of electricity consumption that buys guarantees of origin.

Ammonia demand is inelastic. Its level is `EP_Market.Total_Demand`, shaped by `LOAD_EP`. The RED III industrial target enters as a constraint: at least 42% of the hydrogen basis must be covered by hydrogen certificates (`gamma_GC = 0.42`). That is a regulatory constraint, not a strategic rule.

Transfers cancel in social welfare. The planner therefore counts consumer utility and real resource cost (fuel, variable O&M, annualised capital). It does not count a payment from a buyer to a seller twice.

## 4. Risk

Weather and gas prices make profit random. For a capacity owner the loss in scenario $y$ is operating cost minus revenue, plus annualised capital. Capital is paid in every scenario, so a large plant that earns poorly in the tail is penalised in every tail scenario. CVaR is applied to this full loss.

Rockafellar and Uryasev (2000):

$$
\mathrm{CVaR}_{\beta}(\ell)=\min_{\alpha}\left\{\alpha+\frac{1}{1-\beta}\mathbb{E}\bigl[(\ell-\alpha)_{+}\bigr]\right\}.
$$

In code this is linear: shortfall $u_y\ge \ell_y-\alpha$, and $\mathrm{CVaR}\ge \alpha+\frac{1}{1-\beta}\sum_y P_y u_y$.

Risk-aware firms (solar, wind, electrolyzer, green offtaker) minimise

$$
\gamma\,\mathbb{E}[\ell]+(1-\gamma)\,\mathrm{CVaR}_{\beta}(\ell)
$$

plus ADMM penalties, which are outside the loss. The planner maximises the same split on social welfare: $\gamma$ times expected welfare minus $(1-\gamma)$ times the social CVaR of welfare. Consumers, conventional plants, the grey offtaker, and certificate demand have no private CVaR. Risk reaches them only through prices.

**How $\gamma$ and $\beta$ are used.** The split follows Höschle, Le Cadre, Smeers, Papavasiliou and Belmans (2018), with one labelling difference that matters if their paper is used as a key.

- $\gamma$ sets the weight between the mean and the tail. `gamma = 1` turns CVaR off. `gamma = 0.5` puts equal weight on the mean and on CVaR, which is the risk-averse base case.
- $\beta$ chooses the tail, and only matters once $\gamma<1$. Here $\beta$ is the Rockafellar confidence level. $\beta=0.8$ averages the worst 20% of scenarios. $\beta=0.2$ averages the worst 80%. **Higher $\beta$ is more risk-averse.**
- Höschle et al. sweep a parameter they also call $\beta$ from 1 down toward 0, and on their axis a *lower* value is more risk-averse. The economic direction of the sweep matches. The numbers do not: their $\beta=0.2$ is not this model's $\beta=0.2$.

With 15 equiprobable scenarios, one scenario is $1/15$ of the probability. A tail narrower than that collapses CVaR onto the single worst scenario, so $\beta$ must satisfy $1-\beta\ge 1/15$, i.e. $\beta\le 0.933$. The paper sweep is $\beta\in\{0.2,0.4,0.6,0.8\}$ at $\gamma=0.5$, which is exactly 12, 9, 6, and 3 scenarios. `beta = 0.95` is rejected for this grid. Both the planner and ADMM read `ADMM.gamma` and `ADMM.beta`. Per-agent copies of those keys do not override the ADMM block.

**What changes as risk aversion rises**, compared within one script at fixed $\gamma=0.5$:

- **Planner.** The costly scenarios are the high-gas years. Social CVaR therefore buys more wind, electrolysis, and green ammonia as a fuel hedge. Expected welfare falls and resource cost rises. The step from risk-neutral to $\beta=0.2$ is the large one, because CVaR enters at weight $0.5$ and drops the three cheapest-gas scenarios. Later $\beta$ only thins a nested tail.
- **Market exposure.** Each firm marks its own profit to market. For wind the tail is bad weather, not the social high-gas years, so higher $\beta$ cuts new wind (existing capacity is a floor). Electrolysis and the green offtaker move less.
- **Coalitions.** One CVaR on joint profit at the residual spot prices. That is complete sharing inside the merged firms, and it is not the planner's social CVaR. Compare a coalition to its own risk-neutral run, then to market exposure.
- **Contracts.** At $\gamma=1$ a fairly priced contract has zero expected transfer and equilibrium capacity on the contract goes to zero. At $\gamma<1$ locking a slice at a fixed price trims both parties' tails, so contract capacity can be positive. The hedge is narrower than merging the firms.

Private CVaRs do not internalise the social tail. Market exposure can therefore look well hedged firm by firm and still have a worse ex-post social tail than the planner.

## 5. Social planner

The planner is one convex quadratically constrained program: every agent constraint, market clearing, and one social CVaR. Welfare in each scenario is an epigraph variable, so the CVaR of welfare stays linear in the constraints. Ipopt solves it. Gurobi solves the ADMM subproblems.

Prices are the duals of the balance constraints, scaled back to €/MWh:

$$
\lambda_k(h,d,y)=\frac{\mathrm{dual}_{k,h,d,y}}{W_{d,y}\,\mu_y}.
$$

At $\gamma=1$, $\mu_y=P_y$. At $\gamma<1$, $\mu_y$ is the marginal weight of scenario $y$ in the risk-adjusted objective, read from the dual of the epigraph constraint (expected-welfare weight plus the CVaR tail weight). Dividing by $W\cdot\mu$ is required. The raw dual is not a €/MWh price, because the objective sums hourly terms that have already been multiplied by the day weight and the scenario weight.

Ipopt's `ipopt_tol: 1e-6` is a KKT residual tolerance, not a price tolerance. On this calibration, electricity prices are on the order of 100 €/MWh and expected welfare on the order of tens of billions of euro, so a residual of $10^{-6}$ sits far below anything reported in €/MWh or in capacity (MW). A tighter default of $10^{-8}$ makes the risk-averse QCP fail in Ipopt's restoration phase without changing reported prices. When $\gamma<1$, `risk_warmstart: true` solves the risk-neutral planner first and reseats the CVaR auxiliaries, because a cold start of the same feasible set often dies in restoration even though only the objective has changed.

## 6. ADMM

ADMM is sharing form in the sense of Boyd, Parikh, Chu, Peleato and Eckstein (2011). Each iteration:

1. Every agent solves its own quadratic program at the current prices and consensus targets.
2. Spot imbalances are formed. Price steps against the imbalance: excess supply lowers $\lambda$, excess demand raises it.
3. The penalty $\rho$ is adapted by residual balancing, separately per market.
4. The loop stops when every spot market meets the primal and dual tests below.

The hydrogen-certificate price is projected onto $\lambda\ge 0$. A producer does not issue certificates at a negative price, so without the floor the demand side is unbounded and the market cycles. A modest step-size damping near the tolerance limits end-game oscillation. Both devices are inactive at a cleared market. Checkpoint rollback and basin guards exist in `Source/ADMM.jl` and are switched off; the published algorithm does not use them.

**Capacity is a private variable.** Installed MW is chosen inside the agent program, subject to availability ($g\le \mathrm{AF}\cdot\mathrm{cap}$ for renewables, and the analogous peak constraints for the electrolyzer and the green offtaker). There is no ADMM consensus on capacity. An earlier split that set an auxiliary equal to peak dispatch and penalised the gap tied investment to last iteration's operation. Together with a risk-averse warm start from the planner, that forced market exposure to copy the planner's extra green capacity and then slide back, which is the opposite of the private-CVaR result. `epsilon_cap` does not stop `market_exposure.jl`. It is the tolerance on contract capacity in `me_contracts.jl` only.

**Stopping test.** Let $n=24\times 8\times 15=2880$ be the number of time slots and $\varepsilon$ the per-slot tolerance (`ADMM.epsilon`, shipped at 0.2 MW). A market has converged when the L2 norm of its imbalance, and of its dual residual, is at most $\varepsilon\sqrt{n}$. The square root is Boyd's scaling: it holds the root-mean-square imbalance at $\varepsilon$ if the horizon is refined. At the shipped value the L2 bar is about 10.7 MW. Against a system peak near 20 GW that is about 0.001% of peak, and the money value of a persistent 0.2 MW imbalance is negligible next to social welfare. Tightening `epsilon` is a numerical choice. It does not change the equilibrium definition.

**Warm start.** Run `social_planner.jl` first.

| $\gamma$ | What is loaded from the planner | Why |
|----------|----------------------------------|-----|
| 1 | Prices, quantities, and capacity starts | The two programs share a solution. A full seed converges in a few iterations. |
| below 1 | Prices only | Planner quantities are the social fuel hedge. Using them as ADMM targets pins renewables to that hedge. Risk-averse ADMM must be free to cut wind. |

Risk-averse market exposure from a cold quantity target typically takes a few thousand iterations. `max_iter: 5000` covers that. If the planner result folder is missing, ADMM still runs from the scalar `initial_price` values in `data.yaml`.

## 7. Bilateral contracts

`me_contracts.jl` keeps the five spot markets and adds two links.

| | PPA | HPA |
|--|-----|-----|
| Parties | VRES sells, electrolyzer buys | Electrolyzer sells, green offtaker buys |
| Volume | Pay-as-produced: availability factor times capacity $C$ | Baseload: $C$ every hour |
| Price | One fixed $K$ for the horizon, covering electricity plus its certificate | One fixed $K$, covering hydrogen plus its certificate |

Physical megawatt-hours still clear in the spot pools. On the contracted slice the seller receives $K$ instead of the bundled spot price, and the buyer pays $K$ instead of that spot price. The cash flow enters both firms' losses, hence both CVaRs. Summed across the two parties the transfer is zero. Writing $K$ on top of full spot revenue would pay the seller twice; the code adds only the difference between $K$ and the bundled spot price.

$C$ is one scalar per link, chosen before the scenario is known, and bounded by the plants it is written on (seller capacity, and buyer conversion capacity). Both sides pick their own $C$. ADMM enforces seller capacity equal to the sum of buyer capacities. $K$ is the dual of that equality, floored at zero: a negative fixed price would mean the seller pays the buyer to take the energy. With `initial_price: spot`, $K$ starts at the expected bundled spot price on the contract profile, which is the risk-neutral fair value.

## 8. Uncertainty set

Investment is one operating year. The 15 scenarios are alternative realisations of that year, not a sequence of years in which the agent rebuilds.

$$
n=5\times 3=15,\qquad jy=(\text{weather index}-1)\cdot 3+\text{gas index}.
$$

Gas varies fastest. Scenarios 1–3 are weather label 1 at 1×, 2×, and 3× gas, and so on. All 15 have probability $1/15$. The weather file is read once per label and reused across the three gas levels. Gas does not shift the electricity demand profile. Price response of electricity demand is already inside the elastic consumer; an extra exogenous shift would double-count it.

### Weather labels

Labels were selected, not taken as the last five calendar years. Ten ERA5 years were scored, after a common Netherlands calibration, by a 31-dimensional vector (monthly solar and wind capacity factors, plus annual means, variability, dunkelflaute frequency, and heating and cooling degree-hours). An exhaustive search over the 252 subsets of five years maximised the minimum pairwise distance, so the set cannot be four similar years plus one outlier.

| Label | ERA5 year | Role |
|-------|-----------|------|
| 1 | 2015 | Reference. Highest wind capacity factor, fewest low-renewable days. |
| 2 | 2010 | Stress year. Lowest wind, coldest winter, worst dunkelflaute. |
| 3 | 2016 | Mid-range. |
| 4 | 2017 | Low solar, high wind. |
| 5 | 2018 | Highest solar, hottest summer. |

Dunkelflaute here is the share of days whose daily-mean wind capacity factor is below 0.10 and daily-mean solar capacity factor is below 0.05. Exact figures are in `Input/weather_scenario_summary.json`.

Hourly weather is ERA5 at 52.09°N, 5.12°E (Open-Meteo historical API): global horizontal irradiance, 100 m wind speed, and 2 m temperature. Solar capacity factor is irradiance over 1000 W/m², capped at 1. Wind uses a standard power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s).

**One fleet, five years.** Raw ERA5 conversion does not reproduce the Dutch fleet. Two multipliers are fitted once, on the reference year, so that its annual means hit the CBS 2024 fleet averages of 18.2% solar and 28.0% wind ($m_{\mathrm{solar}}=1.4707$, $m_{\mathrm{wind}}=1.5711$). The same multipliers are then applied to every year and clamped to $[0,1]$. Rescaling each year to its own target would have forced every year to 18% and 28% and erased the energy risk the scenarios exist to represent.

### Electricity demand

Demand is rebuilt for each weather year from the same temperature series. A cold, still year is also a high-demand year. The shape is multiplicative: hour of day, weekday versus weekend (weekend factor 0.87, on the real calendar of the source year), a small non-thermal seasonal term, and a thermal term $1+a_H\,\mathrm{HDD}+a_C\,\mathrm{CDD}$.

Heating base 15.5°C and cooling base 22°C are the European degree-day convention. The electrical sensitivities are deliberately small ($a_H=0.010$ per °C, $a_C=0.008$ per °C). Dutch space heat is still mostly gas, so the electrical temperature response is weaker than in electrically heated systems (Bessec and Fouquau, 2008). A multiplicative thermal factor puts more extra megawatts on the evening peak than overnight, which is what sizes capacity.

All five years are divided by one shared peak, the maximum raw load in the set, not by each year's own peak. A per-year normalisation would pin every year at 1.0 p.u. and delete the demand differences. Label 2 sets the system peak. `PeakLoad: 19500` MW scales that peak to the ENTSO-E 2024 Netherlands peak (19.5 GW; 2024 net consumption about 109 TWh).

`LOAD_H` and `LOAD_EP` stay fixed shapes. Absolute hydrogen and ammonia demand are set in `data.yaml`.

### Representative days

Each 8760-hour year is reduced to eight days by hierarchical clustering with medoids (Pineda and Morales, 2018), as implemented in RepresentativePeriodsFinder.jl. Days are real historical days, not averages. Solar and wind are weighted more heavily than load in the clustering features so that low-renewable days are not averaged away. Cluster sizes sum to 365.

Cluster-count weights do not reproduce annual energy. A medoid is central in shape, not in daily mean, and with eight clusters the gap is large: raw weights understated annual solar by up to about 14% and annual wind by up to about 12%, and they reordered years (the reference year fell from second-sunniest to fourth; the cold year lost its place as the highest-demand year). A risk-averse investor would then have been hedging the wrong ranking.

The published weights are the smallest Euclidean adjustment of the cluster counts such that the weighted means of solar, wind, and electrical load match the full-year means, the weights still sum to 365, and each weight is at least one day. The projection uses Dykstra's algorithm so that the result is the true projection onto that set. After the adjustment, annual means match to well under 0.1% and the year ordering is restored.

The cost is duration-curve fidelity. On labels 1 and 2, two of the eight days are pushed to the one-day floor, so those years are carried by about six days. That is the right trade here: investment is paid by annual energy and by the scenario ranking, and there is no storage state carried across days. If storage or unit commitment were added, the clean fix is more representative days, not a return to raw cluster weights. Solve time scales linearly with hours × days × scenarios.

### Gas prices

`Fuel` is the only place fuel and carbon prices are written. A conventional plant's short-run marginal cost is fuel price over efficiency, plus the emission factor over efficiency times the CO₂ price, plus variable O&M. Grey ammonia is gas intensity times the gas price, plus CO₂ intensity times the CO₂ price, plus variable O&M. One shock therefore moves power and ammonia together. That is intentional: a modern CCGT and an SMR ammonia plant have essentially the same gas exposure per MWh of output (about 1.72 MWh of gas per MWh of electricity at 58% efficiency, and 32 GJ of gas per tonne of ammonia, which is also about 1.72 MWh of gas per MWh of ammonia).

Only gas is multiplied by 1, 2, and 3. Coal, biomass, and the EU ETS price stay at their 2024 anchors, so the shock reorders the merit order instead of shifting every plant by the same amount. At 1× gas, CCGT is the cheapest of the three fossil plants (about 83 €/MWh). At 2× and 3× it is far above coal and biomass. Grey ammonia moves from about 86 €/MWh (about 440 €/t) at 1× to about 204 €/MWh (about 1,050 €/t) at 3×.

The 2024 anchor (TTF 34.40 €/MWh thermal, EUA 64.79 €/t) matches `base_year: 2025` for capacities, which are CBS end-2025 figures, with fuel and wholesale prices from the last complete market year. The derived grey cost sits next to the 2024 Northwest Europe ammonia range, and the CCGT cost sits next to the 2024 Dutch day-ahead average (77 €/MWh), which was often set by renewables and imports rather than by gas.

## 9. Calibration

Netherlands, capacity year 2025, fuel and price year 2024. Endogenous investment and all market prices are outputs. The numbers below are inputs or bounds.

| Input | Value | Why this number |
|-------|-------|-----------------|
| Solar / wind existing capacity | 25,881 / 11,782 MW | CBS installed fleet, end-2025 | 
| Solar / wind fixed cost | 95 / 185 k€ per MW-year | IRENA 2024 capital costs, 25 years, 8% WACC, plus fixed O&M. Implied LCOE about 50 and 62 €/MWh |
| Fossil capacity | 14,040 / 1,800 / 2,160 MW | CCGT / coal / biomass, split of an 18 GW fossil proxy in line with 2024 generation shares |
| Electricity peak | 19,500 MW | ENTSO-E highest Netherlands hourly load in 2024, 19.5 GW |
| Electrolyzer specific consumption | 1.5 MWh/MWh | PEM, about 67% LHV |
| Electrolyzer fixed cost | 262 k€ per MW-year electrical | IEA 2024 capex 2,160 USD/kWe, 0.92 €/USD, 8% over 20 years plus 3% fixed O&M. Nameplate is electrical input |
| Electrolyzer seed capacity | 800 MW electrical | About 20% of Dutch ammonia hydrogen feed. A seed, not a target |
| Green ammonia fixed cost | 158 k€ per MW-year of product | IEA synthesis-loop plus air separation, excluding the electrolyzer, same annuity |
| Green ammonia seed | 400 MW product | Matches the electrolyzer seed at conversion 0.75 |
| Ammonia demand | 1,970 MW average-equivalent | About 3 Mt/year (Yara Sluiskil plus OCI Geleen), times 5.167 MWh/t |
| Grey capacity | 1,570 MW | The rest of that nameplate |
| Grey gas / CO₂ intensity | 1.720 MWh gas and 0.348 tCO₂ per MWh ammonia | 32 GJ and 1.8 tCO₂ per tonne, BAT ammonia. Process CO₂ is charged in full; free allocation is a lump-sum transfer and does not change the marginal decision |
| Haber–Bosch conversion | 0.75 | Hydrogen-to-ammonia LHV efficiency in the 70–80% range |
| Certificate mandate | 0.42 | RED III, renewable hydrogen in industry by 2030 |
| Certificate willingness to pay | intercept 10 €/MWh | Order of recent European guarantee-of-origin prices. The market clears below the intercept |

**Modelling choices, not Dutch observations.** These are set so the economics are well posed. They should not be read as measured Netherlands data.

- Electricity inverse demand: intercept 500 €/MWh and a small slope, so demand is nearly inelastic around a 20 GW system. The slope exists so the consumer problem has a maximum; it is not an estimated elasticity.
- Certificate demand slope, electrolyzer variable O&M (3 €/MWh hydrogen), and green-ammonia processing cost (12 €/MWh, about 62 €/t) are engineering allowances. Fixed O&M of the electrolyzer is already inside the annuity.
- The importer cost is unused while its capacity is zero.
- `gamma`, `beta`, ADMM penalties, and iteration limits are preferences and numerical settings.

`VRES_FixedCost_Scale` multiplies both renewable annuities after the file is loaded. The shipped value is 1.

## 10. Reading the results

Each script writes a results folder beside itself. The files used for the paper comparison are:

| File | What it is |
|------|------------|
| `Cost_Metrics.csv` | Expected social welfare, resource cost, consumer cost |
| `Risk_Metrics.csv` | Social CVaR and, where relevant, the sum of private CVaRs |
| `Market_Prices.csv` | One row per hour, day, and scenario. Same columns for the planner and for ADMM, so the two files compare directly |
| `Agent_Summary.csv` | Net sales by market (positive is a sale), final capacity, investment, objective |
| `run_summary.txt` | The console summary of that run |
| `SP_Capacities.csv`, `SP_Primal_Quantities.csv` | Planner investment and dispatch. ADMM reads these only in the risk-neutral warm start |
| `PPAs.csv`, `HPAs.csv` | Contract capacity, fixed price, fair spot value, and the premium of $K$ over that fair value. Written only by `me_contracts.jl` |

`ADMM_Convergence.csv` records the primal and dual residuals. Convergence is the five spot markets (and, for contracts, each link against `epsilon_cap`). Columns related to a capacity consensus are retained from an earlier formulation and do not decide the stop.

Social welfare in the planner and the sum of agent objectives in ADMM are comparable at `gamma = 1` once transfers are treated as transfers. At `gamma < 1` compare institutions on expected welfare and on resource cost, and expect the allocations to differ.

## 11. Where the equations are in the code

| Piece | File |
|-------|------|
| Scenario grid | `Source/define_scenarios.jl` |
| Agent data from `data.yaml` | `Source/define_*_parameters.jl` |
| Agent programs | `Source/build_*_agent.jl`, `Source/solve_*_agent.jl` |
| Coalitions | `Source/build_merged_agent.jl`, `Source/solve_merged_agent.jl` |
| Planner, including social CVaR | `Source/build_social_planner.jl` |
| Planner prices | `Source/save_social_planner_results.jl` |
| ADMM loop | `Source/ADMM.jl` |
| Contract cash flow and the $K$ update | `Source/contract_settlement.jl`, `Source/ADMM_contracts.jl` |

## 12. References

1. Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1–122.
2. Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–42.
3. d'Aertrycke, G. de Maere, Ehrenmann, A., Ralph, D. and Smeers, Y. (2018). Risk trading in capacity equilibrium models. EPRG Working Paper 1720.
4. Höschle, H., Le Cadre, H., Smeers, Y., Papavasiliou, A. and Belmans, R. (2018). An ADMM-based method for computing risk-averse equilibrium in capacity markets. *IEEE Transactions on Power Systems*, 33(5), 4819–4830.
5. Gabriel, S. A., Conejo, A. J., Fuller, J. D., Hobbs, B. F. and Ruiz, C. (2013). *Complementarity Modeling in Energy Markets*. Springer.
6. Pineda, S. and Morales, J. M. (2018). Chronological time-period clustering for optimal capacity expansion planning with storage. *IEEE Transactions on Power Systems*, 33(6), 7162–7170.
7. Bessec, M. and Fouquau, J. (2008). The non-linear link between electricity consumption and temperature in Europe. *Energy Economics*, 30(5), 2705–2721.
8. Dykstra, R. L. (1983). An algorithm for restricted least squares regression. *Journal of the American Statistical Association*, 78(384), 837–842.

Data sources named in `data.yaml`: CBS renewable and electricity tables 82610ENG, 37823ENG and 80030ENG; ENTSO-E Statistical Factsheet 2024; TTF and EUA 2024 averages; IRENA *Renewable Power Generation Costs in 2024*; IEA *Global Hydrogen Review 2024* assumptions annex; IPCC 2006 stationary combustion factors; the JRC BAT reference for ammonia; PBL/ECN (2019) on the Dutch fertiliser industry; Directive (EU) 2023/2413 (RED III); ERA5 via the Open-Meteo historical API. Representative days were selected with RepresentativePeriodsFinder.jl (KU Leuven), which is not vendored in this release because the solver does not need it. The files in `Input/` are the selected days and the rebalanced weights.
