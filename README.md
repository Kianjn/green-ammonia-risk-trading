# Multi-Agent Energy Market Equilibrium

Competitive equilibrium for coupled electricity, hydrogen, certificate, and ammonia markets. A decentralised ADMM solver and a social-planner benchmark share one technology stack, with endogenous investment and CVaR risk aversion.

**Kian Jafarinejad** · Delft University of Technology · [K.Jafarinejad@tudelft.nl](mailto:K.Jafarinejad@tudelft.nl)

The mathematical assumptions, calibration, and numerical tolerances are in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## What this model does

Independent price-taking firms trade on five spot markets. Investment in wind, solar, electrolysis, and green ammonia is chosen once, before uncertainty resolves. Every risk-aware agent optimises over **15 equiprobable scenarios**: 5 weather years crossed with 3 natural-gas price levels (1× / 2× / 3× the 2024 TTF average). Weather moves renewable availability and, through a degree-day model, electricity demand. Gas moves the short-run cost of gas-fired power and grey ammonia together.

Five institutions are solved from the same configuration file:

| Script | Institution |
|--------|-------------|
| [`social_planner.jl`](social_planner.jl) | Complete risk trading. One social CVaR on aggregate welfare. |
| [`green_social_planner.jl`](green_social_planner.jl) | Complete risk trading inside the green chain (solar, wind, electrolyzer, green offtaker). |
| [`green_h2_social_planner.jl`](green_h2_social_planner.jl) | Complete risk trading inside the hydrogen chain (electrolyzer and green offtaker). |
| [`me_contracts.jl`](me_contracts.jl) | Partial risk trading. Separate firms, plus one fixed-price PPA and one fixed-price HPA. |
| [`market_exposure.jl`](market_exposure.jl) | No risk trading. Each firm hedges only through its own CVaR. |

At `gamma = 1` the decentralised runs reproduce the planner (first welfare theorem). At `gamma < 1` they are different risk institutions: the planner adds green capacity as a fuel-price hedge; market exposure cuts wind because private CVaR penalises low-revenue weather. The contracts case sits between the matching green coalition and full market exposure.

## Requirements

| Component | Role |
|-----------|------|
| [Julia](https://julialang.org/downloads/) 1.9+ (tested on 1.12.4) | Runtime. `Manifest.toml` pins the package versions used for the paper. |
| [Gurobi](https://www.gurobi.com/) 10+ | ADMM agent subproblems. Academic licenses: [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). |
| Ipopt (installed by Julia) | Social-planner QCP. Used because it returns reliable duals, which are the planner prices. |

Gurobi is not redistributed with this repository. A local license is required before `market_exposure.jl`, the coalition scripts, or `me_contracts.jl` will solve.

## Installation

```bash
git clone <repository-url>
cd <repository-name>
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e "using Gurobi; Gurobi.Env(); println(\"Gurobi OK\")"
```

## Run

Edit risk settings in [`Data/data.yaml`](Data/data.yaml) (`ADMM.gamma`, `ADMM.beta`), then:

```bash
julia --project=. social_planner.jl
julia --project=. market_exposure.jl
julia --project=. green_h2_social_planner.jl
julia --project=. green_social_planner.jl
julia --project=. me_contracts.jl
```

Run the social planner before any ADMM script. At `gamma = 1`, ADMM warm-starts from planner prices, quantities, and capacities and typically converges in a few iterations. At `gamma < 1`, only prices are loaded. Seeding risk-averse ADMM with the planner's extra green capacity would bias the result. Risk-averse market exposure usually takes on the order of 3,000–4,000 iterations (`max_iter: 5000`).

Results are written next to the scripts:

| Script | Folder |
|--------|--------|
| `social_planner.jl` | `social_planner_results/` |
| `market_exposure.jl` | `market_exposure_results/` |
| `green_h2_social_planner.jl` | `green_h2_social_planner_results/` |
| `green_social_planner.jl` | `green_social_planner_results/` |
| `me_contracts.jl` | `me_contracts_results/` |

Headline comparison files are `Cost_Metrics.csv` and `run_summary.txt` in each folder. These folders are created by a run and are not part of the repository.

**Risk sweep used in the paper.** Set `gamma: 1` for the risk-neutral benchmark (any `beta`; CVaR drops out). For risk aversion set `gamma: 0.5` and sweep `beta` over `0.2`, `0.4`, `0.6`, `0.8`. With 15 scenarios those values are tails of exactly 12, 9, 6, and 3 scenarios. Higher `beta` is more risk-averse. Do not use `beta: 0.95`: that tail is narrower than one scenario. The file as shipped has `gamma: 0.5` and `beta: 0.6`.

After changing anything under `Source/`, start a new `julia --project=.` process. Reusing a long-lived Julia session can keep a stale copy of a source file.

## Repository layout

```
├── social_planner.jl              # centralised benchmark
├── market_exposure.jl             # decentralised ADMM
├── green_h2_social_planner.jl     # hydrogen-chain coalition
├── green_social_planner.jl        # full green-chain coalition
├── me_contracts.jl                # ADMM plus PPA and HPA
├── Data/data.yaml                 # all economic and numerical settings
├── Input/                         # five weather years, eight representative days each
├── Source/                        # agents, markets, ADMM, planner
├── docs/TECHNICAL.md              # assumptions, formulation, calibration
├── Project.toml
└── Manifest.toml
```

Agents are data, not code. Supported types are added by a new block under `Power`, `Hydrogen`, `Hydrogen_Offtaker`, or `Elec_GC_Demand` in `data.yaml`.

## Citation

If you use this model, please cite the accompanying paper and this repository. Update [`CITATION.cff`](CITATION.cff) with the paper DOI once it is public.

## License

Model code and documentation are released under the [MIT License](LICENSE). Weather inputs are derived from ERA5 via the Open-Meteo historical API and may be reused with attribution to ECMWF/ERA5 and this repository.
