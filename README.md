# Campus Buggy Simulation

A Python simulation and evaluation environment for an autonomous campus buggy.
The project uses a **Model Predictive Path Integral (MPPI)** local controller
to follow a route, avoid static and moving obstacles, perform selected
maneuvers, and report navigation and safety metrics.

The runner currently supports **MPPI only**. RPP and DWA controller code was
removed; this project should not be treated as a benchmark comparing those
controllers.

## Contents

- [Project scope](#project-scope)
- [Features](#features)
- [Scenarios](#scenarios)
- [Simulation model](#simulation-model)
- [Project layout](#project-layout)
- [Requirements and installation](#requirements-and-installation)
- [Run the simulation](#run-the-simulation)
- [Command-line options](#command-line-options)
- [Results and reports](#results-and-reports)
- [Run the tests](#run-the-tests)
- [Configuration](#configuration)

## Project scope

The simulator models a low-speed buggy navigating a two-dimensional campus
road. MPPI samples candidate steering and acceleration sequences over a
prediction horizon, evaluates the resulting trajectories, and uses their
costs to select a control action. This process repeats at every simulation
step.

The runner can execute the five core scenarios, the registered edge-case
scenarios, or a selected subset. Each run resets its scenario and simulation
components and uses a repeatable seed. Optional Pygame visualization displays
the simulation as scenarios execute sequentially.

## Features

- MPPI local planning with trajectory sampling and weighted control updates.
- Cost critics for goal progress, path tracking, obstacles, lane discipline,
  maneuvers, speed, and control smoothness.
- A kinematic bicycle model with speed, acceleration, steering, and turning
  constraints.
- Static and dynamic obstacles, a grid costmap, and lane guidance.
- Maneuver handling including left turns, right turns, U-turns, and recovery
  behavior.
- Health monitoring for collisions, road-boundary exits, lane discipline, and
  stalled movement.
- Per-run and per-timestep CSV logging, summary metrics, and eight analysis
  plots for core scenario results.
- Headless batch operation or interactive Pygame visualization.

## Scenarios

There are **five core scenarios** and **nine registered edge scenarios**.
Together, the runner exposes 14 scenarios. Edge case numbering intentionally
skips EC2; the former EC2 oscillation case was removed with the controller it
targeted.

### Core scenarios

| ID | Scenario | Purpose |
|---|---|---|
| `s1_static_box` | Static box | Navigate around a rectangular obstacle on the reference path. |
| `s2_person_crossing` | Person crossing | React to a moving pedestrian crossing the buggy's route. |
| `s3_narrow_gap` | Narrow gap | Pass between walls through a gap whose width varies by run. |
| `s4_two_obstacles` | Two obstacles | Weave around staggered obstacles, including a drifting obstacle. |
| `s5_sudden_obstacle` | Sudden obstacle | Respond to an obstacle spawned during the run. |

### Edge scenarios

| ID | Scenario | Purpose |
|---|---|---|
| `ec1_dead_end` | Dead end | Exercise dead-end detection and recovery behavior. |
| `ec3_tight_turn` | Tight turn | Exercise maneuvering through a constrained turn. |
| `ec4_late_braking` | Late braking | Test response to a braking hazard. |
| `ec5_crowd` | Crowd | Navigate among multiple moving pedestrians. |
| `ec6_wrong_lane_entry` | Wrong-lane entry | Exercise an obstacle-induced lane change and return. |
| `ec7_u_turn_blocked` | Blocked-road U-turn | Execute a U-turn at a blocked road end and stop after returning to the source node. |
| `ec8_missing_lane_mark` | Missing lane marking | Navigate with lane guidance disabled. |
| `ec9_sensor_noise` | Sensor noise | Test navigation with noisy position and heading feedback. |
| `ec10_all_samples_collide` | All samples collide | Exercise MPPI recovery when sampled trajectories are blocked. |

Use `--edge-cases` to run the core and edge scenarios together, or `--edge-only`
to run only the nine edge scenarios.

## Simulation model

The buggy uses a front-steered kinematic bicycle model:

```text
dx/dt     = v cos(theta)
dy/dt     = v sin(theta)
dtheta/dt = v tan(delta) / L
dv/dt     = a
```

Here, `x` and `y` are position, `theta` is heading, `v` is speed, `delta` is
steering angle, `L` is wheelbase, and `a` is acceleration.

Key defaults are defined centrally in [`config.py`](config.py):

| Parameter | Default |
|---|---:|
| Map size | 20 m x 20 m |
| Wheelbase | 0.5 m |
| Vehicle radius | 0.3 m |
| Maximum speed | 1.5 m/s |
| Maximum acceleration | 1.0 m/s^2 |
| Maximum steering angle | 0.6 rad |
| Simulation timestep | 0.05 s (20 Hz) |
| Runs per scenario | 10 |
| Default run time limit | 60 s |
| Goal tolerance | 0.5 m |

Individual scenarios can define their own start, goal, obstacles, or time limit.
The central defaults and MPPI tuning parameters can be changed in
[`config.py`](config.py).

## Project layout

```text
campus_buggy_sim/
├── config.py                     # Simulation and MPPI configuration
├── main.py                       # CLI, scenario selection, and run orchestration
├── requirements.txt              # Runtime Python dependencies
├── pyproject.toml                # Project metadata and dependency declaration
├── controllers/
│   ├── controller_base.py        # Shared controller interface
│   └── mppi.py                   # MPPI controller
├── environment/
│   ├── costmap.py                # Grid costmap and obstacle costs
│   ├── lane_map.py               # Lane geometry and lane costs
│   └── obstacle.py               # Static and dynamic obstacle model
├── health/
│   └── monitor.py                # Safety and health checks
├── maneuvers/                    # Turn, U-turn, reverse, and maneuver state logic
├── metrics/
│   ├── recorder.py               # Run metrics and CSV output
│   └── report_generator.py       # Core-results report generation
├── mppi_critics/                 # MPPI trajectory-cost components
├── robot/
│   └── bicycle_model.py          # Kinematic vehicle model
├── scenarios/                    # Five core scenarios and scenario base
│   └── edge_cases/               # Nine registered edge scenarios
├── tests/                        # Automated test suite
├── visualization/
│   ├── pygame_viz.py             # Live 2D visualization
│   └── plot_results.py           # Matplotlib analysis plots
└── results/                      # Generated run data, summaries, and plots
```

## Requirements and installation

Python **3.10 or newer** is required. From the project directory, create and
activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
cd C:\simulation_buggy\campus_buggy_sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either adjust the execution policy for your
environment or invoke the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

### Linux or macOS

```bash
cd campus_buggy_sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the simulation

Run these commands from the project directory:

```bash
# Run all five core scenarios, using the configured runs per scenario
python main.py

# Run one episode of each core scenario in the Pygame GUI
python main.py --viz --runs 1

# Run all core and edge scenarios sequentially in one GUI
python main.py --viz --edge-cases --runs 1

# Run one episode of each edge scenario without visualization
python main.py --edge-only --runs 1

# Run only the two-box scenario
python main.py --scenarios s4 --runs 1

# Run only the blocked-road U-turn edge case
python main.py --edge-only --scenarios ec7 --runs 1

# Run headless, even if visualization is otherwise enabled by configuration
python main.py --headless --runs 1
```

With `--viz`, one Pygame window is used while selected scenarios and runs
execute in sequence. Close the window only if you intend to interrupt the run.
To visualize MPPI's sampled trajectories as well, add
`--vis-mppi-samples` to a command that includes `--viz`.

## Command-line options

| Option | Description |
|---|---|
| `--viz` | Enable live Pygame visualization. |
| `--headless` | Force execution without the Pygame window. |
| `--runs N` | Set the number of runs per selected scenario; must be at least 1. |
| `--ctrl MPPI` | Explicitly select the only supported controller. Other controller names are rejected. |
| `--scenario TEXT` | Legacy filter: run scenarios whose names contain this substring. |
| `--scenarios NAME [NAME ...]` | Select scenarios by one or more case-insensitive substrings, such as `s1 s3` or `ec7`. |
| `--edge-cases` | Include edge scenarios together with core scenarios. |
| `--edge-only` | Run only edge scenarios. |
| `--report` | Request generation of the eight core-results plots after the run. |
| `--no-plots` | Skip automatic post-run plot generation. |
| `--vis-mppi-samples` | Display sampled MPPI trajectories in the Pygame visualization; requires `--viz`. |
| `--help` | Display all command-line options. |

When both `--scenario` and `--scenarios` are supplied, both filters apply.
`--scenario` is case-sensitive; `--scenarios` is case-insensitive. Unknown
names passed to `--scenarios` are reported as command-line errors.

## Results and reports

Simulation results are written under [`results/`](results/). A new batch
replaces the corresponding summary CSV files; raw run files use the same
controller/scenario/run naming pattern and may be overwritten when those runs
are repeated.

### Core scenario runs

- `results/all_results.csv` — one summary-metric row per run.
- `results/summary_table.csv` — results grouped by scenario.
- `results/raw/` — per-run CSV data and detailed timestep logs.
- `results/plots/` — eight plots generated from core results unless
  `--no-plots` is specified.

### Edge-only runs

- `results/edge_case_results/all_edge_results.csv` — one summary-metric row
  per edge run.
- `results/edge_case_results/edge_summary_table.csv` — grouped edge results.
- `results/raw/` — per-run CSV data and detailed timestep logs.

The eight plots cover path comparison, minimum obstacle distance, completion
time, acceleration and jerk, failure modes, lane discipline, maneuver success,
and MPPI recovery. Automatic plot generation is for core scenario results;
edge-only batches produce their CSV outputs without regenerating the core
comparison plots.

## Run the tests

Install `pytest` in the active environment if it is not already installed:

```bash
python -m pip install pytest
```

Run the complete test suite from the project directory:

```bash
python -m pytest -q
```

To run just the runner and scenario tests:

```bash
python -m pytest tests/test_main_runner.py -q
```

## Configuration

[`config.py`](config.py) is the central place for vehicle limits, map and lane
geometry, simulation timing, goal tolerance, costmap values, and MPPI tuning.
Scenario-specific behavior is implemented in the corresponding files under
[`scenarios/`](scenarios/) and [`scenarios/edge_cases/`](scenarios/edge_cases/).
The entry point is [`main.py`](main.py).
