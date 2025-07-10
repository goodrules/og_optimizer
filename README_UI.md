# Oil & Gas Field Development Optimizer UI

## Overview

This UI transforms the work package optimizer from `local/main.py` into an oil & gas field development tool focused on Texas oil wells.

## Key UI Transformations

### Control Panel (Left Side)
- **Development Parameters**
  - Oil Price ($/bbl): Base case oil price for economics
  - Discount Rate (%): Minimum acceptable return (hurdle rate)
  - Contingency (%): Budget reserve for cost overruns

- **Drilling Execution**
  - Horizontal Rigs Available: Number of drilling rigs (1-5)
  - Drilling Mode: Continuous vs Pad/Batch drilling
  - Permit Delay: Texas Railroad Commission timeline

- **Lease Selection**
  - 5 Texas leases with toggles and well count sliders
  - MIDLAND_A (Permian): Up to 28 wells
  - MARTIN_B (Permian): Up to 15 wells
  - REEVES_C (Delaware): Up to 32 wells
  - LOVING_D (Delaware): Up to 22 wells
  - HOWARD_E (Permian): Up to 12 wells

### Visualizations (Right Side)

1. **Metrics Card** (replaces work package stats)
   - Total NPV in $MM
   - Total CAPEX deployed
   - NPV improvement from baseline
   - Wells drilled count
   - NPV/$ efficiency ratio
   - Peak production (boe/d)
   - Risk score

2. **Drilling Timeline** (replaces Gantt chart)
   - Shows drilling schedule by lease
   - Color-coded by lease
   - Rig allocation visualization

3. **Production Forecast** (new)
   - 10-year production profile
   - Hyperbolic decline curves
   - P10/P90 uncertainty bands (when Monte Carlo enabled)

4. **Economics Sensitivity** (new)
   - NPV vs oil price chart
   - Shows current scenario marker
   - Helps understand price risk

5. **Optimization History 3D** (adapted)
   - X-axis: NPV ($MM)
   - Y-axis: CAPEX ($MM)
   - Z-axis: Investment efficiency (NPV/$)
   - Shows optimization convergence

## Running the UI

```bash
# Install dependencies
pip install nicegui plotly numpy

# Run the application
python main.py
```

The UI will open at http://localhost:8080

## Key Features

- **Real-time optimization**: Click "Optimize Development" to run heuristic search
- **Interactive controls**: Adjust parameters and see immediate impact
- **Visual convergence**: Watch the optimization improve over ~20 trials
- **Texas-specific defaults**: Pre-configured for Permian/Delaware basin economics

## Architecture

The UI leverages:
- NiceGUI for reactive web interface
- Plotly for interactive charts
- Async execution for optimization runs
- Client-side storage for trial history