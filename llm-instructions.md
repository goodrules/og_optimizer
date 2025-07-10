## 🎯 Simplified Migration Plan: Work Package → Oil & Gas Field Development Optimizer

### 📁 Existing Codebase Foundation
The project builds upon the existing work package optimizer in the `local/` folder:
- **main.py**: Contains a complete OR-Tools scheduling system with:
  - Heuristic optimization using guided random search (`perturb_knobs` function)
  - Monte Carlo simulation for uncertainty modeling
  - NiceGUI interface with real-time visualizations
  - Constraint-based scheduling with crew resource management

This existing architecture provides the foundation for migration to the oil & gas domain.

### Git Repos for Reference
- For Gemini 2.5 pro integration: `GoogleCloudPlatform/generative-ai/gemini/getting-started/`
    - specifically reference async and controlled generation examples
- For more NiceGUI patterns: `zauberzeug/nicegui/examples/chat_with_ai/`
- For OR-Tools examples and patterns: `google/or-tools/examples/python/`
    - Reference for constraint programming, scheduling algorithms, and optimization patterns
- For future Vizier integration (post-MVP): `google/vizier/docs/guides/user/`

### 📊 Simplified Architecture

Your existing code structure maps nicely to the oil & gas domain:

| Current Component | New Purpose | Key Changes |
|------------------|-------------|-------------|
| Tasks → Wells | Individual well entities | Production profiles instead of hours |
| Workers → Rigs | Drilling resources | Rig count instead of crew types |
| Schedule → Drilling Timeline | When to drill each well | Capital allocation over time |
| Cost → NPV | Economic optimization | Revenue streams + decline curves |

### 🔄 Core Component Transformations

#### 1. **Simplified Data Model**

```python
# CURRENT: Tasks with dependencies
TASKS = [{"id": "T1", "hours": 40, "skill": "Pipefitter"...}]

# NEW: Texas oil wells with pre-validated locations
# Default: 5 leases with randomized well counts (5-32 per lease)
LEASES = [
    {
        "id": "MIDLAND_A",
        "basin": "Permian",
        "county": "Midland",
        "available_wells": 28,  # Randomly set between 5-32
        "royalty": 0.1875,  # Standard Texas royalty
        "working_interest": 0.80
    },
    {
        "id": "MARTIN_B",
        "basin": "Permian",
        "county": "Martin",
        "available_wells": 15,  # Randomly set between 5-32
        "royalty": 0.1875,
        "working_interest": 0.75
    },
    {
        "id": "REEVES_C",
        "basin": "Delaware",
        "county": "Reeves",
        "available_wells": 32,  # Randomly set between 5-32
        "royalty": 0.20,
        "working_interest": 0.80
    },
    {
        "id": "LOVING_D",
        "basin": "Delaware",
        "county": "Loving",
        "available_wells": 22,  # Randomly set between 5-32
        "royalty": 0.1875,
        "working_interest": 0.85
    },
    {
        "id": "HOWARD_E",
        "basin": "Permian",
        "county": "Howard",
        "available_wells": 12,  # Randomly set between 5-32
        "royalty": 0.20,
        "working_interest": 0.70
    }
]

# Wells are generated dynamically based on lease selection
WELL_TEMPLATE = {
    "base_cost": 7_500_000,  # Typical Texas horizontal well
    "ip_rate": 1000,  # boe/d (oil only for Texas focus)
    "decline_params": {
        "type": "hyperbolic",
        "di": 0.70,  # annual - typical Permian decline
        "b": 1.1
    },
    "eur_mboe": 400  # Conservative Texas oil EUR
}
```

#### 2. **Constraint Simplification**

```python
# CURRENT: Complex crew scheduling
model.AddCumulative(*buckets["Pipefitter"], pf_cap)

# NEW: Simple drilling constraints
- Max rigs available per month
- Total CAPEX budget
- Max wells per lease (pre-validated)
- Minimum production targets
```

#### 3. **OR-Tools Adaptation**

Transform from task scheduling to drilling sequence optimization:

```python
# Decision variables
wells_drilled[lease][month] = IntVar(0, max_wells_per_month)
rig_usage[month] = Sum(wells_drilled[*][month])

# Constraints
model.Add(rig_usage[month] <= available_rigs)
model.Add(Sum(wells_drilled[lease]) <= lease.available_wells)
model.Add(cumulative_capex[month] <= budget_limit)
```

#### 4. **Economic Engine Components**

New calculation modules needed:

```python
def calculate_production_profile(well, months=360):
    """Generate monthly production using decline curve"""
    
def calculate_well_npv(well, oil_price, discount_rate):
    """NPV for individual well"""
    
def calculate_field_economics(drilling_schedule, assumptions):
    """Aggregate NPV, IRR, payback period"""
```

### 🏗️ Simplified Implementation Phases

#### **Phase 1: Core Economics** (Week 1)
1. Build decline curve calculator
2. Create NPV/IRR functions  
3. Adapt existing Monte Carlo from `local/main.py` for price/cost uncertainty
4. Test with single well scenarios

#### **Phase 2: OR-Tools Integration** (Week 1-2)
1. Adapt existing CP-SAT model from `local/main.py`:
   - Replace task scheduling with drilling sequence optimization
   - Convert crew constraints to rig availability constraints
   - Update objective from minimize duration to maximize NPV
2. Implement budget and operational constraints
3. Test with 3-5 well scenarios

#### **Phase 3: Heuristic Optimization** (Week 2)
1. Adapt existing `perturb_knobs` approach from `local/main.py`:
   - Define oil & gas optimization parameters (wells per lease, timing, etc.)
   - Implement guided random search for parameter exploration
   - Use existing trial tracking and improvement logic
2. Create evaluation function for drilling scenarios
3. Integrate with economic calculations

#### **Phase 4: UI Transformation** (Week 3)
1. Leverage existing NiceGUI structure from `local/main.py`:
   - Convert crew reserve sliders to drilling parameters
   - Adapt 3D optimization history plot for NPV/EUR/efficiency
   - Transform Gantt chart to drilling timeline
2. Add oil & gas specific visualizations:
   - Production forecast curves
   - Economic dashboard (NPV, IRR, payback)

#### **Phase 5: Gemini Integration** (Week 3-4)
1. Natural language scenario builder
2. Results explanation generator
3. What-if query handler

#### **Future Enhancement: Vizier Integration** (Post-MVP)
After validating the heuristic approach:
1. Define Vizier study configuration
2. Replace manual `perturb_knobs` with Vizier's Bayesian optimization
3. Leverage Vizier's multi-objective optimization for NPV vs risk tradeoffs
4. Implement parallel trial evaluation

### 📱 Simplified UI Mapping

| Current Section | New Section | Controls |
|----------------|-------------|----------|
| **Optimization Controls** | **Development Parameters** | |
| Crew Reserve % | Contingency % | Cost overrun buffer (10-30%) |
| Overtime Penalty | Hurdle Rate | Minimum acceptable IRR (15-25%) |
| | Oil Price | Base case $/bbl (WTI: $70-90) |
| | Price Volatility | ±% uncertainty (15-30%) |
| **Supply Chain Risk** | **Drilling Execution** | |
| Part Delay | Permit Delays | Texas regulatory timeline (0-60 days) |
| Sequential/Parallel | Drilling Mode | Batch vs continuous |
| | Rigs Available | 1-5 horizontal rigs |
| **Human Factors** | **Operational Risk** | |
| Sickness Probability | Mechanical Failure % | Equipment downtime (3-10%) |
| | Weather Delays | Texas weather impact (5-20 days/year) |
| **New Section** | **Lease Configuration** | |
| | Active Leases | Select which of 5 Texas leases to develop |
| | Wells per Lease | Adjustable slider (5-32 wells per lease) |

### 📊 MVP Visualizations

1. **Drilling Schedule** (replaces Gantt)
   ```
   Timeline showing:
   - Which wells drill when
   - Rig allocation
   - Cumulative wells drilled
   ```

2. **Production Forecast** (new)
   ```
   Line chart:
   - Total field production over time
   - By lease breakdown
   - P10/P50/P90 bands
   ```

3. **Economics Dashboard** (replaces metrics card)
   ```
   KPIs:
   - NPV at different price scenarios
   - IRR
   - Payback period
   - Total CAPEX deployed
   ```

4. **Optimization History** (keep 3D plot)
   ```
   Axes:
   - X: NPV ($MM)
   - Y: EUR (MMboe)  
   - Z: Investment efficiency ($/boe)
   ```

### 🔧 Key Simplifications for MVP

1. **Pre-validated well locations**: Just specify count per lease (5-32 wells)
2. **Texas-specific type curves**: Permian/Delaware defaults with regional adjustments
3. **Horizontal rig focus**: 1-5 rigs suitable for horizontal drilling
4. **Oil-only economics**: Texas oil focus (gas/NGL monetized as oil equivalent)
5. **Texas regulatory framework**: Railroad Commission permits, standard royalties
6. **Regional defaults**: 5 Texas leases with randomized well counts on initialization

### 📝 MVP Default Configuration - Texas Oil Focus

```python
DEFAULT_CONFIG = {
    "location": {
        "state": "Texas",
        "basins": ["Permian", "Delaware"],
        "focus": "Oil",  # Simplified to oil-only production
        "lease_count": 5,
        "wells_per_lease_range": [5, 32]  # Randomly assigned on init
    },
    "wells": {
        "base_cost": 7_500_000,  # Typical Texas horizontal oil well
        "cost_uncertainty": 0.15,  # Higher uncertainty for complex horizontals
        "drill_days": 25,  # Longer for horizontal wells
        "ip_rate": 1000,  # boe/d (oil only)
        "oil_cut": 0.85,  # 85% oil, 15% associated gas (monetized as oil equivalent)
        "decline_di": 0.70,  # Typical Permian/Delaware decline
        "decline_b": 1.1,
        "eur_mboe": 400  # Conservative Texas horizontal EUR
    },
    "economics": {
        "oil_price": 80.00,  # WTI baseline
        "price_volatility": 0.25,  # Texas market volatility
        "discount_rate": 0.15,  # Higher hurdle rate for shale
        "opex_per_boe": 12.00,  # Texas operating costs
        "royalty": 0.1875,  # Standard Texas royalty
        "working_interest": 0.80,  # Typical operated position
        "severance_tax": 0.046,  # Texas oil severance tax
        "ad_valorem_tax": 0.02  # Property tax estimate
    },
    "operations": {
        "rigs_available": 2,  # Horizontal drilling rigs
        "max_wells_per_month": 3,  # Pad drilling efficiency
        "mechanical_failure_rate": 0.07,  # Complex horizontal drilling
        "weather_delay_days": 12,  # Texas weather impact
        "permit_delay_range": [0, 45],  # Texas Railroad Commission
        "water_sourcing_cost": 2.50  # $/bbl for frac water
    },
    "lease_defaults": {
        "MIDLAND_A": {"wells": 28, "working_interest": 0.80},
        "MARTIN_B": {"wells": 15, "working_interest": 0.75},
        "REEVES_C": {"wells": 32, "working_interest": 0.80},
        "LOVING_D": {"wells": 22, "working_interest": 0.85},
        "HOWARD_E": {"wells": 12, "working_interest": 0.70}
    }
}
```

### 🚀 Development Priority

1. **Start with deterministic optimization** (Adapt OR-Tools from `local/main.py`)
2. **Add Monte Carlo uncertainty** (Leverage existing MC simulation approach)
3. **Implement heuristic optimization** (Use existing `perturb_knobs` pattern)
4. **Enhance UI** (Build on existing NiceGUI structure)
5. **Add Gemini** (Natural language interface)
6. **Future: Integrate Vizier** (Replace heuristic with Bayesian optimization)

### 🔧 Implementation Strategy

The migration leverages the robust architecture already present in `local/main.py`:

1. **Constraint Solver**: The existing OR-Tools CP-SAT model provides a proven foundation for handling complex scheduling constraints. We'll adapt it from crew scheduling to rig allocation.

2. **Heuristic Optimization**: The current `perturb_knobs` approach with guided random search has demonstrated effectiveness. This will be our primary optimization method before considering more advanced techniques.

3. **Uncertainty Modeling**: The Monte Carlo simulation framework is already battle-tested and can be directly adapted for oil price volatility and operational uncertainties.

4. **UI Architecture**: The NiceGUI implementation provides real-time updates, interactive controls, and sophisticated visualizations that map directly to oil & gas metrics.

This approach maximizes code reuse while focusing on the core value: optimizing drilling sequences for maximum NPV under capital constraints.