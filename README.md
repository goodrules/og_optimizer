# Oil & Gas Field Development Optimizer

A Python-based optimization tool for oil and gas field development planning with AI-powered assistance. This application helps optimize drilling programs for Texas oil fields to maximize Net Present Value (NPV) while managing risk.

## Overview

The Oil & Gas Field Development Optimizer started as a general work package optimizer but was transformed into a specialized tool for optimizing drilling programs across multiple Texas oil leases. It combines operations research techniques (constraint programming, heuristic search) with domain-specific modeling (decline curves, economic analysis) and modern AI capabilities for enhanced user interaction.

## Features

- **Economic Optimization**: Maximize NPV across multiple drilling locations using heuristic search algorithms
- **Risk Assessment**: Calculate and visualize risk scores based on portfolio characteristics
- **Texas Focus**: Pre-configured for 5 Texas leases in Permian and Delaware basins
- **AI Assistant**: Integrated Gemini 2.5 Pro/Flash models for natural language interaction and parameter optimization
- **Real-time Visualization**: Interactive charts showing production forecasts, economics, and optimization history
- **Constraint-based Planning**: Respect budget limits, rig availability, and operational constraints
- **Monte Carlo Simulation**: Uncertainty analysis for oil prices and cost variations (optional)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Web UI (NiceGUI)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Control Panel  │  │  Visualizations │  │  AI Assistant  │  │
│  │  - Economics    │  │  - Timeline     │  │  - Gemini 2.5  │  │
│  │  - Drilling     │  │  - Production   │  │  - Chat UI     │  │
│  │  - Leases       │  │  - Sensitivity  │  │                │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬───────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
┌───────────┴────────────────────┴────────────────────┴───────────┐
│                      Core Optimization Engine                     │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  Heuristic Optimizer    │  │  OR-Tools Scheduler         │  │
│  │  - Perturb parameters   │  │  - CP-SAT constraint model  │  │
│  │  - Evaluate scenarios   │  │  - Rig allocation           │  │
│  │  - Convergence tracking │  │  - Timeline optimization    │  │
│  └───────────┬─────────────┘  └──────────┬──────────────────┘  │
└──────────────┼────────────────────────────┼─────────────────────┘
               │                            │
┌──────────────┴────────────────────────────┴─────────────────────┐
│                     Domain Models & Analytics                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Economics    │  │ Decline      │  │ Monte Carlo        │   │
│  │ - NPV/IRR    │  │ Curves       │  │ - Price volatility │   │
│  │ - Payback    │  │ - Hyperbolic │  │ - Cost uncertainty │   │
│  │ - Texas tax  │  │ - Arps eqn   │  │ - Risk metrics     │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/og_optimizer.git
cd og_optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud for Vertex AI (required for AI assistant):
   - Ensure you have a Google Cloud project with Vertex AI API enabled
   - Authenticate with Google Cloud:
     ```bash
     gcloud auth application-default login
     ```
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Update the project ID in `.env` if needed:
     ```
     GCP_PROJECT_ID=your-project-id
     GCP_REGION=us-central1
     ```

## Usage

Run the application:
```bash
python main.py
```

The application will open in your web browser at http://localhost:8080

### Using the Interface

1. **Manual Controls**: Adjust parameters using sliders:
   - Economic assumptions (oil price, discount rate, contingency, budget)
   - Drilling execution (rigs, mode, permits)
   - Well selection per lease

2. **AI Assistant**: Use natural language to:
   - Set parameters: "Set up a conservative development plan"
   - Ask questions: "What's the current NPV and risk?"
   - Request optimization: "Optimize for maximum production"
   - Select AI model: Choose between Gemini 2.5 Pro (more accurate) or Flash (faster responses)

3. **Optimization**: Click "Optimize Development" to run the heuristic optimizer

### Available Leases

- **MIDLAND_A**: Permian Basin, up to 28 wells
- **MARTIN_B**: Permian Basin, up to 15 wells
- **REEVES_C**: Delaware Basin, up to 32 wells
- **LOVING_D**: Delaware Basin, up to 22 wells
- **HOWARD_E**: Permian Basin, up to 12 wells

## Project Structure

```
og_optimizer/
├── main.py                        # Entry point - launches NiceGUI web app
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment configuration template
├── src/
│   └── oil_gas_optimizer/
│       ├── ui_app.py              # Main UI application with controls/visualizations
│       ├── heuristic_optimizer.py # Core optimization engine (perturb & evaluate)
│       ├── drilling_optimizer.py  # OR-Tools CP-SAT scheduling model
│       ├── economics.py           # NPV/IRR/payback calculations
│       ├── decline_curves.py      # Hyperbolic production decline modeling
│       ├── monte_carlo.py         # Uncertainty analysis & risk assessment
│       ├── gemini_client.py       # AI assistant using Gemini 2.5 Pro/Flash
│       ├── data_model.py          # Data structures for leases/wells
│       ├── schema.py              # Parameter validation & constraints
│       ├── system_prompt.py       # AI system prompts & instructions
│       └── visualizations.py      # Chart generation (Plotly)
├── tests/                         # Comprehensive test suite (99 tests)
│   ├── test_decline_curves.py
│   ├── test_economics.py
│   ├── test_monte_carlo.py
│   ├── test_drilling_optimizer.py
│   ├── test_heuristic_optimizer.py
│   ├── test_integration_phase*.py
│   └── test_ui_app.py
└── local/                         # Original work package optimizer (not published)
    └── main.py                    # Original construction/pipeline scheduler
```

## Key Components

### Core Optimization (`heuristic_optimizer.py`)
- Implements a guided random search algorithm
- Perturbs parameters to explore solution space
- Evaluates scenarios using economic models
- Tracks optimization history and convergence

### Scheduling Engine (`drilling_optimizer.py`)
- Uses Google OR-Tools CP-SAT solver
- Models drilling as a constraint satisfaction problem
- Handles rig allocation and timeline optimization
- Respects operational constraints (budget, permits, rig availability)

### Economic Modeling (`economics.py`)
- Calculates Net Present Value (NPV) and Internal Rate of Return (IRR)
- Models Texas-specific economics (royalties, severance tax)
- Handles time value of money with configurable discount rates
- Computes payback periods and investment efficiency metrics

### Production Modeling (`decline_curves.py`)
- Implements Arps hyperbolic decline equation
- Models oil & gas production over time
- Texas-specific initial production rates and decline parameters
- Generates monthly production forecasts

### Risk Analysis (`monte_carlo.py`)
- Optional Monte Carlo simulation for uncertainty
- Models oil price volatility and cost variations
- Calculates P10/P50/P90 confidence intervals
- Provides risk-adjusted metrics

### AI Integration (`gemini_client.py`)
- Natural language interface using Gemini 2.5 Pro or Flash models
- Selectable model via UI dropdown (Pro for accuracy, Flash for speed)
- Function calling for parameter adjustment
- Context-aware responses about optimization state
- Helps users explore scenarios through conversation

## Development

### Running Tests
```bash
# Run all tests
python -m pytest -v

# Run specific test file
python -m pytest tests/test_economics.py -v

# Run with coverage
python -m pytest --cov=src/oil_gas_optimizer -v
```

### Code Style
The project follows Python best practices:
- Type hints throughout
- Comprehensive docstrings
- Test-Driven Development (TDD)
- SOLID principles
- Clear separation of concerns

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow validation
- **UI Tests**: Component interaction verification
- **Total**: 99 tests ensuring reliability

## Performance Characteristics

- **Optimization Speed**: ~100ms per trial (without Monte Carlo)
- **Typical Run**: 20-30 trials to find good solutions
- **Monte Carlo**: Adds 2-5 seconds per trial when enabled
- **UI Responsiveness**: Real-time updates via async execution

## Future Enhancements

### Planned Features
- Database persistence for optimization history
- Multi-user collaboration capabilities
- Advanced 3D reservoir visualizations
- Vizier integration for Bayesian optimization
- Extended geographic coverage beyond Texas

### Monte Carlo Risk Assessment
The framework supports full Monte Carlo simulation but runs in simplified mode by default for performance. To enable:

```python
# In DrillingScenario.from_knobs():
run_monte_carlo=True,  # Enable Monte Carlo
mc_simulations=1000    # Number of simulations
```

This provides:
- P10/P50/P90 NPV distributions
- Probability of positive NPV
- Value at Risk (VaR) metrics
- Sensitivity analysis

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired and expanded from the codebase available at: https://github.com/ssizan01/vizier_or_tools
- Built on the architecture of the original work package optimizer
- Uses Google OR-Tools for constraint programming
- Powered by Gemini 2.5 Pro/Flash for AI capabilities
- NiceGUI for reactive web interface
- Plotly for interactive visualizations