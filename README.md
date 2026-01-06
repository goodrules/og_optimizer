# Oil & Gas Field Development Optimizer

An AI-powered optimization platform for maximizing Net Present Value (NPV) in oil and gas drilling programs. Combines advanced Bayesian optimization, constraint programming, and natural language AI assistance to revolutionize field development planning for Texas oil fields.

**Demo Scenario**: Optimize drilling schedules across 5 Texas leases (Permian and Delaware basins) to maximize NPV while managing operational constraints, risk, and uncertainty. Demonstrated results: 5.5x NPV improvement ($25MM → $142MM).

**Created by**: [@goodrules](https://github.com/goodrules)

## Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud account (free tier works for testing)
- Modern web browser

### Installation

1. Clone and install dependencies:
```bash
git clone https://github.com/goodrules/og_optimizer.git
cd og_optimizer
pip install -r requirements.txt
```

2. Set up Google Cloud (Required for AI features):

**Enable Required APIs:**
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a new project (or select existing)
- Enable these APIs:
  - **Vertex AI API** (for Gemini 2.5 Pro/Flash AI assistant)
  - **Vertex AI Vizier API** (for Bayesian optimization)

**Quick Links:**
- [Enable Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/start/cloud-environment)
- [Enable Vizier](https://docs.cloud.google.com/vertex-ai/docs/vizier/using-vizier)

**Authenticate:**
```bash
gcloud auth application-default login
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and set your GCP_PROJECT_ID
```

Example `.env`:
```
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
```

4. Run the application:
```bash
python main.py
```

The app will open at http://localhost:8080

### Troubleshooting
- **"Permission denied" errors**: Ensure APIs are enabled and authentication is complete
- **"Project not found"**: Verify GCP_PROJECT_ID in `.env` matches your Google Cloud project
- **Vizier errors**: Fall back to Heuristic optimization method (no cloud connection needed)

## Features

### Dual Optimization Methods
- **Heuristic (Fast)**: Guided random search, ~100ms per trial, 10-20 trials typical
- **Vizier Bayesian (Smart)**: Google Cloud AI optimization, **40+ seconds per trial**, 25-50 trials typical, superior convergence

### AI-Powered Interface
- **Gemini 2.5 Integration**: Natural language parameter control and analysis
- **Interactive Chat**: "Set conservative plan" or "What's the current NPV?"
- **Model Selection**: Choose Pro (accuracy) or Flash (speed)

### Risk & Uncertainty Management
- **Monte Carlo Simulation**: P10/P50/P90 NPV scenarios
- **Sensitivity Analysis**: Oil price and parameter impact
- **Risk Scoring**: Portfolio risk assessment

### Economic Optimization
- **Constraint Programming**: OR-Tools CP-SAT for drilling schedules
- **Texas Economics**: Royalties, severance tax, decline curves
- **Real-time Results**: Interactive dashboards and progress tracking

## Using the Application

### 1. Choose Optimization Method
Select your approach in the control panel:
- **Heuristic**: Fast prototyping and analysis
- **Vizier**: Production runs with superior results (requires cloud setup)

**Note**: Vizier optimization runs in the cloud and takes 40+ seconds per trial. You'll see periodic progress updates. For quick experimentation, use Heuristic method.

### 2. Configure Parameters
Adjust using sliders or AI assistant:
- **Economics**: Oil price ($40-120/bbl), discount rate, contingency, budget
- **Drilling**: Number of rigs, simultaneous drilling, permit delays
- **Wells**: Select wells per lease (up to 109 total wells across 5 leases)
- **Lock Parameters**: Click lock icons to fix specific values during optimization

### 3. Use AI Assistant
Natural language control with Gemini:
- "Set up a conservative development plan"
- "Optimize for maximum NPV with 3 rigs"
- "What's the risk score?"
- Switch between Gemini 2.5 Pro and Flash models

### 4. Run Optimization
- Click "Optimize Development"
- Watch real-time progress and convergence
- View trials in navigation dropdown
- Explore optimization trajectory

### 5. Analyze Results
- **Production Timeline**: Wells drilled per lease over time
- **Economics**: NPV, IRR, payback period
- **Optimization History**: Parameter evolution across trials
- **Risk Analysis**: Enable Monte Carlo for uncertainty quantification

## Demo Scenario

### Texas Oil Field Portfolio
**5 Leases** across Permian and Delaware basins:
- **MIDLAND_A**: Permian Basin, up to 28 wells
- **MARTIN_B**: Permian Basin, up to 15 wells
- **REEVES_C**: Delaware Basin, up to 32 wells
- **LOVING_D**: Delaware Basin, up to 22 wells
- **HOWARD_E**: Permian Basin, up to 12 wells

**Total**: 109 potential wells to optimize

### Example Results
- **Baseline**: Manual planning → $25MM NPV
- **Optimized**: Vizier optimization → $142MM NPV
- **Improvement**: 5.5x increase in value
- **Method**: Intelligent well selection, timing, and rig allocation

### Key Insights Demonstrated
- Optimal rig allocation across leases
- Strategic well sequencing for NPV maximization
- Risk-adjusted portfolio optimization
- Budget constraint management
- Permit delay impact analysis

## Project Structure

```
og_optimizer/
├── main.py                      # Application entry point
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── components/
│   ├── ui_app.py               # NiceGUI web interface
│   ├── optimization_manager.py # Unified optimization interface
│   ├── heuristic_optimizer.py  # Fast guided search
│   ├── vizier_optimizer.py     # Bayesian optimization
│   ├── drilling_optimizer.py   # OR-Tools CP-SAT scheduler
│   ├── economics.py            # NPV/IRR calculations
│   ├── decline_curves.py       # Production modeling
│   ├── monte_carlo.py          # Risk analysis
│   ├── gemini_client.py        # AI assistant
│   ├── data_model.py           # Lease data
│   └── visualizations.py       # Plotly charts
└── README.md
```

## Technology Stack

- **Google OR-Tools**: Constraint programming and scheduling
- **Google Cloud Vizier**: Advanced Bayesian optimization
- **Gemini 2.5 Pro/Flash**: AI assistant via Vertex AI
- **NiceGUI**: Reactive web framework
- **Plotly**: Interactive visualizations

## Performance Expectations

### Optimization Speed
- **Heuristic**: ~100ms per trial (local computation)
- **Vizier**: 40+ seconds per trial (cloud-based Bayesian optimization with ML models)
- **Monte Carlo**: +2-5 seconds per trial when enabled

### Typical Runs
- **Heuristic**: 10-20 trials, completes in seconds
- **Vizier**: 25-50 trials, completes in 15-30+ minutes (better results)

### UI Performance
- **Async Execution**: Non-blocking optimization runs
- **Real-time Updates**: Progress indicators maintain connection
- **Responsive Interface**: Continue exploring while optimization runs

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=components -v
```

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- 99 tests ensuring reliability
- SOLID principles

## Advanced Topics

### Monte Carlo Risk Assessment
Enable in UI for uncertainty analysis:
1. Toggle "Use Monte Carlo Simulation"
2. Set number of simulations (10-1000)
3. Run optimization

**Results**: P10/P50/P90 distributions, probability analysis, Value at Risk

### Parameter Locking
Lock specific parameters to run "what-if" scenarios:
- Click lock icon next to any parameter
- Locked parameters are excluded from optimization
- Useful for sensitivity analysis and constrained scenarios

### Architecture

The system uses a multi-layered architecture:

1. **UI Layer**: NiceGUI async interface with real-time updates
2. **Optimization Layer**: Unified manager with dual methods (Heuristic + Vizier)
3. **Constraint Solver**: OR-Tools CP-SAT for feasible schedules
4. **Domain Models**: Economics, production, and risk analytics
5. **AI Layer**: Gemini integration for natural language interaction

## License

MIT License - see LICENSE file for details

## Author

Created by **Mike Goodman** ([@goodrules](https://github.com/goodrules))

## Acknowledgments

- Inspired by: https://github.com/ssizan01/vizier_or_tools
- Built with Google OR-Tools, Vertex AI, and NiceGUI
- Texas oil & gas economics based on industry best practices

---

**Getting Help**: For issues or questions, please open an issue on GitHub.

**Contributing**: Contributions welcome! Please follow the existing code style and include tests.
