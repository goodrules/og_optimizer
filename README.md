# Oil & Gas Field Development Optimizer

A Python-based optimization tool for oil and gas field development planning with AI-powered assistance. This application helps optimize drilling programs for Texas oil fields to maximize Net Present Value (NPV) while managing risk and uncertainty.

## Overview

The Oil & Gas Field Development Optimizer is a sophisticated platform that combines **operations research**, **AI optimization**, and **economic simulation** to revolutionize field development planning. Originally a general work package optimizer, it has evolved into a specialized tool for optimizing drilling programs across multiple Texas oil leases using advanced constraint programming, dual optimization approaches (heuristic + Bayesian), and modern AI capabilities for enhanced user interaction.

## Features

### Core Optimization Capabilities
- **Dual Optimization Methods**: Choose between fast heuristic search or advanced Bayesian optimization (Google Vizier)
- **Economic Optimization**: Maximize NPV across multiple drilling locations with intelligent parameter exploration
- **Constraint Programming**: Uses OR-Tools CP-SAT for drilling schedule optimization with operational constraints
- **Parameter Locking**: Lock specific parameters for "what-if" analysis while optimizing others
- **Real-time Results**: Complete optimization runs in minutes instead of weeks

### Risk & Uncertainty Management
- **Monte Carlo Simulation**: Comprehensive uncertainty analysis for oil prices and cost variations
- **Risk Quantification**: P10/P50/P90 NPV scenarios with probability analysis
- **Portfolio Risk Assessment**: Calculate and visualize risk scores based on portfolio characteristics

### AI-Powered Interface
- **AI Assistant**: Integrated Gemini 2.5 Pro/Flash models for natural language interaction and parameter optimization
- **Intelligent Parameter Setting**: Use natural language to configure optimization scenarios
- **Context-Aware Responses**: AI understands current optimization state and provides relevant insights

### Visualization & Analysis
- **Interactive Dashboards**: Real-time charts showing production forecasts, economics, and optimization history
- **Progress Tracking**: Visual convergence monitoring and optimization trajectory analysis
- **Sensitivity Analysis**: Oil price sensitivity and parameter impact visualization

### Domain Expertise
- **Texas Focus**: Pre-configured for 5 Texas leases in Permian and Delaware basins
- **Industry-Specific Modeling**: Texas royalties, severance tax, decline curves, and operational constraints
- **Proven Results**: Demonstrated 5.5x NPV improvements ($25MM → $142MM) in testing scenarios

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (NiceGUI) - Async Interface            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Control Panel  │  │  Visualizations │  │  AI Assistant  │  │
│  │  - Economics    │  │  - Timeline     │  │  - Gemini 2.5  │  │
│  │  - Drilling     │  │  - Production   │  │  - Chat UI     │  │
│  │  - Leases       │  │  - Sensitivity  │  │  - Streaming   │  │
│  │  - Method Sel.  │  │  - 3D History   │  │                │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬───────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
┌───────────┴────────────────────┴────────────────────┴───────────┐
│                 Unified Optimization Manager                     │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  Heuristic Optimizer    │  │  Vizier Bayesian Optimizer  │  │
│  │  - Fast exploration     │  │  - Google Cloud Vizier      │  │
│  │  - Perturb parameters   │  │  - Intelligent exploration  │  │
│  │  - Convergence tracking │  │  - Superior convergence     │  │
│  │  - Locked params        │  │  - Locked params            │  │
│  └───────────┬─────────────┘  └──────────┬──────────────────┘  │
└──────────────┼────────────────────────────┼─────────────────────┘
               │                            │
┌──────────────┴────────────────────────────┴─────────────────────┐
│              OR-Tools Constraint Programming Engine              │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  CP-SAT Scheduler       │  │  Parameter Optimization     │  │
│  │  - Rig allocation       │  │  - Scenario evaluation      │  │
│  │  - Timeline optimization│  │  - Constraint satisfaction  │  │
│  │  - Resource constraints │  │  - Budget adherence         │  │
│  └───────────┬─────────────┘  └──────────┬──────────────────┘  │
└──────────────┼────────────────────────────┼─────────────────────┘
               │                            │
┌──────────────┴────────────────────────────┴─────────────────────┐
│                Domain Models & Risk Analytics                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Economics    │  │ Decline      │  │ Monte Carlo        │   │
│  │ - NPV/IRR    │  │ Curves       │  │ - Price volatility │   │
│  │ - Payback    │  │ - Hyperbolic │  │ - Cost uncertainty │   │
│  │ - Texas tax  │  │ - Arps eqn   │  │ - P10/P50/P90      │   │
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

3. Set up Google Cloud for AI features (required for AI assistant and Vizier optimization):
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

4. Install optional dependencies for advanced optimization:
   ```bash
   # For Google Vizier Bayesian optimization
   pip install google-cloud-aiplatform
   ```

## Usage

Run the application:
```bash
python main.py
```

The application will open in your web browser at http://localhost:8080

### Using the Interface

1. **Optimization Method Selection**: Choose your optimization approach:
   - **Heuristic (Fast)**: Quick guided random search for rapid analysis
   - **Vizier Bayesian (Smart)**: Advanced AI optimization for superior results

2. **Parameter Controls**: Adjust parameters using sliders and controls:
   - Economic assumptions (oil price, discount rate, contingency, budget)
   - Drilling execution (rigs, mode, permits)
   - Well selection per lease
   - **Parameter Locking**: Click lock icons to fix specific parameters during optimization

3. **AI Assistant**: Use natural language to:
   - Set parameters: "Set up a conservative development plan"
   - Ask questions: "What's the current NPV and risk?"
   - Request optimization: "Optimize for maximum production"
   - Select AI model: Choose between Gemini 2.5 Pro (more accurate) or Flash (faster responses)

4. **Optimization Execution**: 
   - Click "Optimize Development" to start optimization
   - Watch real-time progress with async execution
   - View method-specific progress indicators
   - Automatic fallback from Vizier to Heuristic if needed

5. **Results Analysis**:
   - Interactive visualizations update in real-time
   - Compare optimization methods and results
   - Analyze parameter evolution across trials
   - Enable Monte Carlo simulation for risk assessment

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
├── CLAUDE.md                      # Project instructions and architecture guide
├── components/                    # Main source code (refactored from src/)
│   ├── ui_app.py                  # Main UI application with async optimization
│   ├── optimization_manager.py    # Unified interface for multiple optimization methods
│   ├── heuristic_optimizer.py     # Fast guided random search optimization
│   ├── vizier_optimizer.py        # Google Cloud Vizier Bayesian optimization
│   ├── drilling_optimizer.py      # OR-Tools CP-SAT scheduling & constraints
│   ├── economics.py               # NPV/IRR calculations with Texas economics
│   ├── decline_curves.py          # Arps hyperbolic production decline modeling
│   ├── monte_carlo.py             # Risk analysis & uncertainty modeling
│   ├── gemini_client.py           # AI assistant (Gemini 2.5 Pro/Flash)
│   ├── data_model.py              # Texas lease & well data structures
│   ├── schema.py                  # Parameter validation & constraints
│   ├── system_prompt.py           # AI system prompts & instructions
│   └── visualizations.py          # Plotly interactive charts
├── local/                         # Development & original code
│   ├── tests/                     # Test suite (99 tests)
│   │   ├── test_economics.py
│   │   ├── test_monte_carlo.py
│   │   ├── test_drilling_optimizer.py
│   │   ├── test_heuristic_optimizer.py
│   │   ├── test_vizier_optimizer.py
│   │   └── test_integration_*.py
│   ├── vizier-plan.md             # Vizier integration planning document
│   ├── demo-narrative.md          # Business demo narrative for energy industry
│   ├── vizier_demo.py             # Vizier demonstration script
│   └── main.py                    # Original work package optimizer
```

## Key Components

### Unified Optimization Management (`optimization_manager.py`)
- Provides seamless interface between multiple optimization methods
- Automatic fallback from Vizier to Heuristic when needed
- Handles locked parameter constraints across both methods
- Method availability checking and setup validation

### Fast Heuristic Optimization (`heuristic_optimizer.py`)
- Implements guided random search algorithm with locked parameter support
- Perturbs unlocked parameters to explore solution space
- Evaluates scenarios using economic models and constraints
- Tracks optimization history and convergence (typically 10-20 trials)

### Advanced Bayesian Optimization (`vizier_optimizer.py`)
- Google Cloud Vizier integration for intelligent parameter exploration
- Excludes locked parameters from optimization search space
- Superior convergence with fewer evaluations (typically 25-50 trials)
- Machine learning-driven parameter space exploration

### Constraint Programming Engine (`drilling_optimizer.py`)
- Uses Google OR-Tools CP-SAT solver for drilling schedules
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
- Function calling for parameter adjustment and scenario configuration
- Context-aware responses about optimization state and results
- Streaming responses for real-time interaction
- Helps users explore scenarios through conversational interface

## Development

### Running Tests
```bash
# Run all tests
python -m pytest local/tests/ -v

# Run specific test file
python -m pytest local/tests/test_economics.py -v

# Run with coverage
python -m pytest local/tests/ --cov=components -v

# Test specific optimization methods
python -m pytest local/tests/test_vizier_optimizer.py -v
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

### Optimization Performance
- **Heuristic Speed**: ~100ms per trial (without Monte Carlo)
- **Vizier Speed**: ~2-5 seconds per trial (includes cloud communication)
- **Typical Convergence**: Heuristic 10-20 trials, Vizier 25-50 trials
- **Monte Carlo**: Adds 2-5 seconds per trial when enabled
- **Demonstrated Results**: 5.5x NPV improvement (Vizier vs manual planning)

### UI & System Performance
- **Real-time Updates**: Async execution prevents UI blocking
- **Connection Stability**: Periodic progress updates maintain connectivity
- **Background Processing**: Long-running optimizations don't freeze interface
- **Cross-method Compatibility**: Seamless switching between optimization approaches

## Future Enhancements

### Planned Features
- **Database Persistence**: Save optimization history and portfolio analysis
- **Multi-user Collaboration**: Shared optimization scenarios and team workflows
- **Advanced Visualizations**: 3D reservoir modeling and enhanced analytics
- **Extended Geography**: Support for additional oil & gas regions beyond Texas
- **Custom Constraints**: User-defined operational and financial constraints
- **Portfolio Optimization**: Multi-field development strategy optimization
- **Real-time Data Integration**: Live price feeds and market data integration

### Monte Carlo Risk Assessment
The framework supports comprehensive Monte Carlo simulation for uncertainty analysis. Enable through the UI:

1. **Toggle Monte Carlo**: Use the "Use Monte Carlo Simulation" switch
2. **Set Simulations**: Adjust the number of simulations (10-1000)
3. **Run Optimization**: Monte Carlo will be applied to all trials

**Results Include**:
- **P10/P50/P90 NPV distributions**: Risk-adjusted value ranges
- **Probability Analysis**: Chance of positive returns and target achievement
- **Value at Risk (VaR)**: Downside risk quantification
- **Sensitivity Analysis**: Parameter impact on uncertainty

**Performance Impact**: Adds 2-5 seconds per optimization trial depending on simulation count.

## License

MIT License - see LICENSE file for details

## Acknowledgments

### Technology Stack
- **Google OR-Tools**: Constraint programming and optimization engine
- **Google Cloud Vizier**: Advanced Bayesian optimization via AI Platform
- **Gemini 2.5 Pro/Flash**: AI assistant and natural language processing
- **NiceGUI**: Reactive web interface framework
- **Plotly**: Interactive data visualizations and charts

### Inspiration & References
- Inspired and expanded from: https://github.com/ssizan01/vizier_or_tools
- Built on the architecture of the original work package optimizer
- Texas oil & gas economics modeling based on industry best practices
- Optimization approaches informed by operations research literature

### Development
- Test-driven development with comprehensive test suite
- Follows Python best practices and SOLID principles
- Async-first UI design for responsive user experience
- Modular architecture enabling extensibility and maintenance