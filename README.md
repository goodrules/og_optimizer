# Oil & Gas Field Development Optimizer

A Python-based optimization tool for oil and gas field development planning with AI-powered assistance. This application helps optimize drilling programs for Texas oil fields to maximize Net Present Value (NPV) while managing risk.

## Features

- **Economic Optimization**: Maximize NPV across multiple drilling locations
- **Risk Assessment**: Calculate and visualize risk scores based on portfolio characteristics
- **Texas Focus**: Pre-configured for 5 Texas leases in Permian and Delaware basins
- **AI Assistant**: Integrated Gemini 2.5 Pro for natural language interaction and parameter optimization
- **Real-time Visualization**: Interactive charts showing production forecasts, economics, and optimization history
- **Constraint-based Planning**: Respect budget limits, rig availability, and operational constraints

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

3. **Optimization**: Click "Optimize Development" to run the heuristic optimizer

### Available Leases

- **MIDLAND_A**: Permian Basin, up to 28 wells
- **MARTIN_B**: Permian Basin, up to 15 wells
- **REEVES_C**: Delaware Basin, up to 32 wells
- **LOVING_D**: Delaware Basin, up to 22 wells
- **HOWARD_E**: Permian Basin, up to 12 wells

## Development

### Running Tests
```bash
python -m pytest -v
```

### Project Structure
```
src/oil_gas_optimizer/
├── ui_app.py              # Main UI application
├── heuristic_optimizer.py  # Optimization engine
├── drilling_optimizer.py   # OR-Tools scheduling
├── economics.py           # NPV/IRR calculations
├── decline_curves.py      # Production modeling
├── monte_carlo.py         # Uncertainty analysis
├── gemini_client.py       # AI assistant integration
└── schema.py              # Parameter validation
```

## License

MIT License - see LICENSE file for details