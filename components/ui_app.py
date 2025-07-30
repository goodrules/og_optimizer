"""
Oil & Gas Field Development Optimizer UI
Adapted from the work package optimizer UI in local/main.py
"""
import math
import random
import asyncio
import concurrent.futures
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import run, ui, app

from .heuristic_optimizer import (
    OptimizationKnobs,
    HeuristicOptimizer,
    evaluate_scenario,
    perturb_knobs,
    worst_case_knobs
)
from .optimization_manager import OptimizationManager, get_available_methods
from .economics import WellEconomics, EconomicParameters
from .data_model import generate_texas_wells
from .monte_carlo import MonteCarloParameters
from .gemini_client import get_gemini_client
from .schema import apply_params_to_ui

# Constants
IMPROVE_TRIALS = 20  # Number of trials for visual convergence
MC_SIMS = 100
DEFAULT_BUDGET = 100_000_000  # $100MM development budget

# Texas lease configurations
TEXAS_LEASES = {
    "MIDLAND_A": {"basin": "Permian", "wells": 28, "ip_base": 1100, "capex": 7_500_000},
    "MARTIN_B": {"basin": "Permian", "wells": 15, "ip_base": 950, "capex": 7_000_000},
    "REEVES_C": {"basin": "Delaware", "wells": 32, "ip_base": 1300, "capex": 8_500_000},
    "LOVING_D": {"basin": "Delaware", "wells": 22, "ip_base": 1050, "capex": 7_800_000},
    "HOWARD_E": {"basin": "Permian", "wells": 12, "ip_base": 850, "capex": 6_800_000}
}


def create_well_portfolio() -> List[WellEconomics]:
    """Create portfolio of Texas wells."""
    wells = []
    for lease_id, config in TEXAS_LEASES.items():
        wells.extend(generate_texas_wells(
            lease_id=lease_id,
            n_wells=config["wells"],
            base_ip=config["ip_base"],
            base_capex=config["capex"]
        ))
    return wells


def get_best_trial_idx(client: Dict) -> int:
    """Find trial with highest NPV."""
    if not client["trials"]:
        return -1
    best_idx = 0
    best_npv = client["trials"][0]["metrics"]["total_npv"]
    for i, trial in enumerate(client["trials"]):
        if trial["metrics"]["total_npv"] > best_npv:
            best_npv = trial["metrics"]["total_npv"]
            best_idx = i
    return best_idx


def get_selected_trial_idx(client: Dict) -> int:
    """Get currently selected trial index."""
    if not client["trials"]:
        return -1
    
    # Use selected trial if set and valid
    selected_idx = client.get("selected_trial_idx", -1)
    if 0 <= selected_idx < len(client["trials"]):
        return selected_idx
    
    # Default to latest trial
    return len(client["trials"]) - 1


def update_trial_display(client: Dict) -> None:
    """Update all visualizations for currently selected trial."""
    idx = get_selected_trial_idx(client)
    if idx >= 0:
        update_metrics_card(idx, client)
        update_drilling_timeline(idx, client)
        update_production_forecast(idx, client)
        update_economics_chart(idx, client)
    
    # Always update optimization history and trajectory (show all trials)
    update_optimization_history(client)
    update_trajectory_chart(client)
    
    # Update trial selector UI
    update_trial_selector(client)


def update_trial_selector(client: Dict) -> None:
    """Update trial selector dropdown and navigation buttons."""
    if "trial_selector" not in client:
        return
    
    trials = client["trials"]
    if not trials:
        client["trial_selector"].set_options([])
        client["trial_selector"].set_enabled(False)
        if "best_trial_btn" in client:
            client["best_trial_btn"].set_enabled(False)
        if "latest_trial_btn" in client:
            client["latest_trial_btn"].set_enabled(False)
        return
    
    # Build selector options
    best_idx = get_best_trial_idx(client)
    latest_idx = len(trials) - 1
    
    options = {}
    for i, trial in enumerate(trials):
        npv = trial["metrics"]["total_npv"] / 1e6
        
        # Add status indicators
        indicators = []
        if i == best_idx:
            indicators.append("⭐")
        if i == latest_idx:
            indicators.append("🆕")
        
        indicator_str = " " + "".join(indicators) if indicators else ""
        label = f"Trial {i+1} - ${npv:.1f}MM{indicator_str}"
        
        options[i] = label
    
    client["trial_selector"].set_options(options)
    client["trial_selector"].set_enabled(True)
    
    # Set current selection
    selected_idx = get_selected_trial_idx(client)
    client["trial_selector"].set_value(selected_idx)
    
    # Update button states
    if "best_trial_btn" in client:
        best_npv = trials[best_idx]["metrics"]["total_npv"] / 1e6
        client["best_trial_btn"].set_text(f"⭐ Best (${best_npv:.1f}MM)")
        client["best_trial_btn"].set_enabled(True)
    
    if "latest_trial_btn" in client:
        client["latest_trial_btn"].set_enabled(True)


def select_trial(client: Dict, trial_idx: int) -> None:
    """Select a specific trial for display."""
    if not client["trials"] or trial_idx < 0 or trial_idx >= len(client["trials"]):
        return
    
    client["selected_trial_idx"] = trial_idx
    client["auto_follow_latest"] = False  # Disable auto-follow when user selects
    update_trial_display(client)


def select_best_trial(client: Dict) -> None:
    """Select the trial with highest NPV."""
    best_idx = get_best_trial_idx(client)
    if best_idx >= 0:
        select_trial(client, best_idx)


def select_latest_trial(client: Dict) -> None:
    """Select the latest trial and resume auto-follow."""
    if client["trials"]:
        latest_idx = len(client["trials"]) - 1
        client["selected_trial_idx"] = latest_idx
        client["auto_follow_latest"] = True  # Re-enable auto-follow
        update_trial_display(client)


def add_trial(knobs: OptimizationKnobs, metrics: Dict, client: Dict) -> None:
    """Add optimization trial to history and update UI."""
    trial = {
        "knobs": knobs,
        "metrics": metrics,
        "timestamp": datetime.now()
    }
    client["trials"].append(trial)
    
    # Auto-follow latest trial if enabled
    if client.get("auto_follow_latest", True):
        client["selected_trial_idx"] = len(client["trials"]) - 1
    
    # Update all visualizations
    update_trial_display(client)


def update_metrics_card(idx: int, client: Dict) -> None:
    """Update the metrics display card."""
    if idx < 0 or idx >= len(client["trials"]):
        return
    
    m = client["trials"][idx]["metrics"]
    k = client["trials"][idx]["knobs"]
    
    # Calculate improvements
    initial_npv = client.get("initial_npv", 0)
    if initial_npv == 0 and idx == 0:
        client["initial_npv"] = m["total_npv"]
        initial_npv = m["total_npv"]
    
    npv_improvement = m["total_npv"] - initial_npv
    color = "text-green-600" if npv_improvement > 0 else "text-gray-800"
    
    # Optimization method display
    method_used = m.get('optimization_method', 'heuristic')
    method_icon = "🧠" if method_used == "vizier" else "🔍"
    method_color = "text-blue-600" if method_used == "vizier" else "text-green-600"
    method_name = "Vizier Bayesian" if method_used == "vizier" else "Heuristic"
    
    # Trial status indicators
    best_idx = get_best_trial_idx(client)
    latest_idx = len(client["trials"]) - 1
    selected_idx = get_selected_trial_idx(client)
    
    trial_status = ""
    trial_badges = []
    if idx == best_idx:
        trial_badges.append('<span class="inline-block bg-amber-100 text-amber-800 text-xs px-2 py-1 rounded-full">⭐ BEST</span>')
    if idx == latest_idx:
        trial_badges.append('<span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">🆕 LATEST</span>')
    if idx == selected_idx and len(client["trials"]) > 1:
        trial_badges.append('<span class="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">👁️ VIEWING</span>')
    
    if trial_badges:
        trial_status = f'<div class="mb-2 flex gap-1">{"".join(trial_badges)}</div>'
    
    # Build stats display with consistent layout
    monte_carlo_row = ""
    if m.get('run_monte_carlo', False) and 'p10_npv' in m:
        # Add Monte Carlo row only when available
        monte_carlo_row = f"""
          <!-- Row 3 - Monte Carlo Results -->
          <div><div class="text-gray-500 text-sm">P10 NPV (Downside)</div>
               <div class="text-xl font-bold text-red-600">${m['p10_npv']/1e6:.1f}MM</div></div>
               
          <div><div class="text-gray-500 text-sm">P50 NPV (Expected)</div>
               <div class="text-xl font-bold">${m['total_npv']/1e6:.1f}MM</div></div>
               
          <div><div class="text-gray-500 text-sm">P90 NPV (Upside)</div>
               <div class="text-xl font-bold text-green-600">${m['p90_npv']/1e6:.1f}MM</div></div>
               
          <div><div class="text-gray-500 text-sm">Prob. NPV > 0</div>
               <div class="text-xl font-bold">{m['probability_positive']:.0%}</div>
               <div class="text-xs text-gray-600">{m.get('n_simulations', 100)} simulations</div></div>
        """
    
    client["stats_html"].content = f"""
    {trial_status}
    <div class="grid grid-cols-4 gap-x-4 gap-y-2">
      <!-- Row 1 -->
      <div><div class="text-gray-500 text-sm">Total NPV</div>
           <div class="text-2xl font-bold">${m['total_npv']/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">Total CAPEX</div>
           <div class="text-xl font-bold text-indigo-600">${m['total_capex']/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">NPV Improvement</div>
           <div class="text-2xl font-bold {color}">${npv_improvement/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">Wells Selected</div>
           <div class="text-2xl font-bold">{m['wells_selected']}</div></div>
           
      <!-- Row 2 -->
      <div><div class="text-gray-500 text-sm">NPV/$ Invested</div>
           <div class="text-2xl font-bold">{m.get('npv_per_dollar', 0):.2f}</div></div>
           
      <div><div class="text-gray-500 text-sm">Avg Production/Well</div>
           <div class="text-2xl font-bold">{m.get('peak_production', 0):.0f} boe/d</div></div>
           
      <div><div class="text-gray-500 text-sm">Trial #</div>
           <div class="text-2xl font-bold">{idx + 1}</div></div>
           
      <div><div class="text-gray-500 text-sm">Optimization Method</div>
           <div class="text-xl font-bold {method_color}">{method_icon} {method_name}</div></div>
           
      <!-- Risk Score (always display) -->
      <div><div class="text-gray-500 text-sm">Risk Score</div>
           <div class="text-xl font-bold text-orange-600">{m.get('risk_score', 0.5):.1%}</div>
           <div class="text-xs text-gray-600">{'MC Risk' if m.get('run_monte_carlo', False) else 'Basic Risk'}</div></div>
           
      {monte_carlo_row}
    </div>"""


def update_drilling_timeline(idx: int, client: Dict) -> None:
    """Update wells per lease visualization."""
    m = client["trials"][idx]["metrics"]
    k = client["trials"][idx]["knobs"]
    
    fig = go.Figure()
    
    # Create bar chart showing wells drilled per lease
    colors = {
        "LEASE001": "#3b82f6",  # blue - Midland
        "LEASE002": "#ef4444",  # red - Martin
        "LEASE003": "#10b981",  # green - Reeves
        "LEASE004": "#f59e0b",  # amber - Loving
        "LEASE005": "#8b5cf6",  # purple - Howard
        "LEASE006": "#ec4899",  # pink - Dawson
        "LEASE007": "#14b8a6",  # teal - Glasscock
        "LEASE008": "#6366f1",  # indigo - Upton
        "LEASE009": "#84cc16",  # lime - Reagan
        "LEASE010": "#f97316"   # orange - Crockett
    }
    
    # Prepare data for bar chart
    leases = []
    wells = []
    basins = []
    costs = []
    
    for lease_id in sorted(k.wells_per_lease.keys()):
        well_count = k.wells_per_lease[lease_id]
        if lease_id in TEXAS_LEASES:
            lease_info = TEXAS_LEASES[lease_id]
            leases.append(f"{lease_id}<br>({lease_info['basin']})")
            wells.append(well_count)
            basins.append(lease_info['basin'])
            # Estimate cost based on basin (6 years production)
            if lease_info['basin'] == 'Midland':
                cost_per_well = 5.5  # $5.5M per well
            elif lease_info['basin'] == 'Delaware':
                cost_per_well = 6.0  # $6M per well
            else:
                cost_per_well = 4.5  # $4.5M per well
            costs.append(well_count * cost_per_well)
    
    # Create bar chart
    fig.add_trace(go.Bar(
        x=leases,
        y=wells,
        marker=dict(
            color=[colors.get(lease_id.split('<')[0], "#666") for lease_id in leases],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{w} wells<br>${c:.1f}M' for w, c in zip(wells, costs)],
        textposition='auto',
        hovertemplate=(
            "%{x}<br>"
            "Wells: %{y}<br>"
            "Est. Cost: $%{customdata:.1f}M<br>"
            "<extra></extra>"
        ),
        customdata=costs
    ))
    
    fig.update_layout(
        height=250,
        yaxis_title="Number of Wells",
        xaxis_title="",
        xaxis=dict(tickangle=-45),
        margin=dict(l=50, r=10, t=10, b=80),
        showlegend=False
    )
    
    client["timeline_chart"].figure = fig
    client["timeline_chart"].update()


def update_production_forecast(idx: int, client: Dict) -> None:
    """Update production forecast chart."""
    m = client["trials"][idx]["metrics"]
    k = client["trials"][idx]["knobs"]
    
    # Generate production forecast
    months = np.arange(0, 120)  # 10 years
    
    # Simplified production calculation
    total_wells = sum(k.wells_per_lease.values())
    if total_wells == 0:
        production = np.zeros_like(months)
    else:
        # Hyperbolic decline
        ip_total = m.get('peak_production', total_wells * 1000)
        di = 0.70  # Annual decline
        b = 1.1
        
        time_years = months / 12.0
        production = ip_total / ((1 + b * di * time_years) ** (1 / b))
    
    fig = go.Figure()
    
    # Base production
    fig.add_trace(go.Scatter(
        x=months,
        y=production,
        mode='lines',
        name='Production',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add P10/P90 bands if Monte Carlo was run
    if hasattr(m, 'p10_production'):
        fig.add_trace(go.Scatter(
            x=months,
            y=production * 1.2,  # Simplified P90
            mode='lines',
            name='P90',
            line=dict(color='#10b981', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=months,
            y=production * 0.8,  # Simplified P10
            mode='lines',
            name='P10',
            line=dict(color='#ef4444', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.1)',
            showlegend=False
        ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Months",
        yaxis_title="Production (boe/d)",
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )
    
    client["production_chart"].figure = fig
    client["production_chart"].update()


def update_economics_chart(idx: int, client: Dict) -> None:
    """Update economics sensitivity chart."""
    m = client["trials"][idx]["metrics"]
    k = client["trials"][idx]["knobs"]
    
    # Oil price sensitivity
    oil_prices = np.linspace(50, 110, 7)
    npvs = []
    
    base_npv = m["total_npv"]
    base_price = k.oil_price_forecast
    
    for price in oil_prices:
        # Simplified NPV scaling with oil price
        price_factor = price / base_price
        npv = base_npv * price_factor ** 0.8  # Non-linear relationship
        npvs.append(npv / 1e6)  # Convert to MM
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=oil_prices,
        y=npvs,
        mode='lines+markers',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ))
    
    # Mark current price
    current_idx = np.argmin(np.abs(oil_prices - base_price))
    fig.add_trace(go.Scatter(
        x=[base_price],
        y=[npvs[current_idx]],
        mode='markers',
        marker=dict(size=12, color='#ef4444'),
        showlegend=False
    ))
    
    fig.update_layout(
        height=250,
        xaxis_title="Oil Price ($/bbl)",
        yaxis_title="NPV ($MM)",
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )
    
    client["economics_chart"].figure = fig
    client["economics_chart"].update()


def update_optimization_history(client: Dict) -> None:
    """Update 3D optimization history plot."""
    if not client["trials"]:
        return
    
    # Extract metrics for all trials
    npvs = []
    risk_scores = []
    efficiencies = []
    
    for trial in client["trials"]:
        m = trial["metrics"]
        npvs.append(m["total_npv"] / 1e6)  # MM
        risk_scores.append(m.get("risk_score", 0.5) * 100)  # Convert to percentage
        efficiencies.append(m.get("npv_per_dollar", 0))
    
    fig = go.Figure()
    
    # Create 3D scatter
    fig.add_trace(go.Scatter3d(
        x=npvs,
        y=risk_scores,
        z=efficiencies,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=list(range(len(npvs))),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Trial")
        ),
        line=dict(color='rgba(100,100,100,0.5)', width=2),
        text=[f"Trial {i+1}" for i in range(len(npvs))],
        hovertemplate=(
            "NPV: $%{x:.1f}MM<br>"
            "Risk Score: %{y:.1f}%<br>"
            "Efficiency: %{z:.2f}<br>"
            "%{text}<extra></extra>"
        )
    ))
    
    # Highlight best solution
    best_idx = np.argmax(npvs)
    fig.add_trace(go.Scatter3d(
        x=[npvs[best_idx]],
        y=[risk_scores[best_idx]],
        z=[efficiencies[best_idx]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        showlegend=False
    ))
    
    fig.update_layout(
        height=400,
        scene=dict(
            xaxis_title="NPV ($MM)",
            yaxis_title="Risk Score (%)",
            zaxis_title="NPV/$ Efficiency",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    client["history_plot"].figure = fig
    client["history_plot"].update()


def update_trajectory_chart(client: Dict) -> None:
    """Update optimization trajectory visualization showing parameter evolution."""
    if not client["trials"] or "trajectory_chart" not in client:
        return
    
    # Extract parameter values across trials
    trials = client["trials"]
    n_trials = len(trials)
    
    # Parameter names for display
    param_names = {
        'oil_price_forecast': 'Oil Price ($/bbl)',
        'hurdle_rate': 'Discount Rate (%)',
        'contingency_percent': 'Contingency (%)',
        'rig_count': 'Rigs',
        'drilling_mode': 'Drilling Mode',
        'wells_per_lease': 'Total Wells'
    }
    
    # Prepare data for parallel coordinates
    data = []
    for i, trial in enumerate(trials):
        knobs = trial["knobs"]
        row = {
            'Trial': i + 1,
            'Oil Price ($/bbl)': knobs.oil_price_forecast,
            'Discount Rate (%)': knobs.hurdle_rate * 100,
            'Contingency (%)': knobs.contingency_percent * 100,
            'Rigs': knobs.rig_count,
            'Drilling Mode': 1 if knobs.drilling_mode == "batch" else 0,
            'Total Wells': sum(knobs.wells_per_lease.values()),
            'NPV ($MM)': trial["metrics"]["total_npv"] / 1e6
        }
        data.append(row)
    
    # Create parallel coordinates plot
    df = pd.DataFrame(data)
    
    # Normalize values for better visualization
    dimensions = []
    for col in df.columns:
        if col == 'Trial':
            dimensions.append(dict(
                range=[1, n_trials],
                label=col,
                values=df[col]
            ))
        elif col == 'Drilling Mode':
            dimensions.append(dict(
                range=[0, 1],
                label=col,
                values=df[col],
                tickvals=[0, 1],
                ticktext=['Continuous', 'Batch']
            ))
        else:
            dimensions.append(dict(
                range=[df[col].min(), df[col].max()],
                label=col,
                values=df[col]
            ))
    
    # Color by NPV
    colors = df['NPV ($MM)'].values
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=colors,
                colorscale='Viridis',
                showscale=True,
                cmin=colors.min(),
                cmax=colors.max(),
                colorbar=dict(
                    title='NPV ($MM)'
                )
            ),
            dimensions=dimensions,
            labelangle=-45
        )
    )
    
    fig.update_layout(
        height=350,
        margin=dict(t=50, l=50, r=50, b=50),
        title=dict(
            text=f"Parameter Evolution Across {n_trials} Trials",
            x=0.5,
            xanchor='center'
        )
    )
    
    client["trajectory_chart"].figure = fig
    client["trajectory_chart"].update()


@ui.page("/")
def main() -> None:
    """Main UI page for Oil & Gas Field Development Optimizer."""
    client = app.storage.client
    
    # Initialize client storage
    if "trials" not in client:
        client["trials"] = []
    if "best_knobs" not in client:
        client["best_knobs"] = None
    if "best_score" not in client:
        client["best_score"] = float("-inf")
    if "wells" not in client:
        client["wells"] = create_well_portfolio()
    if "selected_trial_idx" not in client:
        client["selected_trial_idx"] = -1
    if "auto_follow_latest" not in client:
        client["auto_follow_latest"] = True
    
    # Header
    with ui.header().classes(replace='row items-center h-16 bg-gray-800'):
        ui.label("🛢️").style('font-size: 2em').tailwind("pr-2")
        ui.label('Oil & Gas Field Development Optimizer').style(
            'color:white;font-size:125%;'
        ).tailwind("px-2.5 pl-4", "font-bold")
        ui.chip('Texas Oil Focus', color="grey").props("outline")
    
    # Main layout
    with ui.column().classes('w-full p-4').style('max-width:1600px;margin:0 auto'):
        
        # AI Assistant Chat - Compact version at top
        with ui.card().classes('w-full p-3 mb-4'):
            # Model selection and AI response area
            with ui.row().classes('w-full items-start gap-3 mb-2'):
                ui.icon("smart_toy", size="sm").classes("text-blue-600 mt-1")
                ui.label("AI:").classes("font-semibold mt-1")
                
                # Chat display area - expandable markdown
                chat_display = ui.markdown("Ask me to set parameters or analyze results...").classes(
                    'flex-1 text-sm text-gray-600'
                ).style('max-height: 200px; overflow-y: auto;')
                
                # Model selector dropdown
                model_selector = ui.select(
                    options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
                    value="gemini-2.5-flash",
                    label="Model"
                ).classes('w-40').props('outlined dense')
            
            # User input line
            with ui.row().classes('w-full items-center gap-2'):
                ui.label("You:").classes("font-semibold text-gray-700")
                
                # Initialize Gemini client with selected model
                try:
                    gemini_client = get_gemini_client(model_selector.value)
                    client["gemini_client"] = gemini_client
                    
                    # Update model when selection changes
                    def on_model_change():
                        client["gemini_client"].set_model(model_selector.value)
                        chat_display.set_content(f"Switched to {model_selector.value}. Ask me to set parameters or analyze results...")
                    
                    model_selector.on('update:model-value', on_model_change)
                    
                    # Input field - full width
                    chat_input = ui.input(
                        placeholder="e.g., 'Set conservative plan' or 'What's the current NPV?'"
                    ).classes('flex-1').props('outlined dense')
                    
                    async def send_message():
                        """Process chat message and update UI with streaming"""
                        message = chat_input.value
                        if not message:
                            return
                        
                        # Clear input immediately
                        chat_input.value = ""
                        
                        # Update Gemini context with current data
                        if client["trials"]:
                            latest_trial = client["trials"][-1]
                            trial_data = {
                                "trial_number": len(client["trials"]),
                                "npv_mm": latest_trial["metrics"]["total_npv"] / 1e6,
                                "capex_mm": latest_trial["metrics"]["total_capex"] / 1e6,
                                "wells_selected": latest_trial["metrics"]["wells_selected"],
                                "risk_score": latest_trial["metrics"]["risk_score"] * 100,
                                "avg_production": latest_trial["metrics"]["peak_production"],
                                "npv_per_dollar": latest_trial["metrics"]["npv_per_dollar"]
                            }
                            
                            current_params = {
                                "oil_price": oil_price.value,
                                "discount_rate": discount_rate.value,
                                "contingency": contingency.value,
                                "budget": budget.value,
                                "rigs": rigs.value,
                                "drilling_mode": drilling_mode.value,
                                "permit_delay": permit_delay.value,
                                "use_monte_carlo": monte_carlo_toggle.value,
                                "monte_carlo_simulations": int(n_simulations.value),
                                "wells_per_lease": {
                                    lease_id: well_sliders[lease_id].value
                                    for lease_id in TEXAS_LEASES
                                }
                            }
                            
                            client["gemini_client"].update_context(trial_data, current_params)
                        
                        # Show user prompt first
                        chat_display.set_content(f"**You:** {message}\n\n**AI:** _Thinking..._")
                        
                        # Define callback for streaming updates
                        async def update_stream(text: str, param_updates: Optional[Dict] = None, is_final: bool = False):
                            """Handle streaming updates from Gemini"""
                            # Update the markdown display with accumulated text
                            chat_display.set_content(f"**You:** {message}\n\n**AI:** {text}")
                            
                            # If this is the final update with parameters
                            if is_final and param_updates:
                                # Apply parameter updates
                                ui_elements = {
                                    "oil_price": oil_price,
                                    "discount_rate": discount_rate,
                                    "contingency": contingency,
                                    "budget": budget,
                                    "rigs": rigs,
                                    "drilling_mode": drilling_mode,
                                    "permit_delay": permit_delay,
                                    "monte_carlo_toggle": monte_carlo_toggle,
                                    "n_simulations": n_simulations,
                                    "well_sliders": well_sliders,
                                    "oil_price_lock": oil_price_lock,
                                    "discount_rate_lock": discount_rate_lock,
                                    "contingency_lock": contingency_lock,
                                    "rigs_lock": rigs_lock,
                                    "drilling_mode_lock": drilling_mode_lock,
                                    "permit_delay_lock": permit_delay_lock,
                                    "well_locks": well_locks,
                                    "well_labels": well_labels
                                }
                                
                                # Apply the updates
                                applied = apply_params_to_ui(param_updates, ui_elements)
                                
                                # Update labels
                                if "oil_price" in applied:
                                    oil_price_label.set_text(f'${oil_price.value}/bbl')
                                if "discount_rate" in applied:
                                    discount_rate_label.set_text(f'{discount_rate.value}%')
                                if "contingency" in applied:
                                    contingency_label.set_text(f'{contingency.value}%')
                                if "budget" in applied:
                                    budget_label.set_text(f'${budget.value}MM')
                                if "rigs" in applied:
                                    rigs_label.set_text(f'{rigs.value} rig{"s" if rigs.value != 1 else ""}')
                                if "permit_delay" in applied:
                                    permit_delay_label.set_text(f'{permit_delay.value} days')
                                
                                # Update well labels
                                for lease_id in TEXAS_LEASES:
                                    if f"wells_{lease_id}" in applied:
                                        count = well_sliders[lease_id].value
                                        well_labels[lease_id].set_text(f'{count} well{"s" if count != 1 else ""}')
                                
                                # Flash the optimize button to indicate parameters changed
                                optimize_button.classes(add='animate-pulse')
                                await asyncio.sleep(2)
                                optimize_button.classes(remove='animate-pulse')
                        
                        try:
                            # Get streaming response from Gemini
                            await client["gemini_client"].process_message_stream(message, update_stream)
                                
                        except Exception as e:
                            chat_display.set_content(f"**You:** {message}\n\n**AI Error:** {str(e)}")
                    
                    send_button = ui.button(icon="send", on_click=send_message).props('round dense flat')
                    chat_input.on('keydown.enter', send_message)
                    
                except Exception as e:
                    chat_display.set_content("**AI Assistant unavailable.** Set GCP_PROJECT_ID and authenticate with gcloud.")
        
        with ui.row().classes('w-full gap-4'):
            
            # LEFT CONTROL PANEL
            with ui.card().classes('p-4').style('width:400px;height:fit-content'):
                with ui.row().classes('w-full items-center justify-between mb-3'):
                    ui.label("Development Parameters").classes("text-lg font-bold")
                    optimize_button = ui.button(
                        "Optimize Development",
                        icon="psychology",
                        color="green"
                    ).classes('px-4')
                
                # Optimization Method Selection
                with ui.column().classes('gap-2 mb-3'):
                    ui.label("Optimization Method").classes("font-semibold text-sm")
                    
                    # Check method availability
                    method_info = get_available_methods()
                    vizier_available = method_info["vizier"]["available"]
                    vizier_setup_ok = method_info["vizier"].get("setup_ok", False) if vizier_available else False
                    
                    # Method options with trial counts
                    method_options = ["Heuristic (Fast) - 10 trials"]
                    if vizier_available:
                        if vizier_setup_ok:
                            method_options.append("Vizier Bayesian (Smart) - 25 trials")
                        else:
                            method_options.append("Vizier (Setup Required) - 25 trials")
                    
                    optimization_method = ui.radio(
                        method_options,
                        value="Heuristic (Fast) - 10 trials"
                    ).props('inline').classes('w-full')
                    
                    # Status and guidance
                    method_status = ui.label("").classes("text-xs text-gray-600")
                    
                    def update_method_status():
                        """Update method status based on current selection."""
                        current_method = optimization_method.value
                        
                        if "Heuristic" in current_method:
                            method_status.set_text("✓ Ready - Fast guided random search (~3-5 seconds)")
                            method_status.classes(replace='text-xs text-green-600')
                        elif "Vizier" in current_method:
                            if not vizier_available:
                                method_status.set_text("⚠ Install: pip install google-cloud-aiplatform")
                                method_status.classes(replace='text-xs text-orange-600')
                            elif not vizier_setup_ok:
                                method_status.set_text("⚠ Configure GCP_PROJECT_ID and authenticate")
                                method_status.classes(replace='text-xs text-orange-600')
                            else:
                                method_status.set_text("✓ Ready - Advanced Bayesian optimization (~2 minutes)")
                                method_status.classes(replace='text-xs text-green-600')
                    
                    # Initial status
                    update_method_status()
                    optimization_method.on('update:model-value', lambda: update_method_status())
                    
                    # Add tooltips via help icon
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('help_outline', size='sm').classes('text-gray-400').tooltip(
                            'Heuristic: Fast exploration, good for quick analysis\n'
                            'Vizier: Smart Bayesian optimization, better convergence'
                        )
                
                # Monte Carlo settings directly under optimize button
                with ui.column().classes('gap-2 mb-3'):
                    with ui.row().classes('w-full items-center gap-2'):
                        monte_carlo_toggle = ui.switch(
                            "Use Monte Carlo Simulation",
                            value=False
                        ).classes('text-sm')
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label("Simulations:").classes("text-sm text-gray-600")
                        n_simulations = ui.number(
                            value=100,
                            min=10,
                            max=1000,
                            step=10,
                            format='%d'
                        ).classes('w-24').props('dense outlined')
                        ui.label("(10-1000)").classes("text-xs text-gray-500")
                    # Disable simulations input when toggle is off
                    monte_carlo_toggle.on('update:model-value', lambda e: n_simulations.set_enabled(e.args))
                    n_simulations.set_enabled(False)  # Start disabled
                
                # Helper function to toggle lock state
                def toggle_lock(lock_icon, control):
                    if lock_icon.name == 'lock_open':
                        lock_icon.name = 'lock'
                        lock_icon.classes(replace='cursor-pointer text-orange-600')
                        control._locked = True
                        if hasattr(control, 'props'):
                            control.props(add='color=orange')
                    else:
                        lock_icon.name = 'lock_open'
                        lock_icon.classes(replace='cursor-pointer text-gray-400')
                        control._locked = False
                        if hasattr(control, 'props'):
                            control.props(remove='color=orange')
                
                with ui.column().classes('gap-2'):
                    # Economic parameters
                    ui.label("Economic Assumptions").classes("font-bold mt-2")
                    
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label("Oil Price ($/bbl)").classes("text-sm flex-1")
                        oil_price_label = ui.label("$80/bbl").classes("text-sm font-semibold text-gray-700")
                        oil_price_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                        oil_price_lock.on('click', lambda: toggle_lock(oil_price_lock, oil_price))
                    oil_price = ui.slider(
                        min=40, max=120, step=5, value=80
                    ).props("thumb-label").classes("w-full")
                    oil_price.on('update:model-value', lambda e: oil_price_label.set_text(f'${e.args}/bbl'))
                    
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label("Discount Rate (%)").classes("text-sm flex-1")
                        discount_rate_label = ui.label("15%").classes("text-sm font-semibold text-gray-700")
                        discount_rate_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                        discount_rate_lock.on('click', lambda: toggle_lock(discount_rate_lock, discount_rate))
                    discount_rate = ui.slider(
                        min=8, max=20, step=1, value=15
                    ).props("thumb-label").classes("w-full")
                    discount_rate.on('update:model-value', lambda e: discount_rate_label.set_text(f'{e.args}%'))
                    
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label("Contingency (%)").classes("text-sm flex-1")
                        contingency_label = ui.label("20%").classes("text-sm font-semibold text-gray-700")
                        contingency_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                        contingency_lock.on('click', lambda: toggle_lock(contingency_lock, contingency))
                    contingency = ui.slider(
                        min=10, max=30, step=1, value=20
                    ).props("thumb-label").classes("w-full")
                    contingency.on('update:model-value', lambda e: contingency_label.set_text(f'{e.args}%'))
                    
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label("CAPEX Budget ($MM)").classes("text-sm flex-1")
                        budget_label = ui.label("$100MM").classes("text-sm font-semibold text-gray-700")
                        # No lock icon for budget - it's always manually set
                    budget = ui.slider(
                        min=50, max=200, step=10, value=100
                    ).props("thumb-label").classes("w-full")
                    budget.on('update:model-value', lambda e: budget_label.set_text(f'${e.args}MM'))
                    
                    ui.separator().classes('my-3')
                    
                    # Drilling execution
                    ui.label("Drilling Execution").classes("font-bold")
                    
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.label("Horizontal Rigs Available").classes("text-sm flex-1")
                        rigs_label = ui.label("2 rigs").classes("text-sm font-semibold text-gray-700")
                        rigs_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                        rigs_lock.on('click', lambda: toggle_lock(rigs_lock, rigs))
                    rigs = ui.slider(
                        min=1, max=5, step=1, value=2
                    ).props("thumb-label").classes("w-full")
                    rigs.on('update:model-value', lambda e: rigs_label.set_text(f'{e.args} rig{"s" if e.args != 1 else ""}'))
                    
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label("Drilling Mode").classes("text-sm flex-1")
                        drilling_mode_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                    drilling_mode = ui.toggle(
                        {"continuous": "Continuous", "batch": "Pad/Batch"},
                        value="continuous"
                    ).classes("w-full")
                    drilling_mode_lock.on('click', lambda: toggle_lock(drilling_mode_lock, drilling_mode))
                    
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label("Permit Delay (days)").classes("text-sm flex-1")
                        permit_delay_label = ui.label("30 days").classes("text-sm font-semibold text-gray-700")
                        permit_delay_lock = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                        permit_delay_lock.on('click', lambda: toggle_lock(permit_delay_lock, permit_delay))
                    permit_delay = ui.slider(
                        min=0, max=60, step=5, value=30
                    ).props("thumb-label").classes("w-full")
                    permit_delay.on('update:model-value', lambda e: permit_delay_label.set_text(f'{e.args} days'))
                    
                    ui.separator().classes('my-3')
                    
                    # Lease configuration
                    ui.label("Lease Selection").classes("font-bold")
                    ui.label("Lock individual wells to prevent optimizer from changing them").classes("text-gray-600 text-xs")
                    
                    lease_toggles = {}
                    well_sliders = {}
                    well_labels = {}
                    well_locks = {}
                    
                    for lease_id, config in TEXAS_LEASES.items():
                        with ui.row().classes('w-full items-center gap-2'):
                            lease_toggles[lease_id] = ui.switch(
                                f"{lease_id} ({config['basin']})",
                                value=True
                            ).classes('flex-grow')
                            
                            initial_value = min(10, config["wells"])
                            well_labels[lease_id] = ui.label(f"{initial_value} wells").classes("text-sm font-semibold text-gray-700 w-20 text-right")
                            
                            well_locks[lease_id] = ui.icon('lock_open', size='sm').classes('cursor-pointer text-gray-400')
                            def toggle_well_lock(lid):
                                lock = well_locks[lid]
                                slider = well_sliders[lid]
                                label = well_labels[lid]
                                
                                if lock.name == 'lock_open':
                                    lock.name = 'lock'
                                    lock.classes(replace='cursor-pointer text-orange-600')
                                    slider._locked = True
                                    slider.disable()
                                    slider.props(add='color=orange')
                                    label.classes(replace='text-sm font-semibold w-20 text-right text-orange-600')
                                else:
                                    lock.name = 'lock_open'
                                    lock.classes(replace='cursor-pointer text-gray-400')
                                    slider._locked = False
                                    slider.enable()
                                    slider.props(remove='color=orange')
                                    label.classes(replace='text-sm font-semibold w-20 text-right text-gray-700')
                            
                            well_locks[lease_id].on('click', lambda lid=lease_id: toggle_well_lock(lid))
                            
                            well_sliders[lease_id] = ui.slider(
                                min=0,
                                max=config["wells"],
                                step=1,
                                value=initial_value
                            ).props("thumb-label").classes("w-32")
                            
                            well_sliders[lease_id].on('update:model-value', 
                                lambda e, lid=lease_id: well_labels[lid].set_text(f'{e.args} well{"s" if e.args != 1 else ""}')
                            )
                            
                
                # Run optimization button
                async def run_optimization() -> None:
                    """Run optimization using selected method."""
                    try:
                        # Determine optimization method
                        selected_method = optimization_method.value
                        method = "vizier" if "Vizier" in selected_method else "heuristic"
                        
                        # Check if Vizier is properly configured
                        if method == "vizier":
                            method_info = get_available_methods()
                            if not method_info["vizier"]["available"] or not method_info["vizier"].get("setup_ok", False):
                                ui.notify("Vizier not properly configured. Falling back to heuristic method.", type='warning')
                                method = "heuristic"
                        
                        # Prepare lease limits for active leases
                        lease_limits = {}
                        for lease_id, config in TEXAS_LEASES.items():
                            if lease_toggles[lease_id].value:
                                lease_limits[lease_id] = config["wells"]
                        
                        # Extract locked parameters from UI
                        locked_parameters = {}
                        
                        # Check economic parameter locks
                        if getattr(oil_price, '_locked', False):
                            locked_parameters['oil_price_forecast'] = float(oil_price.value)
                        if getattr(discount_rate, '_locked', False):
                            locked_parameters['hurdle_rate'] = float(discount_rate.value) / 100.0  # Convert % to decimal
                        if getattr(contingency, '_locked', False):
                            locked_parameters['contingency_percent'] = float(contingency.value) / 100.0  # Convert % to decimal
                        
                        # Check drilling parameter locks
                        if getattr(rigs, '_locked', False):
                            locked_parameters['rig_count'] = int(rigs.value)
                        if getattr(drilling_mode, '_locked', False):
                            locked_parameters['drilling_mode'] = str(drilling_mode.value)
                        if getattr(permit_delay, '_locked', False):
                            # permit_delay doesn't directly map to optimization parameters, but we'll store it for completeness
                            pass
                        
                        # Check well parameter locks  
                        for lease_id in TEXAS_LEASES:
                            if lease_id in well_sliders and getattr(well_sliders[lease_id], '_locked', False):
                                locked_parameters[f'wells_{lease_id}'] = int(well_sliders[lease_id].value)
                        
                        # Log locked parameters for debugging
                        if locked_parameters:
                            print(f"Locked parameters: {locked_parameters}")
                        
                        # Create optimization manager
                        optimizer = OptimizationManager(method=method)
                        
                        # Set optimization parameters
                        n_trials = 25 if method == "vizier" else 10  # Fewer trials for UI responsiveness
                        
                        # Performance warning for Monte Carlo + Vizier combination
                        if monte_carlo_toggle.value and method == "vizier":
                            ui.notify(
                                f"⚠️ Monte Carlo + Vizier: Expected runtime ~{n_trials * 5} seconds", 
                                type="warning", 
                                timeout=5000
                            )
                        elif monte_carlo_toggle.value:
                            ui.notify(
                                f"📊 Monte Carlo enabled: +2-5 seconds per trial", 
                                type="info", 
                                timeout=3000
                            )
                        
                        # Update button to show method-specific progress
                        optimize_button.props('loading')
                        mc_indicator = " + MC" if monte_carlo_toggle.value else ""
                        if method == "vizier":
                            optimize_button.set_text(f"🧠 Vizier Optimizing{mc_indicator}...")
                            optimize_button.classes(add='bg-blue-600')
                        else:
                            optimize_button.set_text(f"🔍 Heuristic Optimizing{mc_indicator}...")
                            optimize_button.classes(add='bg-green-600')
                        
                        # Store initial parameters for comparison
                        initial_trial_count = len(client["trials"])
                        
                        print(f"\n=== Starting {method} optimization ===")
                        print(f"Available wells: {len(client['wells'])}")
                        print(f"Active leases: {list(lease_limits.keys())}")
                        print(f"Budget: ${budget.value}MM")
                        print(f"Target trials: {n_trials}")
                        
                        # Run optimization in background thread to avoid blocking UI
                        loop = asyncio.get_event_loop()
                        
                        # Create a future for the optimization task
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                optimizer.optimize,
                                available_wells=client["wells"],
                                lease_limits=lease_limits,
                                capex_budget=budget.value * 1_000_000,  # Convert MM to dollars
                                n_trials=n_trials,
                                locked_parameters=locked_parameters,
                                run_monte_carlo=monte_carlo_toggle.value,
                                mc_simulations=int(n_simulations.value)
                            )
                            
                            # Monitor progress with periodic UI updates
                            progress_counter = 0
                            while not future.done():
                                await asyncio.sleep(2)  # Update every 2 seconds
                                progress_counter += 1
                                
                                # Update progress indication
                                if method == "vizier":
                                    dots = "." * (progress_counter % 4)
                                    optimize_button.set_text(f"🧠 Vizier Optimizing{mc_indicator}{dots}")
                                else:
                                    dots = "." * (progress_counter % 4)
                                    optimize_button.set_text(f"🔍 Heuristic Optimizing{mc_indicator}{dots}")
                                
                                # Keep connection alive with a progress notification
                                if progress_counter % 5 == 0:  # Every 10 seconds
                                    elapsed_time = progress_counter * 2
                                    ui.notify(f"{method.title()} optimization running... ({elapsed_time}s elapsed)", type='info')
                            
                            # Get the results with timeout and proper error handling
                            try:
                                best_knobs, best_metrics, history = future.result(timeout=300)  # 5 minute timeout
                            except concurrent.futures.TimeoutError:
                                future.cancel()
                                raise TimeoutError(f"{method.title()} optimization timed out after 5 minutes")
                            except Exception as e:
                                raise RuntimeError(f"{method.title()} optimization failed in background thread: {str(e)}")
                        
                        # Process results from optimization
                        actual_method = optimizer.actual_method_used.value if optimizer.actual_method_used else method
                        
                        print(f"\n=== Optimization complete ===")
                        print(f"Method used: {actual_method}")
                        print(f"Best NPV: ${best_metrics.total_npv/1e6:.1f}MM")
                        print(f"Trials completed: {len(history.trials)}")
                        
                        # Add each trial to UI history
                        for trial in history.trials:
                            knobs = trial["knobs"]
                            metrics = trial["metrics"]
                            
                            # Convert OptimizationMetrics to dict format for UI
                            metrics_dict = {
                                "total_npv": metrics.total_npv,
                                "total_capex": metrics.total_capex,
                                "npv_per_dollar": metrics.npv_per_dollar,
                                "peak_production": metrics.peak_production,
                                "wells_selected": sum(knobs.wells_per_lease.values()),
                                "risk_score": metrics.risk_score,
                                "run_monte_carlo": monte_carlo_toggle.value,
                                "optimization_method": actual_method
                            }
                            
                            # Add Monte Carlo results if available
                            if metrics.p10_npv is not None:
                                metrics_dict["p10_npv"] = metrics.p10_npv
                                metrics_dict["p90_npv"] = metrics.p90_npv
                                metrics_dict["probability_positive"] = metrics.probability_positive
                                metrics_dict["n_simulations"] = int(n_simulations.value)
                            
                            add_trial(knobs, metrics_dict, client)
                        
                        # Update client best solution
                        client["best_score"] = best_metrics.total_npv
                        client["best_knobs"] = best_knobs
                        
                        # Update UI with best solution (only for unlocked parameters)
                        locked_params = {
                            'oil_price_forecast': getattr(oil_price, '_locked', False),
                            'hurdle_rate': getattr(discount_rate, '_locked', False),
                            'contingency_percent': getattr(contingency, '_locked', False),
                            'rig_count': getattr(rigs, '_locked', False),
                            'drilling_mode': getattr(drilling_mode, '_locked', False),
                        }
                        
                        well_locked = {}
                        for lease_id in TEXAS_LEASES:
                            well_locked[lease_id] = getattr(well_sliders[lease_id], '_locked', False)
                        
                        # Update UI controls with optimized values
                        if not locked_params.get('oil_price_forecast', False):
                            oil_price.value = best_knobs.oil_price_forecast
                            oil_price_label.set_text(f'${best_knobs.oil_price_forecast}/bbl')
                        if not locked_params.get('hurdle_rate', False):
                            discount_rate.value = int(best_knobs.hurdle_rate * 100)
                            discount_rate_label.set_text(f'{int(best_knobs.hurdle_rate * 100)}%')
                        if not locked_params.get('contingency_percent', False):
                            contingency.value = int(best_knobs.contingency_percent * 100)
                            contingency_label.set_text(f'{int(best_knobs.contingency_percent * 100)}%')
                        if not locked_params.get('rig_count', False):
                            rigs.value = best_knobs.rig_count
                            rigs_label.set_text(f'{best_knobs.rig_count} rig{"s" if best_knobs.rig_count != 1 else ""}')
                        if not locked_params.get('drilling_mode', False):
                            drilling_mode.value = best_knobs.drilling_mode
                        
                        # Update well sliders for unlocked wells
                        for lease_id in best_knobs.wells_per_lease:
                            if lease_id in well_sliders and not well_locked.get(lease_id, False):
                                well_count = best_knobs.wells_per_lease[lease_id]
                                well_sliders[lease_id].value = well_count
                                well_labels[lease_id].set_text(f'{well_count} well{"s" if well_count != 1 else ""}')
                        
                        # Show success notification
                        trials_completed = len(client["trials"]) - initial_trial_count
                        ui.notify(
                            f"{actual_method.title()} optimization complete! "
                            f"{trials_completed} trials, Best NPV: ${best_metrics.total_npv/1e6:.1f}MM",
                            type='positive'
                        )
                        
                        # Show fallback notification if applicable
                        if optimizer.fallback_reason:
                            ui.notify(f"Note: Fell back to heuristic method - {optimizer.fallback_reason}", type='info')
                        
                    except Exception as e:
                        print(f"Optimization failed: {e}")
                        ui.notify(f"Optimization failed: {str(e)}", type='negative')
                    
                    finally:
                        # Reset button state
                        optimize_button.props(remove='loading')
                        optimize_button.classes(remove='bg-blue-600 bg-green-600')
                        optimize_button.set_text("Optimize Development")
                
                # Connect button to function
                optimize_button.on_click(run_optimization)
            
            # RIGHT VISUALIZATION AREA
            with ui.column().classes('flex-1 gap-4'):
                
                # Metrics and wells per lease row
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        # Trial Navigation
                        with ui.card().classes('p-3 mb-2').style('width:700px'):
                            ui.label("Trial Navigation").classes("text-sm font-semibold text-gray-700 mb-2")
                            
                            with ui.row().classes('w-full items-center gap-2'):
                                # Trial selector dropdown
                                client["trial_selector"] = ui.select(
                                    options=[],
                                    label="Select Trial",
                                    value=None
                                ).classes('flex-1').props('outlined dense')
                                client["trial_selector"].set_enabled(False)
                                
                                # Quick action buttons
                                client["best_trial_btn"] = ui.button(
                                    "⭐ Best",
                                    on_click=lambda: select_best_trial(client)
                                ).classes('bg-amber-500 text-white px-3 py-1').props('dense')
                                client["best_trial_btn"].set_enabled(False)
                                
                                client["latest_trial_btn"] = ui.button(
                                    "🆕 Latest",
                                    on_click=lambda: select_latest_trial(client)
                                ).classes('bg-blue-500 text-white px-3 py-1').props('dense')
                                client["latest_trial_btn"].set_enabled(False)
                            
                            # Trial selector change handler
                            def on_trial_select(e):
                                if e.value is not None:
                                    select_trial(client, e.value)
                            
                            client["trial_selector"].on('update:model-value', on_trial_select)
                        
                        # Metrics card
                        client["stats_card"] = ui.card().classes(
                            'p-4'
                        ).style('width:700px;min-height:230px')
                        with client["stats_card"]:
                            client["stats_html"] = ui.html(
                                '<div class="text-gray-500">Click "Optimize Development" to see results</div>'
                            )
                    
                    with ui.column():
                        # Wells per lease
                        ui.label("Wells Drilled per Lease").classes("text-lg font-bold")
                        client["timeline_chart"] = ui.plotly(go.Figure()).classes(
                            'w-full'
                        ).style('width:400px;height:250px')
                
                # Optimization Trajectory
                with ui.card().classes('p-4 w-full'):
                    ui.label("Optimization Trajectory").classes("text-lg font-bold mb-2")
                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Shows how each parameter evolves during optimization").classes("text-gray-600 text-sm flex-1")
                        client["trajectory_mode"] = ui.toggle(
                            {"parallel": "Parallel Coordinates", "spider": "Spider Chart"},
                            value="parallel"
                        ).classes("w-64")
                    client["trajectory_chart"] = ui.plotly(go.Figure()).classes(
                        'w-full'
                    ).style('height:350px')
                
                # NPV sensitivity and production forecast
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label("NPV Sensitivity").classes("text-lg font-bold")
                        client["economics_chart"] = ui.plotly(go.Figure()).classes(
                            'w-full'
                        ).style('height:250px')
                    
                    with ui.column().classes('flex-1'):
                        ui.label("Production Forecast").classes("text-lg font-bold")
                        client["production_chart"] = ui.plotly(go.Figure()).classes(
                            'w-full'
                        ).style('height:300px')
                
                # Optimization history
                with ui.card().classes('p-4 w-full'):
                    ui.label("Optimization History (3D)").classes("text-lg font-bold mb-2")
                    client["history_plot"] = ui.plotly(go.Figure()).classes(
                        'w-full'
                    ).style('height:400px')


# The @ui.page decorator registers the route when this module is imported