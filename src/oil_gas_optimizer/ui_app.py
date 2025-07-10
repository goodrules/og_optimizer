"""
Oil & Gas Field Development Optimizer UI
Adapted from the work package optimizer UI in local/main.py
"""
import math
import random
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
from .economics import WellEconomics, EconomicParameters
from .data_model import generate_texas_wells
from .monte_carlo import MonteCarloParameters

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


def add_trial(knobs: OptimizationKnobs, metrics: Dict, client: Dict) -> None:
    """Add optimization trial to history and update UI."""
    trial = {
        "knobs": knobs,
        "metrics": metrics,
        "timestamp": datetime.now()
    }
    client["trials"].append(trial)
    
    # Update all visualizations
    idx = len(client["trials"]) - 1
    update_metrics_card(idx, client)
    update_drilling_timeline(idx, client)
    update_production_forecast(idx, client)
    update_economics_chart(idx, client)
    update_optimization_history(client)
    update_trajectory_chart(client)


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
    
    client["stats_html"].content = f"""
    <div class="grid grid-cols-4 gap-x-4 gap-y-2">
      <div><div class="text-gray-500 text-sm">Total NPV</div>
           <div class="text-2xl font-bold">${m['total_npv']/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">Total CAPEX</div>
           <div class="text-xl font-bold text-indigo-600">${m['total_capex']/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">NPV Improvement</div>
           <div class="text-2xl font-bold {color}">${npv_improvement/1e6:.1f}MM</div></div>
           
      <div><div class="text-gray-500 text-sm">Wells Drilled</div>
           <div class="text-2xl font-bold">{m['wells_drilled']}</div></div>
           
      <!-- Row 2 -->
      <div><div class="text-gray-500 text-sm">NPV/$ Invested</div>
           <div class="text-2xl font-bold">{m.get('npv_per_dollar', 0):.2f}</div></div>
           
      <div><div class="text-gray-500 text-sm">Peak Production</div>
           <div class="text-2xl font-bold">{m.get('peak_production', 0):.0f} boe/d</div></div>
           
      <div><div class="text-gray-500 text-sm">Trial #</div>
           <div class="text-2xl font-bold">{idx + 1}</div></div>
           
      <div><div class="text-gray-500 text-sm">Risk Score</div>
           <div class="text-2xl font-bold">{m.get('risk_score', 0.5):.0%}</div></div>
    </div>"""


def update_drilling_timeline(idx: int, client: Dict) -> None:
    """Update drilling timeline visualization (replaces Gantt chart)."""
    m = client["trials"][idx]["metrics"]
    k = client["trials"][idx]["knobs"]
    
    fig = go.Figure()
    
    # Create timeline bars for each lease
    colors = {
        "MIDLAND_A": "#3b82f6",  # blue
        "MARTIN_B": "#ef4444",   # red
        "REEVES_C": "#10b981",   # green
        "LOVING_D": "#f59e0b",   # amber
        "HOWARD_E": "#8b5cf6"    # purple
    }
    
    y_pos = 0
    for lease, well_count in k.wells_per_lease.items():
        if well_count > 0:
            # Simulate drilling schedule
            drill_months = well_count / k.rig_count * 1.5  # Simplified
            
            fig.add_trace(go.Bar(
                y=[lease],
                x=[drill_months],
                base=[0],
                orientation="h",
                width=0.6,
                marker={"color": colors.get(lease, "#666")},
                hovertemplate=(
                    f"{lease}<br>"
                    f"Wells: {well_count}<br>"
                    f"Duration: {drill_months:.1f} months<extra></extra>"
                ),
                showlegend=False
            ))
    
    fig.update_layout(
        barmode="stack",
        height=250,
        xaxis_title="Months",
        yaxis_title="",
        xaxis=dict(range=[0, 24]),
        margin=dict(l=80, r=10, t=10, b=30)
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
    capexs = []
    efficiencies = []
    
    for trial in client["trials"]:
        m = trial["metrics"]
        npvs.append(m["total_npv"] / 1e6)  # MM
        capexs.append(m["total_capex"] / 1e6)  # MM
        efficiencies.append(m.get("npv_per_dollar", 0))
    
    fig = go.Figure()
    
    # Create 3D scatter
    fig.add_trace(go.Scatter3d(
        x=npvs,
        y=capexs,
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
            "CAPEX: $%{y:.1f}MM<br>"
            "Efficiency: %{z:.2f}<br>"
            "%{text}<extra></extra>"
        )
    ))
    
    # Highlight best solution
    best_idx = np.argmax(npvs)
    fig.add_trace(go.Scatter3d(
        x=[npvs[best_idx]],
        y=[capexs[best_idx]],
        z=[efficiencies[best_idx]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        showlegend=False
    ))
    
    fig.update_layout(
        height=400,
        scene=dict(
            xaxis_title="NPV ($MM)",
            yaxis_title="CAPEX ($MM)",
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
    
    # Header
    with ui.header().classes(replace='row items-center h-16 bg-gray-800'):
        ui.label("🛢️").style('font-size: 2em').tailwind("pr-2")
        ui.label('Oil & Gas Field Development Optimizer').style(
            'color:white;font-size:125%;'
        ).tailwind("px-2.5 pl-4", "font-bold")
        ui.chip('Texas Oil Focus', color="grey").props("outline")
    
    # Main layout
    with ui.column().classes('w-full p-4').style('max-width:1600px;margin:0 auto'):
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
                
                # Helper function to toggle lock state
                def toggle_lock(lock_icon, slider):
                    if lock_icon.name == 'lock_open':
                        lock_icon.name = 'lock'
                        lock_icon.classes(replace='cursor-pointer text-orange-600')
                        slider._locked = True
                        slider.props(add='color=orange')
                    else:
                        lock_icon.name = 'lock_open'
                        lock_icon.classes(replace='cursor-pointer text-gray-400')
                        slider._locked = False
                        slider.props(remove='color=orange')
                
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
                    
                    # Add optimizer control toggle
                    optimizer_controls_wells = ui.switch(
                        "Let optimizer control well selection",
                        value=False
                    ).props('color=orange')
                    ui.label("When enabled, the optimizer will determine optimal well counts").classes("text-gray-600 text-xs")
                    
                    lease_toggles = {}
                    well_sliders = {}
                    well_labels = {}
                    
                    for lease_id, config in TEXAS_LEASES.items():
                        with ui.row().classes('w-full items-center gap-2'):
                            lease_toggles[lease_id] = ui.switch(
                                f"{lease_id} ({config['basin']})",
                                value=True
                            ).classes('flex-1')
                            
                            well_sliders[lease_id] = ui.slider(
                                min=0,
                                max=config["wells"],
                                step=1,
                                value=min(10, config["wells"])
                            ).props("thumb-label").classes("w-32").bind_enabled_from(
                                optimizer_controls_wells, 'value', lambda v: not v
                            )
                            
                            initial_value = min(10, config["wells"])
                            well_labels[lease_id] = ui.label(f"{initial_value} wells").classes("text-sm font-semibold text-gray-700 w-16")
                            well_sliders[lease_id].on('update:model-value', 
                                lambda e, lid=lease_id: well_labels[lid].set_text(f'{e.args} well{"s" if e.args != 1 else ""}')
                            )
                            # Make label gray when optimizer controls wells
                            optimizer_controls_wells.on('update:model-value', 
                                lambda e, lid=lease_id: well_labels[lid].classes(
                                    replace='text-sm font-semibold w-16 ' + ('text-gray-400' if e.args else 'text-gray-700')
                                )
                            )
                
                # Run optimization button
                async def run_optimization() -> None:
                    """Run one optimization iteration."""
                    t_idx = len(client["trials"])
                    
                    # Collect lock states
                    locked_params = {
                        'oil_price_forecast': getattr(oil_price, '_locked', False),
                        'hurdle_rate': getattr(discount_rate, '_locked', False),
                        'contingency_percent': getattr(contingency, '_locked', False),
                        'rig_count': getattr(rigs, '_locked', False),
                        'drilling_mode': getattr(drilling_mode, '_locked', False),
                        'permit_delay': getattr(permit_delay, '_locked', False),
                        'wells_per_lease': not optimizer_controls_wells.value
                    }
                    
                    # Build knobs from UI values
                    wells_per_lease = {}
                    for lease_id in TEXAS_LEASES:
                        if lease_toggles[lease_id].value:
                            wells_per_lease[lease_id] = well_sliders[lease_id].value
                    
                    if t_idx == 0 or client["best_knobs"] is None:
                        # Start with worst case
                        knobs = worst_case_knobs()
                        if not optimizer_controls_wells.value:
                            knobs.wells_per_lease = wells_per_lease
                    else:
                        # Perturb from best
                        # Increase minimum scale to 0.3 for more exploration
                        scale = max(0.3, 1.0 - (t_idx / IMPROVE_TRIALS))
                        knobs = perturb_knobs(client["best_knobs"], scale, locked_params)
                        # Only override well selection if optimizer is not controlling it
                        if not optimizer_controls_wells.value:
                            knobs.wells_per_lease = wells_per_lease
                    
                    # Update knobs from UI for locked parameters only
                    if locked_params.get('oil_price_forecast', False):
                        knobs.oil_price_forecast = oil_price.value
                    if locked_params.get('hurdle_rate', False):
                        knobs.hurdle_rate = discount_rate.value / 100.0
                    if locked_params.get('contingency_percent', False):
                        knobs.contingency_percent = contingency.value / 100.0
                    if locked_params.get('rig_count', False):
                        knobs.rig_count = rigs.value
                    if locked_params.get('drilling_mode', False):
                        knobs.drilling_mode = drilling_mode.value
                    
                    # Evaluate scenario
                    # Run synchronously to avoid pickling issues with OR-Tools
                    metrics = evaluate_scenario(
                        knobs,
                        client["wells"],
                        DEFAULT_BUDGET
                    )
                    
                    # Convert to dict format
                    metrics_dict = {
                        "total_npv": metrics.total_npv,
                        "total_capex": metrics.total_capex,
                        "npv_per_dollar": metrics.npv_per_dollar,
                        "peak_production": metrics.peak_production,
                        "wells_drilled": metrics.wells_drilled,
                        "risk_score": metrics.risk_score
                    }
                    
                    # Update best solution
                    score = metrics.total_npv
                    if score > client["best_score"]:
                        client["best_score"] = score
                        client["best_knobs"] = knobs
                    
                    # Add trial and update UI
                    add_trial(knobs, metrics_dict, client)
                    
                    # Update UI sliders to show optimized values (only for unlocked parameters)
                    if not locked_params.get('oil_price_forecast', False):
                        oil_price.value = knobs.oil_price_forecast
                    if not locked_params.get('hurdle_rate', False):
                        discount_rate.value = int(knobs.hurdle_rate * 100)
                    if not locked_params.get('contingency_percent', False):
                        contingency.value = int(knobs.contingency_percent * 100)
                    if not locked_params.get('rig_count', False):
                        rigs.value = knobs.rig_count
                    if not locked_params.get('drilling_mode', False):
                        drilling_mode.value = knobs.drilling_mode
                    
                    # Update well sliders if optimizer controls them
                    if optimizer_controls_wells.value:
                        for lease_id in knobs.wells_per_lease:
                            if lease_id in well_sliders:
                                well_sliders[lease_id].value = knobs.wells_per_lease[lease_id]
                
                # Connect button to function
                optimize_button.on_click(run_optimization)
            
            # RIGHT VISUALIZATION AREA
            with ui.column().classes('flex-1 gap-4'):
                
                # Metrics and economics row
                with ui.row().classes('w-full gap-4'):
                    with ui.column():
                        # Metrics card
                        client["stats_card"] = ui.card().classes(
                            'p-4'
                        ).style('width:700px;height:230px')
                        with client["stats_card"]:
                            client["stats_html"] = ui.html(
                                '<div class="text-gray-500">Run optimization to see results</div>'
                            )
                    
                    with ui.column():
                        # Economics sensitivity
                        ui.label("NPV Sensitivity").classes("text-lg font-bold")
                        client["economics_chart"] = ui.plotly(go.Figure()).classes(
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
                
                # Drilling timeline and production forecast
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label("Drilling Timeline").classes("text-lg font-bold")
                        client["timeline_chart"] = ui.plotly(go.Figure()).classes(
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