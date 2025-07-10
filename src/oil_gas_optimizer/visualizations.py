"""
Visualization components for oil & gas field development.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

from .economics import WellEconomics
from .decline_curves import calculate_monthly_production
from .drilling_optimizer import DrillingSchedule


def create_production_forecast_chart(
    wells: List[WellEconomics],
    months: int = 120,
    show_uncertainty: bool = False
) -> go.Figure:
    """
    Create production forecast chart with decline curves.
    
    Args:
        wells: List of wells to forecast
        months: Forecast period in months
        show_uncertainty: Whether to show P10/P90 bands
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Calculate total production
    time_points = np.arange(0, months)
    total_production = np.zeros(months)
    
    for well in wells:
        well_production = []
        for month in time_points:
            prod = calculate_monthly_production(
                well.decline_params,
                month,
                month + 1
            )
            well_production.append(prod)
        total_production += np.array(well_production)
    
    # Base production line
    fig.add_trace(go.Scatter(
        x=time_points,
        y=total_production,
        mode='lines',
        name='Expected Production',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add uncertainty bands if requested
    if show_uncertainty and len(wells) > 0:
        # Simplified P10/P90 bands
        p90_production = total_production * 1.15
        p10_production = total_production * 0.85
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=p90_production,
            mode='lines',
            name='P90',
            line=dict(color='#10b981', dash='dash', width=2),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=p10_production,
            mode='lines',
            name='P10',
            line=dict(color='#ef4444', dash='dash', width=2),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.05)',
            showlegend=True
        ))
    
    # Add annotations for key metrics
    if len(wells) > 0:
        peak_prod = total_production[0]
        eur = np.sum(total_production) / 1000  # Convert to Mboe
        
        fig.add_annotation(
            x=0, y=peak_prod,
            text=f"Peak: {peak_prod:,.0f} boe/d",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-30
        )
        
        fig.add_annotation(
            x=months/2, y=total_production[months//2],
            text=f"EUR: {eur:,.0f} Mboe",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-30
        )
    
    fig.update_layout(
        title="Field Production Forecast",
        xaxis_title="Months",
        yaxis_title="Production (boe/d)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_economics_waterfall_chart(
    schedule: DrillingSchedule,
    oil_price: float = 80.0
) -> go.Figure:
    """
    Create waterfall chart showing economics breakdown.
    
    Args:
        schedule: Drilling schedule with economics
        oil_price: Oil price assumption
        
    Returns:
        Plotly figure object
    """
    # Calculate economic components
    gross_revenue = sum(w.ip_rate * 365 * oil_price for w in schedule.wells_drilled)
    total_capex = schedule.total_capex
    opex = gross_revenue * 0.15  # Simplified
    royalties = gross_revenue * 0.1875
    taxes = gross_revenue * 0.046  # Texas severance
    net_income = gross_revenue - opex - royalties - taxes - total_capex
    
    # Create waterfall data
    x = ["Gross Revenue", "OPEX", "Royalties", "Taxes", "CAPEX", "Net Income"]
    y = [gross_revenue, -opex, -royalties, -taxes, -total_capex, net_income]
    
    # Determine colors
    colors = ["green", "red", "red", "red", "red", 
              "green" if net_income > 0 else "red"]
    
    fig = go.Figure(go.Waterfall(
        x=x,
        y=y,
        textposition="outside",
        text=[f"${v/1e6:.1f}M" for v in y],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))
    
    fig.update_layout(
        title=f"Economic Waterfall at ${oil_price}/bbl",
        yaxis_title="Value ($)",
        template='plotly_white',
        height=400
    )
    
    return fig


def create_drilling_gantt_chart(schedule: DrillingSchedule) -> go.Figure:
    """
    Create Gantt chart showing drilling timeline.
    
    Args:
        schedule: Drilling schedule
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Define colors by lease
    lease_colors = {
        "MIDLAND": "#3b82f6",
        "MARTIN": "#ef4444",
        "REEVES": "#10b981",
        "LOVING": "#f59e0b",
        "HOWARD": "#8b5cf6"
    }
    
    # Create timeline bars
    for i, well in enumerate(schedule.wells_drilled):
        lease_prefix = well.name.split("_")[0]
        color = lease_colors.get(lease_prefix, "#666666")
        
        start_date = schedule.start_dates.get(well.name)
        if start_date:
            # Simplified duration calculation
            duration_days = 30  # Approximate
            
            fig.add_trace(go.Bar(
                y=[well.name],
                x=[duration_days],
                base=[i * 35],  # Stagger for visibility
                orientation='h',
                marker_color=color,
                name=lease_prefix,
                showlegend=i == 0,  # Only show legend once per lease
                hovertemplate=(
                    f"{well.name}<br>"
                    f"Start: Day {i * 35}<br>"
                    f"Duration: {duration_days} days<br>"
                    f"CAPEX: ${well.capex/1e6:.1f}MM<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title="Drilling Schedule Timeline",
        xaxis_title="Days from Start",
        yaxis_title="Well",
        barmode='stack',
        template='plotly_white',
        height=max(300, len(schedule.wells_drilled) * 25)
    )
    
    return fig


def create_lease_summary_chart(
    wells_by_lease: Dict[str, List[WellEconomics]]
) -> go.Figure:
    """
    Create summary chart by lease showing well count and economics.
    
    Args:
        wells_by_lease: Dictionary mapping lease names to well lists
        
    Returns:
        Plotly figure object
    """
    leases = []
    well_counts = []
    total_capex = []
    avg_ip = []
    
    for lease, wells in wells_by_lease.items():
        if wells:
            leases.append(lease)
            well_counts.append(len(wells))
            total_capex.append(sum(w.capex for w in wells) / 1e6)
            avg_ip.append(np.mean([w.ip_rate for w in wells]))
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Wells per Lease', 'CAPEX by Lease ($MM)', 'Avg IP Rate (boe/d)')
    )
    
    # Well count
    fig.add_trace(
        go.Bar(x=leases, y=well_counts, marker_color='#3b82f6'),
        row=1, col=1
    )
    
    # CAPEX
    fig.add_trace(
        go.Bar(x=leases, y=total_capex, marker_color='#ef4444'),
        row=1, col=2
    )
    
    # Average IP
    fig.add_trace(
        go.Bar(x=leases, y=avg_ip, marker_color='#10b981'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Lease Summary Statistics",
        showlegend=False,
        template='plotly_white',
        height=350
    )
    
    return fig


def create_optimization_scatter_matrix(trials: List[Dict]) -> go.Figure:
    """
    Create scatter matrix of optimization parameters and results.
    
    Args:
        trials: List of optimization trials
        
    Returns:
        Plotly figure object
    """
    if not trials:
        return go.Figure()
    
    # Extract data
    npvs = []
    capexs = []
    well_counts = []
    oil_prices = []
    rig_counts = []
    
    for trial in trials:
        m = trial["metrics"]
        k = trial["knobs"]
        
        npvs.append(m["total_npv"] / 1e6)
        capexs.append(m["total_capex"] / 1e6)
        well_counts.append(m["wells_drilled"])
        oil_prices.append(k.oil_price_forecast)
        rig_counts.append(k.rig_count)
    
    # Create scatter matrix
    import pandas as pd
    
    df = pd.DataFrame({
        'NPV ($MM)': npvs,
        'CAPEX ($MM)': capexs,
        'Wells': well_counts,
        'Oil Price': oil_prices,
        'Rigs': rig_counts
    })
    
    fig = px.scatter_matrix(
        df,
        dimensions=['NPV ($MM)', 'CAPEX ($MM)', 'Wells', 'Oil Price', 'Rigs'],
        color=list(range(len(trials))),
        color_continuous_scale='Viridis',
        title="Optimization Parameter Relationships"
    )
    
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=600, template='plotly_white')
    
    return fig