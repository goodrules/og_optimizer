"""
Oil & gas decline curve calculations.
Implements hyperbolic decline curves for production forecasting.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date, timedelta
import numpy as np
import calendar


@dataclass
class DeclineCurveParameters:
    """Parameters for hyperbolic decline curve calculations."""
    ip_rate: float  # Initial production rate (boe/d)
    di: float  # Initial decline rate (annual)
    b: float  # Hyperbolic exponent (0-2)
    eur_mboe: Optional[float] = None  # Estimated Ultimate Recovery (thousand boe)
    
    def __post_init__(self):
        """Validate parameters."""
        if self.ip_rate <= 0:
            raise ValueError("IP rate must be positive")
        if not 0 <= self.di <= 1:
            raise ValueError("Decline rate must be between 0 and 1")
        if not 0 <= self.b <= 2:
            raise ValueError("b factor must be between 0 and 2")


def hyperbolic_decline(params: DeclineCurveParameters, time_months: float) -> float:
    """
    Calculate production rate using hyperbolic decline curve.
    
    Formula:
    - If b = 0 (exponential): q = qi * exp(-Di * t)
    - If b = 1 (harmonic): q = qi / (1 + Di * t)
    - Otherwise: q = qi / (1 + b * Di * t)^(1/b)
    
    Args:
        params: Decline curve parameters
        time_months: Time in months from start of production
        
    Returns:
        Production rate in boe/d
    """
    if time_months == 0:
        return params.ip_rate
    
    # Convert months to years for annual decline rate
    time_years = time_months / 12.0
    
    if params.b == 0:
        # Exponential decline
        return params.ip_rate * np.exp(-params.di * time_years)
    elif params.b == 1:
        # Harmonic decline
        return params.ip_rate / (1 + params.di * time_years)
    else:
        # Hyperbolic decline
        return params.ip_rate / ((1 + params.b * params.di * time_years) ** (1 / params.b))


def calculate_monthly_production(
    params: DeclineCurveParameters,
    months: int,
    start_date: Optional[date] = None
) -> List[Dict[str, any]]:
    """
    Generate monthly production profile.
    
    Args:
        params: Decline curve parameters
        months: Number of months to calculate
        start_date: Start date for production (defaults to today)
        
    Returns:
        List of dictionaries with monthly production data
    """
    if months == 0:
        return []
    
    if start_date is None:
        start_date = date.today()
    
    monthly_production = []
    
    for month_idx in range(months):
        # Calculate date for this month
        current_date = start_date + timedelta(days=30 * month_idx)
        # Get actual month/year for proper day count
        year = start_date.year + (start_date.month + month_idx - 1) // 12
        month = ((start_date.month + month_idx - 1) % 12) + 1
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Calculate average daily rate for the month (using midpoint)
        day_rate = hyperbolic_decline(params, month_idx + 0.5)
        
        # Calculate total production for the month
        monthly_boe = day_rate * days_in_month
        
        monthly_production.append({
            'date': date(year, month, 1),
            'production_boe': monthly_boe,
            'day_rate_boed': day_rate,
            'days': days_in_month
        })
    
    return monthly_production


def calculate_cumulative_production(
    params: DeclineCurveParameters,
    months: int
) -> float:
    """
    Calculate cumulative production over time period.
    
    Args:
        params: Decline curve parameters
        months: Number of months to calculate
        
    Returns:
        Cumulative production in boe
    """
    if months == 0:
        return 0.0
    
    # Use numerical integration for accuracy
    monthly_prod = calculate_monthly_production(params, months)
    cumulative = sum(month['production_boe'] for month in monthly_prod)
    
    # Cap at EUR if specified
    if params.eur_mboe is not None:
        eur_boe = params.eur_mboe * 1000
        cumulative = min(cumulative, eur_boe)
    
    return cumulative