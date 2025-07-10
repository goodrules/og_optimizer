"""
Economic calculations for oil & gas field development.
Implements NPV, IRR, and other financial metrics.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from datetime import date, timedelta
import numpy as np
from scipy import optimize

from .decline_curves import (
    DeclineCurveParameters,
    calculate_monthly_production,
    calculate_cumulative_production
)


@dataclass
class EconomicParameters:
    """Economic assumptions for calculations."""
    oil_price: float  # $/bbl
    discount_rate: float  # Annual discount rate
    opex_per_boe: float = 12.0  # $/boe operating cost
    royalty: float = 0.1875  # Royalty fraction
    working_interest: float = 0.80  # Working interest fraction
    severance_tax: float = 0.046  # Texas severance tax
    ad_valorem_tax: float = 0.02  # Property tax
    
    def __post_init__(self):
        """Validate parameters."""
        if self.oil_price <= 0:
            raise ValueError("Oil price must be positive")
        if not 0 <= self.discount_rate < 1:
            raise ValueError("Discount rate must be between 0 and 1")
    
    @property
    def net_revenue_interest(self) -> float:
        """Calculate net revenue interest (NRI)."""
        return self.working_interest * (1 - self.royalty)


@dataclass
class CashFlow:
    """Cash flow series with dates and values."""
    dates: List[date]
    values: List[float]
    
    def __post_init__(self):
        """Validate cash flows."""
        if len(self.dates) != len(self.values):
            raise ValueError("Dates and values must have same length")


@dataclass
class WellEconomics:
    """Economic model for a single well."""
    name: str
    capex: float  # Capital expenditure ($)
    ip_rate: float  # Initial production (boe/d)
    di: float  # Annual decline rate
    b: float  # Hyperbolic exponent
    months: int = 360  # Production life (months)
    eur_mboe: Optional[float] = None
    
    def get_decline_params(self) -> DeclineCurveParameters:
        """Get decline curve parameters."""
        return DeclineCurveParameters(
            ip_rate=self.ip_rate,
            di=self.di,
            b=self.b,
            eur_mboe=self.eur_mboe
        )
    
    def get_cash_flows(self, econ_params: EconomicParameters, start_month: int = 0) -> CashFlow:
        """Generate cash flows for this well."""
        # Get production profile
        decline_params = self.get_decline_params()
        monthly_prod = calculate_monthly_production(decline_params, self.months)
        
        dates = []
        values = []
        
        # Initial capex (negative cash flow)
        start_date = date.today() + timedelta(days=30 * start_month)
        dates.append(start_date)
        values.append(-self.capex)
        
        # Monthly cash flows
        for i, month_data in enumerate(monthly_prod):
            production_boe = month_data['production_boe']
            
            # Revenue
            gross_revenue = production_boe * econ_params.oil_price
            net_revenue = gross_revenue * econ_params.net_revenue_interest
            
            # Costs
            opex = production_boe * econ_params.opex_per_boe
            severance = net_revenue * econ_params.severance_tax
            ad_valorem = net_revenue * econ_params.ad_valorem_tax
            
            # Net cash flow
            net_cash = net_revenue - opex - severance - ad_valorem
            
            dates.append(start_date + timedelta(days=30 * (i + 1)))
            values.append(net_cash)
        
        return CashFlow(dates=dates, values=values)


def calculate_well_npv(
    well: WellEconomics,
    econ_params: EconomicParameters,
    start_month: int = 0
) -> float:
    """
    Calculate Net Present Value for a single well.
    
    Args:
        well: Well economics model
        econ_params: Economic parameters
        start_month: Months delay before drilling
        
    Returns:
        NPV in dollars
    """
    cash_flows = well.get_cash_flows(econ_params, start_month)
    
    # Calculate NPV
    npv = 0.0
    monthly_discount = (1 + econ_params.discount_rate) ** (1/12) - 1
    
    # If delayed start, discount all cash flows back to time 0
    today = date.today()
    for date_val, cash in zip(cash_flows.dates, cash_flows.values):
        # Calculate months from today
        months_from_today = ((date_val.year - today.year) * 12 + 
                           (date_val.month - today.month))
        discount_factor = 1 / ((1 + monthly_discount) ** months_from_today)
        npv += cash * discount_factor
    
    return npv


def calculate_irr(cash_flows: CashFlow) -> Optional[float]:
    """
    Calculate Internal Rate of Return.
    
    Args:
        cash_flows: Cash flow series
        
    Returns:
        Annual IRR (None if no solution exists)
    """
    if not cash_flows.values:
        return None
    
    # Check if project ever goes positive
    cumulative = np.cumsum(cash_flows.values)
    if all(c < 0 for c in cumulative):
        return None
    
    # Create time series in years
    reference_date = cash_flows.dates[0]
    times = []
    for date_val in cash_flows.dates:
        days_diff = (date_val - reference_date).days
        times.append(days_diff / 365.25)
    
    # Define NPV function for root finding
    def npv_at_rate(rate):
        npv = 0
        for t, cf in zip(times, cash_flows.values):
            if t == 0:
                npv += cf
            else:
                npv += cf / ((1 + rate) ** t)
        return npv
    
    try:
        # Find IRR using root finding
        result = optimize.brentq(npv_at_rate, -0.99, 10.0)
        return result
    except:
        return None


def calculate_payback_period(cash_flows: CashFlow) -> Optional[int]:
    """
    Calculate payback period in months.
    
    Args:
        cash_flows: Cash flow series
        
    Returns:
        Payback period in months (None if never pays back)
    """
    cumulative = 0.0
    
    for i, value in enumerate(cash_flows.values):
        cumulative += value
        if cumulative >= 0:
            return i
    
    return None


def calculate_field_economics(
    drilling_schedule: Union[Dict[date, List[WellEconomics]], List[WellEconomics]],
    econ_params: EconomicParameters,
    capital_budget: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate field-level economics.
    
    Args:
        drilling_schedule: Either a dict of {date: [wells]} or list of wells
        econ_params: Economic parameters
        capital_budget: Optional capital constraint
        
    Returns:
        Dictionary of field metrics
    """
    # Handle different input formats
    if isinstance(drilling_schedule, dict):
        all_wells = []
        for wells in drilling_schedule.values():
            all_wells.extend(wells)
    else:
        all_wells = drilling_schedule
    
    # Apply capital budget constraint if specified
    if capital_budget and all_wells:
        cumulative_capex = 0
        constrained_wells = []
        for well in all_wells:
            if cumulative_capex + well.capex <= capital_budget:
                constrained_wells.append(well)
                cumulative_capex += well.capex
        all_wells = constrained_wells
    
    # Calculate metrics
    total_npv = 0
    total_capex = 0
    total_eur = 0
    irr_values = []
    
    # Production profile tracking
    monthly_production = {}
    
    for i, well in enumerate(all_wells):
        # Calculate well NPV
        npv = calculate_well_npv(well, econ_params, start_month=i)
        total_npv += npv
        total_capex += well.capex
        
        # Calculate EUR
        decline_params = well.get_decline_params()
        eur = calculate_cumulative_production(decline_params, well.months)
        total_eur += eur
        
        # Calculate IRR
        cash_flows = well.get_cash_flows(econ_params, start_month=i)
        irr = calculate_irr(cash_flows)
        if irr is not None:
            irr_values.append(irr)
        
        # Track production
        monthly_prod = calculate_monthly_production(decline_params, well.months)
        for j, month_data in enumerate(monthly_prod):
            month_key = i + j
            if month_key not in monthly_production:
                monthly_production[month_key] = 0
            monthly_production[month_key] += month_data['day_rate_boed']
    
    # Find peak production
    peak_production = max(monthly_production.values()) if monthly_production else 0
    
    return {
        'total_npv': total_npv,
        'total_capex': total_capex,
        'total_eur_mboe': total_eur / 1000,  # Convert to thousand barrels
        'peak_production_boed': peak_production,
        'average_irr': np.mean(irr_values) if irr_values else 0,
        'wells_drilled': len(all_wells)
    }