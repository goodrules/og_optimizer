"""
Monte Carlo simulation for oil & gas field development uncertainty.
Adapts the existing Monte Carlo approach from local/main.py for oil price and cost uncertainty.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from .economics import (
    EconomicParameters,
    WellEconomics,
    calculate_well_npv,
    calculate_field_economics
)


@dataclass
class OilPriceModel:
    """Model for oil price uncertainty using mean-reverting process."""
    base_price: float
    volatility: float = 0.25
    mean_reversion_rate: float = 0.2
    long_term_mean: Optional[float] = None
    
    def __post_init__(self):
        if self.long_term_mean is None:
            self.long_term_mean = self.base_price
    
    def generate_price_path(self, months: int, n_simulations: int) -> np.ndarray:
        """
        Generate oil price paths using Ornstein-Uhlenbeck process.
        
        Returns:
            Array of shape (n_simulations, months) with price paths
        """
        dt = 1/12  # Monthly time step
        prices = np.zeros((n_simulations, months))
        prices[:, 0] = self.base_price
        
        for t in range(1, months):
            # Mean reversion term
            drift = self.mean_reversion_rate * (self.long_term_mean - prices[:, t-1]) * dt
            # Random shock
            shock = self.volatility * np.sqrt(dt) * np.random.normal(0, 1, n_simulations)
            # Update price (ensure positive)
            prices[:, t] = np.maximum(prices[:, t-1] + drift * prices[:, t-1] + shock * prices[:, t-1], 10.0)
        
        return prices


@dataclass
class CostUncertaintyModel:
    """Model for drilling cost uncertainty."""
    base_cost: float
    uncertainty: float = 0.15
    correlation: float = 0.3  # Correlation between wells
    annual_escalation: float = 0.0
    
    def generate_well_costs(self, n_wells: int, n_simulations: int) -> np.ndarray:
        """
        Generate correlated cost realizations for multiple wells.
        
        Returns:
            Array of shape (n_simulations, n_wells) with costs
        """
        # Create correlation matrix
        corr_matrix = np.full((n_wells, n_wells), self.correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Generate correlated normal variables
        mean = np.zeros(n_wells)
        random_vars = np.random.multivariate_normal(mean, corr_matrix, n_simulations)
        
        # Convert to lognormal costs
        costs = np.zeros((n_simulations, n_wells))
        for i in range(n_wells):
            # Lognormal parameters
            sigma = self.uncertainty
            mu = np.log(self.base_cost) - 0.5 * sigma**2
            costs[:, i] = np.exp(mu + sigma * random_vars[:, i])
        
        return costs
    
    def generate_well_costs_with_timing(
        self, 
        drill_months: List[int], 
        n_simulations: int
    ) -> np.ndarray:
        """Generate costs with time-based escalation."""
        n_wells = len(drill_months)
        base_costs = self.generate_well_costs(n_wells, n_simulations)
        
        # Apply escalation
        for i, month in enumerate(drill_months):
            years = month / 12.0
            escalation_factor = (1 + self.annual_escalation) ** years
            base_costs[:, i] *= escalation_factor
        
        return base_costs


@dataclass
class OperationalRiskModel:
    """Model for operational delays and risks."""
    mechanical_failure_rate: float = 0.07
    weather_delay_days: float = 12.0
    permit_delay_range: Tuple[float, float] = (0, 45)
    
    def generate_delays(self, n_wells: int, n_simulations: int) -> np.ndarray:
        """
        Generate operational delays for wells.
        
        Returns:
            Array of shape (n_simulations, n_wells) with delay days
        """
        delays = np.zeros((n_simulations, n_wells))
        
        for i in range(n_wells):
            # Mechanical failures (binary event)
            failures = np.random.binomial(1, self.mechanical_failure_rate, n_simulations)
            failure_delays = failures * np.random.uniform(10, 30, n_simulations)
            
            # Weather delays (continuous)
            weather_delays = np.random.exponential(self.weather_delay_days / 3, n_simulations)
            weather_delays = np.minimum(weather_delays, self.weather_delay_days * 2)
            
            # Permit delays (uniform)
            permit_delays = np.random.uniform(*self.permit_delay_range, n_simulations)
            
            # Total delay is not simply additive - some overlap
            delays[:, i] = failure_delays + weather_delays * 0.5 + permit_delays * 0.3
        
        return delays


@dataclass
class MonteCarloParameters:
    """Parameters for Monte Carlo simulation."""
    n_simulations: int = 1000
    oil_price_volatility: float = 0.25
    cost_uncertainty: float = 0.15
    mechanical_failure_rate: float = 0.07
    price_cost_correlation: float = 0.0
    scenarios: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        if self.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if not 0 <= self.oil_price_volatility <= 1:
            raise ValueError("Volatility must be between 0 and 1")
        if not 0 <= self.cost_uncertainty <= 1:
            raise ValueError("Cost uncertainty must be between 0 and 1")


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    npv_distribution: np.ndarray
    mean_npv: float
    std_npv: float
    p10_npv: float
    p50_npv: float
    p90_npv: float
    probability_positive: float
    probability_exceeds_hurdle: float = 0.0
    var_95: Optional[float] = None  # Value at Risk
    cvar_95: Optional[float] = None  # Conditional Value at Risk
    scenario_results: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        # Calculate VaR and CVaR
        self.var_95 = np.percentile(self.npv_distribution, 5)
        # CVaR is mean of values below VaR
        below_var = self.npv_distribution[self.npv_distribution <= self.var_95]
        self.cvar_95 = np.mean(below_var) if len(below_var) > 0 else self.var_95


def run_monte_carlo_npv(
    wells: List[WellEconomics],
    mc_params: MonteCarloParameters,
    base_econ_params: Optional[EconomicParameters] = None,
    hurdle_rate: float = 0.15
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation for field NPV.
    
    Args:
        wells: List of wells to evaluate
        mc_params: Monte Carlo parameters
        base_econ_params: Base economic parameters (default if None)
        hurdle_rate: Minimum acceptable return
        
    Returns:
        Monte Carlo results with statistics
    """
    if base_econ_params is None:
        base_econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
    
    n_sims = mc_params.n_simulations
    n_wells = len(wells)
    
    # Initialize models
    price_model = OilPriceModel(
        base_price=base_econ_params.oil_price,
        volatility=mc_params.oil_price_volatility
    )
    
    cost_model = CostUncertaintyModel(
        base_cost=np.mean([w.capex for w in wells]),
        uncertainty=mc_params.cost_uncertainty
    )
    
    # Generate price paths and costs
    max_months = max(w.months for w in wells)
    price_paths = price_model.generate_price_path(max_months, n_sims)
    well_costs = cost_model.generate_well_costs(n_wells, n_sims)
    
    # Run simulations
    npv_results = np.zeros(n_sims)
    
    for sim in range(n_sims):
        # Create economic parameters for this simulation
        sim_econ = EconomicParameters(
            oil_price=np.mean(price_paths[sim, :]),  # Use average price
            discount_rate=base_econ_params.discount_rate,
            opex_per_boe=base_econ_params.opex_per_boe,
            royalty=base_econ_params.royalty,
            working_interest=base_econ_params.working_interest,
            severance_tax=base_econ_params.severance_tax,
            ad_valorem_tax=base_econ_params.ad_valorem_tax
        )
        
        # Calculate field NPV with simulated costs
        total_npv = 0
        for i, well in enumerate(wells):
            # Create well with simulated cost
            sim_well = WellEconomics(
                name=well.name,
                capex=well_costs[sim, i],
                ip_rate=well.ip_rate,
                di=well.di,
                b=well.b,
                months=well.months,
                eur_mboe=well.eur_mboe
            )
            
            # Add to total NPV
            total_npv += calculate_well_npv(sim_well, sim_econ, start_month=i)
        
        npv_results[sim] = total_npv
    
    # Handle scenarios if provided
    scenario_results = None
    if mc_params.scenarios:
        scenario_results = {}
        for scenario_name, scenario_params in mc_params.scenarios.items():
            # Run subset of simulations for each scenario
            scenario_price = scenario_params.get('oil_price_base', base_econ_params.oil_price)
            scenario_prob = scenario_params.get('probability', 1.0 / len(mc_params.scenarios))
            
            scenario_econ = EconomicParameters(
                oil_price=scenario_price,
                discount_rate=base_econ_params.discount_rate,
                opex_per_boe=base_econ_params.opex_per_boe,
                royalty=base_econ_params.royalty,
                working_interest=base_econ_params.working_interest
            )
            
            # Calculate deterministic NPV for scenario
            scenario_npv = sum(calculate_well_npv(w, scenario_econ, i) for i, w in enumerate(wells))
            
            scenario_results[scenario_name] = {
                'mean_npv': scenario_npv,
                'probability': scenario_prob
            }
    
    # Calculate statistics
    return MonteCarloResults(
        npv_distribution=npv_results,
        mean_npv=np.mean(npv_results),
        std_npv=np.std(npv_results),
        p10_npv=np.percentile(npv_results, 10),
        p50_npv=np.percentile(npv_results, 50),
        p90_npv=np.percentile(npv_results, 90),
        probability_positive=np.mean(npv_results > 0),
        probability_exceeds_hurdle=np.mean(npv_results / sum(w.capex for w in wells) > hurdle_rate),
        scenario_results=scenario_results
    )