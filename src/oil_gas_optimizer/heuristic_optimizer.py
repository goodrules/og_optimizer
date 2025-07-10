"""
Heuristic optimization for oil & gas field development.
Adapted from the perturb_knobs approach in local/main.py.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
import math
from copy import deepcopy

from .economics import WellEconomics, EconomicParameters
from .drilling_optimizer import (
    DrillingConstraints, 
    DrillingScheduleOptimizer,
    OptimizationObjective,
    solve_drilling_schedule
)
from .data_model import DrillingParameters
from .monte_carlo import MonteCarloParameters, run_monte_carlo_npv


def _get_well_ip_rate(well: WellEconomics) -> float:
    """Helper function to get well IP rate for sorting."""
    return well.ip_rate


@dataclass
class OptimizationKnobs:
    """Adjustable parameters for field development optimization."""
    wells_per_lease: Dict[str, int]
    rig_count: int
    drilling_mode: str = "continuous"  # continuous or batch
    contingency_percent: float = 0.15
    hurdle_rate: float = 0.15
    oil_price_forecast: float = 80.0
    price_volatility: float = 0.25
    permit_strategy: str = "balanced"  # aggressive, conservative, balanced
    development_pace: str = "moderate"  # slow, moderate, fast
    
    def __post_init__(self):
        if self.rig_count <= 0:
            raise ValueError("Rig count must be positive")
        if not 0 <= self.contingency_percent <= 0.5:
            raise ValueError("Contingency must be between 0 and 0.5")
        if not 0 <= self.hurdle_rate <= 0.5:
            raise ValueError("Hurdle rate must be between 0 and 0.5")
        if self.oil_price_forecast <= 0:
            raise ValueError("Oil price must be positive")
        if not 0 <= self.price_volatility <= 1:
            raise ValueError("Price volatility must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "wells_per_lease": self.wells_per_lease.copy(),
            "rig_count": self.rig_count,
            "drilling_mode": self.drilling_mode,
            "contingency_percent": self.contingency_percent,
            "hurdle_rate": self.hurdle_rate,
            "oil_price_forecast": self.oil_price_forecast,
            "price_volatility": self.price_volatility,
            "permit_strategy": self.permit_strategy,
            "development_pace": self.development_pace
        }
    
    def __eq__(self, other):
        """Check equality for testing."""
        if not isinstance(other, OptimizationKnobs):
            return False
        return self.to_dict() == other.to_dict()
    
    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)


@dataclass
class OptimizationMetrics:
    """Metrics for evaluating a drilling scenario."""
    total_npv: float
    total_capex: float
    npv_per_dollar: float = 0.0
    peak_production: float = 0.0
    wells_drilled: int = 0
    risk_score: float = 0.5  # 0 = low risk, 1 = high risk
    # Uncertainty metrics (if Monte Carlo run)
    p10_npv: Optional[float] = None
    p90_npv: Optional[float] = None
    probability_positive: Optional[float] = None
    
    def __post_init__(self):
        if self.total_capex > 0:
            self.npv_per_dollar = self.total_npv / self.total_capex


@dataclass
class DrillingScenario:
    """A complete drilling scenario for evaluation."""
    selected_wells: List[WellEconomics]
    constraints: DrillingConstraints
    econ_params: EconomicParameters
    drilling_params: Optional[DrillingParameters] = None
    run_monte_carlo: bool = False
    mc_simulations: int = 100
    
    @staticmethod
    def from_knobs(
        knobs: OptimizationKnobs,
        available_wells: List[WellEconomics],
        base_capex_budget: float = 100_000_000
    ) -> 'DrillingScenario':
        """Create a drilling scenario from optimization knobs."""
        # Select wells based on knobs
        selected_wells = []
        for lease, count in knobs.wells_per_lease.items():
            lease_wells = [w for w in available_wells if lease in w.name]
            # Sort by NPV potential (simplified - by IP rate)
            lease_wells.sort(key=_get_well_ip_rate, reverse=True)
            selected_wells.extend(lease_wells[:count])
        
        # Create constraints
        budget_after_contingency = base_capex_budget * (1 - knobs.contingency_percent)
        constraints = DrillingConstraints(
            max_rigs_available=knobs.rig_count,
            total_capex_budget=budget_after_contingency,
            planning_horizon_months=24,
            lease_well_limits=knobs.wells_per_lease
        )
        
        # Create economic parameters
        econ_params = EconomicParameters(
            oil_price=knobs.oil_price_forecast,
            discount_rate=knobs.hurdle_rate,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        # Create drilling parameters based on strategy
        permit_delays = {
            "aggressive": 15,
            "balanced": 30,
            "conservative": 45
        }
        
        drilling_params = DrillingParameters(
            drill_days_per_well=25,
            rig_move_days=5,
            batch_drilling_efficiency=0.85 if knobs.drilling_mode == "batch" else 1.0,
            permit_delay_days=permit_delays.get(knobs.permit_strategy, 30),
            weather_delay_factor=0.05
        )
        
        return DrillingScenario(
            selected_wells=selected_wells,
            constraints=constraints,
            econ_params=econ_params,
            drilling_params=drilling_params
        )


class ScenarioEvaluator:
    """Evaluates drilling scenarios and calculates metrics."""
    
    def evaluate(self, scenario: DrillingScenario) -> OptimizationMetrics:
        """Evaluate a drilling scenario and return metrics."""
        # Run optimization
        optimizer = DrillingScheduleOptimizer(
            wells=scenario.selected_wells,
            constraints=scenario.constraints,
            objective=OptimizationObjective.MAXIMIZE_NPV,
            econ_params=scenario.econ_params
        )
        
        schedule = optimizer.solve()
        
        # Calculate base metrics
        schedule_metrics = schedule.calculate_metrics(scenario.econ_params)
        
        metrics = OptimizationMetrics(
            total_npv=schedule_metrics['total_npv'],
            total_capex=schedule_metrics['total_capex'],
            peak_production=schedule_metrics.get('peak_production_boed', 0),
            wells_drilled=schedule_metrics.get('wells_drilled_count', 0)
        )
        
        # Run Monte Carlo if requested
        if scenario.run_monte_carlo and schedule.wells_drilled:
            mc_params = MonteCarloParameters(
                n_simulations=scenario.mc_simulations,
                oil_price_volatility=0.25,
                cost_uncertainty=0.15
            )
            
            mc_results = run_monte_carlo_npv(
                schedule.wells_drilled,
                mc_params,
                scenario.econ_params
            )
            
            metrics.p10_npv = mc_results.p10_npv
            metrics.p90_npv = mc_results.p90_npv
            metrics.probability_positive = mc_results.probability_positive
            
            # Calculate risk score
            if mc_results.mean_npv > 0:
                risk_score = 1 - (mc_results.p10_npv / mc_results.mean_npv)
                metrics.risk_score = max(0, min(1, risk_score))
        
        return metrics


def perturb_knobs(base_knobs: OptimizationKnobs, scale: float = 0.5, locked_params: Optional[Dict[str, bool]] = None) -> OptimizationKnobs:
    """
    Perturb optimization knobs for exploration.
    Adapted from local/main.py perturb_knobs approach.
    
    Args:
        base_knobs: Current best knobs
        scale: Scale factor for perturbation (0-1)
        locked_params: Dict of parameter names to lock states (True = locked)
        
    Returns:
        New perturbed knobs
    """
    new_knobs = deepcopy(base_knobs)
    locked = locked_params or {}
    
    # Perturb wells per lease
    if not locked.get('wells_per_lease', False):
        for lease in new_knobs.wells_per_lease:
            current = new_knobs.wells_per_lease[lease]
            change = int(random.uniform(-5, 5) * scale)
            new_knobs.wells_per_lease[lease] = max(0, min(32, current + change))
    
    # Perturb rig count
    if not locked.get('rig_count', False) and random.random() < 0.3 * scale:
        new_knobs.rig_count = max(1, min(5, 
            new_knobs.rig_count + random.choice([-1, 1])))
    
    # Perturb drilling mode
    if not locked.get('drilling_mode', False) and random.random() < 0.2 * scale:
        new_knobs.drilling_mode = "batch" if new_knobs.drilling_mode == "continuous" else "continuous"
    
    # Perturb contingency
    if not locked.get('contingency_percent', False):
        new_knobs.contingency_percent = max(0.05, min(0.30,
            new_knobs.contingency_percent + random.uniform(-0.05, 0.05) * scale))
    
    # Perturb hurdle rate
    if not locked.get('hurdle_rate', False):
        new_knobs.hurdle_rate = max(0.10, min(0.30,
            new_knobs.hurdle_rate + random.uniform(-0.05, 0.05) * scale))
    
    # Perturb oil price forecast
    if not locked.get('oil_price_forecast', False):
        new_knobs.oil_price_forecast = max(30, min(120,
            new_knobs.oil_price_forecast + random.uniform(-10, 10) * scale))
    
    # Perturb price volatility
    new_knobs.price_volatility = max(0.1, min(0.5,
        new_knobs.price_volatility + random.uniform(-0.1, 0.1) * scale))
    
    # Perturb strategies
    if random.random() < 0.25 * scale:
        strategies = ["aggressive", "balanced", "conservative"]
        new_knobs.permit_strategy = random.choice(strategies)
    
    if random.random() < 0.25 * scale:
        paces = ["slow", "moderate", "fast"]
        new_knobs.development_pace = random.choice(paces)
    
    return new_knobs


def worst_case_knobs() -> OptimizationKnobs:
    """Generate worst-case scenario knobs."""
    # Start with minimal wells on each lease
    wells_per_lease = {
        "LEASE001": 1,
        "LEASE002": 1, 
        "LEASE003": 1,
        "LEASE004": 1,
        "LEASE005": 1,
        "LEASE006": 1,
        "LEASE007": 1,
        "LEASE008": 1,
        "LEASE009": 1,
        "LEASE010": 1
    }
    return OptimizationKnobs(
        wells_per_lease=wells_per_lease,  # Minimal wells
        rig_count=1,  # Minimum rigs
        drilling_mode="continuous",  # Less efficient
        contingency_percent=0.30,  # High contingency
        hurdle_rate=0.25,  # High hurdle rate
        oil_price_forecast=45.0,  # Low oil price
        price_volatility=0.40,  # High volatility
        permit_strategy="conservative",  # Slow permitting
        development_pace="slow"  # Slow development
    )


def evaluate_scenario(
    knobs: OptimizationKnobs,
    available_wells: List[WellEconomics],
    capex_budget: float
) -> OptimizationMetrics:
    """Evaluate a set of knobs by creating and evaluating a scenario."""
    scenario = DrillingScenario.from_knobs(knobs, available_wells, capex_budget)
    evaluator = ScenarioEvaluator()
    return evaluator.evaluate(scenario)


@dataclass
class TrialHistory:
    """Track optimization trial history."""
    trials: List[Dict[str, Any]] = field(default_factory=list)
    best_trial_idx: int = -1
    
    def add_trial(self, knobs: OptimizationKnobs, metrics: OptimizationMetrics):
        """Add a trial to history."""
        self.trials.append({
            "knobs": deepcopy(knobs),
            "metrics": metrics,
            "trial_num": len(self.trials)
        })
        
        # Update best trial
        if self.best_trial_idx == -1 or metrics.total_npv > self.trials[self.best_trial_idx]["metrics"].total_npv:
            self.best_trial_idx = len(self.trials) - 1
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Get the best trial."""
        if self.best_trial_idx >= 0:
            return self.trials[self.best_trial_idx]
        return None
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics across trials."""
        if not self.trials:
            return {}
        
        npvs = [t["metrics"].total_npv for t in self.trials]
        return {
            "mean_npv": sum(npvs) / len(npvs),
            "best_npv": max(npvs),
            "worst_npv": min(npvs),
            "improvement_ratio": max(npvs) / min(npvs) if min(npvs) > 0 else float('inf')
        }


class HeuristicOptimizer:
    """
    Heuristic optimizer for field development using guided random search.
    Based on the approach in local/main.py.
    """
    
    def __init__(
        self,
        available_wells: List[WellEconomics],
        lease_limits: Dict[str, int],
        capex_budget: float,
        n_trials: int = 20,
        improvement_threshold: float = 0.95
    ):
        self.available_wells = available_wells
        self.lease_limits = lease_limits
        self.capex_budget = capex_budget
        self.n_trials = n_trials
        self.improvement_threshold = improvement_threshold
        
        self.best_knobs = None
        self.best_score = float('-inf')
        self.history = TrialHistory()
    
    def optimize(self) -> Tuple[OptimizationKnobs, OptimizationMetrics, TrialHistory]:
        """Run heuristic optimization."""
        # Start with worst case
        current_knobs = worst_case_knobs()
        # Update with actual lease limits
        current_knobs.wells_per_lease = {
            lease: min(3, limit) for lease, limit in self.lease_limits.items()
        }
        
        # Evaluate initial scenario
        metrics = evaluate_scenario(current_knobs, self.available_wells, self.capex_budget)
        self.history.add_trial(current_knobs, metrics)
        
        if metrics.total_npv > self.best_score:
            self.best_knobs = deepcopy(current_knobs)
            self.best_score = metrics.total_npv
        
        # Run optimization trials
        for trial in range(1, self.n_trials):
            # Decay scale factor over time (more exploration early, refinement later)
            scale = max(0.2, 1.0 - (trial / self.n_trials) * 0.7)
            
            # Perturb from best known solution
            trial_knobs = perturb_knobs(self.best_knobs, scale)
            
            # Ensure lease limits are respected
            for lease in trial_knobs.wells_per_lease:
                if lease in self.lease_limits:
                    trial_knobs.wells_per_lease[lease] = min(
                        trial_knobs.wells_per_lease[lease],
                        self.lease_limits[lease]
                    )
            
            # Evaluate
            metrics = evaluate_scenario(trial_knobs, self.available_wells, self.capex_budget)
            self.history.add_trial(trial_knobs, metrics)
            
            # Update best if improved
            if metrics.total_npv > self.best_score * self.improvement_threshold:
                self.best_knobs = deepcopy(trial_knobs)
                self.best_score = metrics.total_npv
        
        # Return best solution
        best_trial = self.history.get_best_trial()
        return best_trial["knobs"], best_trial["metrics"], self.history