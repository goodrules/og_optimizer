"""
Google Cloud Vizier integration for Oil & Gas Field Development Optimizer.
Provides Bayesian optimization as an alternative to heuristic search.
"""
import os
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy

#from dotenv import load_dotenv

PROJECT_ID = os.environ.get("PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.vizier import Study, pyvizier
    VIZIER_AVAILABLE = True
except ImportError:
    VIZIER_AVAILABLE = False
    # Create mock classes for type hints when Vizier not available
    Study = None
    pyvizier = None

from .heuristic_optimizer import (
    OptimizationKnobs, 
    OptimizationMetrics, 
    TrialHistory,
    evaluate_scenario
)
from .economics import WellEconomics

# Load environment variables
#load_dotenv()


@dataclass
class VizierTrialHistory:
    """Trial history compatible with existing TrialHistory interface."""
    trials: List[Dict[str, Any]] = field(default_factory=list)
    best_trial_idx: int = -1
    vizier_study: Any = None
    
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


class VizierOptimizer:
    """
    Google Cloud Vizier Bayesian optimization for field development.
    Provides intelligent parameter exploration superior to random search.
    """
    
    def __init__(
        self,
        available_wells: List[WellEconomics],
        lease_limits: Dict[str, int],
        capex_budget: float,
        n_trials: int = 50,
        improvement_threshold: float = 0.95,
        locked_parameters: Optional[Dict[str, Any]] = None,
        run_monte_carlo: bool = False,
        mc_simulations: int = 100,
        project_id: Optional[str] = None,
        region: Optional[str] = None
    ):
        if not VIZIER_AVAILABLE:
            raise ImportError(
                "Google Cloud AI Platform not available. "
                "Install with: pip install google-cloud-aiplatform"
            )
        
        self.available_wells = available_wells
        self.lease_limits = lease_limits
        self.capex_budget = capex_budget
        self.n_trials = n_trials
        self.improvement_threshold = improvement_threshold
        self.locked_parameters = locked_parameters or {}
        self.run_monte_carlo = run_monte_carlo
        self.mc_simulations = mc_simulations
        
        # GCP configuration using existing pattern
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.region = region or os.getenv("GCP_REGION", "us-central1")
        
        if not self.project_id:
            raise ValueError("GCP Project ID not found. Set GCP_PROJECT_ID environment variable.")
        
        # Initialize AI Platform
        try:
            aiplatform.init(project=self.project_id, location=self.region)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud AI Platform: {e}")
        
        self.study = None
        self.history = VizierTrialHistory()
    
    def _create_study_config(self) -> Any:
        """Create Vizier study configuration with parameter space."""
        problem = pyvizier.StudyConfig()
        problem.algorithm = pyvizier.Algorithm.ALGORITHM_UNSPECIFIED  # Uses Bayesian optimization
        
        # Define parameter space
        root = problem.search_space.select_root()
        
        # Wells per lease (integer parameters)
        lease_params = {
            "MIDLAND_A": 28,
            "MARTIN_B": 15, 
            "REEVES_C": 32,
            "LOVING_D": 22,
            "HOWARD_E": 12
        }
        
        for lease, max_wells in lease_params.items():
            if lease in self.lease_limits:
                # Only add parameter to search space if not locked
                wells_param = f"wells_{lease}"
                if wells_param not in self.locked_parameters:
                    actual_limit = min(max_wells, self.lease_limits[lease])
                    root.add_int_param(wells_param, 0, actual_limit)
        
        # Rig count (only if not locked)
        if "rig_count" not in self.locked_parameters:
            root.add_int_param("rig_count", 1, 5)
        
        # Categorical parameters (only if not locked)
        if "drilling_mode" not in self.locked_parameters:
            root.add_categorical_param("drilling_mode", ["continuous", "batch"])
        if "permit_strategy" not in self.locked_parameters:
            root.add_categorical_param("permit_strategy", ["aggressive", "balanced", "conservative"])
        if "development_pace" not in self.locked_parameters:
            root.add_categorical_param("development_pace", ["slow", "moderate", "fast"])
        
        # Continuous parameters (only if not locked)
        if "contingency_percent" not in self.locked_parameters:
            root.add_float_param("contingency_percent", 0.05, 0.30, scale_type=pyvizier.ScaleType.LINEAR)
        if "hurdle_rate" not in self.locked_parameters:
            root.add_float_param("hurdle_rate", 0.10, 0.30, scale_type=pyvizier.ScaleType.LINEAR)
        if "oil_price_forecast" not in self.locked_parameters:
            root.add_float_param("oil_price_forecast", 30.0, 120.0, scale_type=pyvizier.ScaleType.LINEAR)
        if "price_volatility" not in self.locked_parameters:
            root.add_float_param("price_volatility", 0.1, 0.5, scale_type=pyvizier.ScaleType.LINEAR)
        
        # Objective: Maximize NPV
        problem.metric_information.append(
            pyvizier.MetricInformation(name="total_npv", goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
        )
        
        return problem
    
    def _parameters_to_knobs(self, parameters: Dict[str, Any]) -> OptimizationKnobs:
        """Convert Vizier trial parameters to OptimizationKnobs."""
        # Extract wells per lease
        wells_per_lease = {}
        for lease in ["MIDLAND_A", "MARTIN_B", "REEVES_C", "LOVING_D", "HOWARD_E"]:
            param_name = f"wells_{lease}"
            if param_name in self.locked_parameters:
                # Use locked value
                wells_per_lease[lease] = self.locked_parameters[param_name]
            elif param_name in parameters:
                wells_per_lease[lease] = int(parameters[param_name].value)
            elif lease in self.lease_limits:
                # Default to minimum if parameter not found
                wells_per_lease[lease] = 0
        
        # Extract other parameters (use locked values if available)
        rig_count = self.locked_parameters.get("rig_count", int(parameters["rig_count"].value) if "rig_count" in parameters else 2)
        drilling_mode = self.locked_parameters.get("drilling_mode", str(parameters["drilling_mode"].value) if "drilling_mode" in parameters else "continuous")
        permit_strategy = self.locked_parameters.get("permit_strategy", str(parameters["permit_strategy"].value) if "permit_strategy" in parameters else "balanced")
        development_pace = self.locked_parameters.get("development_pace", str(parameters["development_pace"].value) if "development_pace" in parameters else "moderate")
        contingency_percent = self.locked_parameters.get("contingency_percent", float(parameters["contingency_percent"].value) if "contingency_percent" in parameters else 0.15)
        hurdle_rate = self.locked_parameters.get("hurdle_rate", float(parameters["hurdle_rate"].value) if "hurdle_rate" in parameters else 0.15)
        oil_price_forecast = self.locked_parameters.get("oil_price_forecast", float(parameters["oil_price_forecast"].value) if "oil_price_forecast" in parameters else 80.0)
        price_volatility = self.locked_parameters.get("price_volatility", float(parameters["price_volatility"].value) if "price_volatility" in parameters else 0.25)
        
        return OptimizationKnobs(
            wells_per_lease=wells_per_lease,
            rig_count=rig_count,
            drilling_mode=drilling_mode,
            contingency_percent=contingency_percent,
            hurdle_rate=hurdle_rate,
            oil_price_forecast=oil_price_forecast,
            price_volatility=price_volatility,
            permit_strategy=permit_strategy,
            development_pace=development_pace
        )
    
    def _knobs_to_parameters(self, knobs: OptimizationKnobs) -> Dict[str, Any]:
        """Convert OptimizationKnobs to Vizier parameter dictionary."""
        params = {}
        
        # Wells per lease
        for lease, count in knobs.wells_per_lease.items():
            params[f"wells_{lease}"] = count
        
        # Other parameters
        params.update({
            "rig_count": knobs.rig_count,
            "drilling_mode": knobs.drilling_mode,
            "permit_strategy": knobs.permit_strategy,
            "development_pace": knobs.development_pace,
            "contingency_percent": knobs.contingency_percent,
            "hurdle_rate": knobs.hurdle_rate,
            "oil_price_forecast": knobs.oil_price_forecast,
            "price_volatility": knobs.price_volatility
        })
        
        return params
    
    def optimize(self) -> Tuple[OptimizationKnobs, OptimizationMetrics, VizierTrialHistory]:
        """Run Vizier Bayesian optimization."""
        print(f"\n--- Vizier Bayesian Optimization ---")
        print(f"Project: {self.project_id}, Region: {self.region}")
        print(f"Target trials: {self.n_trials}")
        
        # Create study
        problem = self._create_study_config()
        study_name = f"oil_gas_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.study = Study.create_or_load(display_name=study_name, problem=problem)
            self.history.vizier_study = self.study
            print(f"Created Vizier study: {study_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to create Vizier study: {e}")
        
        # Optimization loop
        trials_per_batch = 5
        completed_trials = 0
        
        while completed_trials < self.n_trials:
            remaining_trials = self.n_trials - completed_trials
            batch_size = min(trials_per_batch, remaining_trials)
            
            try:
                # Get trial suggestions from Vizier
                trials = self.study.suggest(count=batch_size)
                print(f"\nTrial batch {completed_trials//trials_per_batch + 1}: {len(trials)} trials")
                
                for i, trial in enumerate(trials):
                    try:
                        print(f"  Trial {completed_trials + i + 1}/{self.n_trials}: Evaluating...")
                        
                        # Convert Vizier parameters to OptimizationKnobs
                        knobs = self._parameters_to_knobs(trial.parameters)
                        
                        # Evaluate using existing pipeline
                        metrics = evaluate_scenario(
                            knobs, 
                            self.available_wells, 
                            self.capex_budget,
                            run_monte_carlo=self.run_monte_carlo,
                            n_simulations=self.mc_simulations
                        )
                        
                        # Report result back to Vizier
                        measurement = pyvizier.Measurement()
                        measurement.metrics["total_npv"] = metrics.total_npv
                        
                        trial.add_measurement(measurement)
                        trial.complete(measurement)
                        
                        # Add to history
                        self.history.add_trial(knobs, metrics)
                        
                        print(f"    NPV: ${metrics.total_npv/1e6:.1f}MM")
                        
                    except Exception as e:
                        print(f"    Trial failed: {e}")
                        # Report as infeasible
                        trial.complete(infeasible_reason=str(e))
                    
                    completed_trials += 1
                    
            except Exception as e:
                print(f"Batch suggestion failed: {e}")
                break
        
        # Get best result
        best_trial = self.history.get_best_trial()
        if best_trial:
            print(f"\nOptimization complete!")
            print(f"Best NPV: ${best_trial['metrics'].total_npv/1e6:.1f}MM")
            print(f"Total trials: {len(self.history.trials)}")
            
            return best_trial["knobs"], best_trial["metrics"], self.history
        else:
            raise RuntimeError("No successful trials completed")


def is_vizier_available() -> bool:
    """Check if Vizier dependencies are available."""
    return VIZIER_AVAILABLE


def check_vizier_setup() -> Tuple[bool, str]:
    """Check if Vizier is properly configured."""
    if not VIZIER_AVAILABLE:
        return False, "Google Cloud AI Platform not installed"
    
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        return False, "GCP_PROJECT_ID environment variable not set"
    
    try:
        # Try to initialize (this will check authentication)
        region = os.getenv("GCP_REGION", "us-central1")
        aiplatform.init(project=project_id, location=region)
        return True, "Vizier ready"
    except Exception as e:
        return False, f"GCP authentication failed: {e}"