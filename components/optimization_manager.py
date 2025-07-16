"""
Unified optimization interface supporting both Heuristic and Vizier methods.
Provides seamless method selection and automatic fallback mechanisms.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from .heuristic_optimizer import (
    HeuristicOptimizer,
    OptimizationKnobs, 
    OptimizationMetrics, 
    TrialHistory
)
from .vizier_optimizer import (
    VizierOptimizer, 
    VizierTrialHistory,
    is_vizier_available,
    check_vizier_setup
)
from .economics import WellEconomics


class OptimizationMethod(Enum):
    """Available optimization methods."""
    HEURISTIC = "heuristic"
    VIZIER = "vizier"


class OptimizationManager:
    """
    Unified interface for both Heuristic and Vizier optimization methods.
    Handles method selection, execution, and automatic fallback.
    """
    
    def __init__(
        self,
        method: Union[str, OptimizationMethod] = OptimizationMethod.HEURISTIC,
        available_wells: Optional[List[WellEconomics]] = None,
        lease_limits: Optional[Dict[str, int]] = None,
        capex_budget: Optional[float] = None
    ):
        """
        Initialize optimization manager.
        
        Args:
            method: Optimization method to use ("heuristic" or "vizier")
            available_wells: List of wells to consider
            lease_limits: Maximum wells per lease
            capex_budget: Total capital budget
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            method = OptimizationMethod(method.lower())
        
        self.method = method
        self.available_wells = available_wells
        self.lease_limits = lease_limits
        self.capex_budget = capex_budget
        
        # Optimizer instances
        self.heuristic_optimizer = None
        self.vizier_optimizer = None
        
        # Track actual method used (may differ from requested due to fallback)
        self.actual_method_used = None
        self.fallback_reason = None
    
    def optimize(
        self,
        available_wells: Optional[List[WellEconomics]] = None,
        lease_limits: Optional[Dict[str, int]] = None,
        capex_budget: Optional[float] = None,
        n_trials: Optional[int] = None,
        **kwargs
    ) -> Tuple[OptimizationKnobs, OptimizationMetrics, Union[TrialHistory, VizierTrialHistory]]:
        """
        Run optimization using the selected method with automatic fallback.
        
        Args:
            available_wells: Wells to consider (uses constructor value if None)
            lease_limits: Lease limits (uses constructor value if None)
            capex_budget: Capital budget (uses constructor value if None)
            n_trials: Number of trials (method-specific defaults if None)
            **kwargs: Additional method-specific arguments
            
        Returns:
            Tuple of (best_knobs, best_metrics, trial_history)
        """
        # Use provided values or fall back to constructor values
        wells = available_wells or self.available_wells
        limits = lease_limits or self.lease_limits
        budget = capex_budget or self.capex_budget
        
        if not all([wells, limits, budget]):
            raise ValueError("Must provide available_wells, lease_limits, and capex_budget")
        
        # Set method-specific defaults for n_trials
        if n_trials is None:
            n_trials = 50 if self.method == OptimizationMethod.VIZIER else 20
        
        # Try requested method first
        if self.method == OptimizationMethod.VIZIER:
            try:
                return self._optimize_with_vizier(wells, limits, budget, n_trials, **kwargs)
            except Exception as e:
                print(f"Vizier optimization failed: {e}")
                print("Falling back to heuristic optimization...")
                self.actual_method_used = OptimizationMethod.HEURISTIC
                self.fallback_reason = str(e)
                # Fall through to heuristic
        
        # Heuristic optimization (default or fallback)
        self.actual_method_used = OptimizationMethod.HEURISTIC
        return self._optimize_with_heuristic(wells, limits, budget, n_trials, **kwargs)
    
    def _optimize_with_vizier(
        self, 
        wells: List[WellEconomics], 
        limits: Dict[str, int], 
        budget: float,
        n_trials: int,
        **kwargs
    ) -> Tuple[OptimizationKnobs, OptimizationMetrics, VizierTrialHistory]:
        """Run Vizier Bayesian optimization."""
        self.vizier_optimizer = VizierOptimizer(
            available_wells=wells,
            lease_limits=limits,
            capex_budget=budget,
            n_trials=n_trials,
            **kwargs
        )
        
        result = self.vizier_optimizer.optimize()
        self.actual_method_used = OptimizationMethod.VIZIER
        return result
    
    def _optimize_with_heuristic(
        self, 
        wells: List[WellEconomics], 
        limits: Dict[str, int], 
        budget: float,
        n_trials: int,
        **kwargs
    ) -> Tuple[OptimizationKnobs, OptimizationMetrics, TrialHistory]:
        """Run heuristic guided random search."""
        self.heuristic_optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits=limits,
            capex_budget=budget,
            n_trials=n_trials,
            **kwargs
        )
        
        result = self.heuristic_optimizer.optimize()
        self.actual_method_used = OptimizationMethod.HEURISTIC
        return result
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about optimization methods and their availability."""
        return {
            "available_methods": {
                "heuristic": {
                    "name": "Heuristic Optimization",
                    "description": "Fast guided random search",
                    "available": True,
                    "typical_trials": 20,
                    "strengths": ["Fast", "No external dependencies", "Reliable"],
                    "use_cases": ["Quick analysis", "Parameter exploration", "Baseline optimization"]
                },
                "vizier": {
                    "name": "Vizier Bayesian Optimization", 
                    "description": "Advanced Bayesian optimization with intelligent parameter exploration",
                    "available": is_vizier_available(),
                    "setup_ok": check_vizier_setup()[0] if is_vizier_available() else False,
                    "setup_message": check_vizier_setup()[1] if is_vizier_available() else "Not installed",
                    "typical_trials": 50,
                    "strengths": ["Superior convergence", "Fewer trials needed", "Intelligent exploration"],
                    "use_cases": ["Production optimization", "Fine-tuning", "Complex parameter spaces"],
                    "requirements": ["GCP project", "Authentication", "google-cloud-aiplatform"]
                }
            },
            "current_method": self.method.value,
            "actual_method_used": self.actual_method_used.value if self.actual_method_used else None,
            "fallback_occurred": self.fallback_reason is not None,
            "fallback_reason": self.fallback_reason
        }
    
    def can_use_vizier(self) -> Tuple[bool, str]:
        """Check if Vizier optimization is available and properly configured."""
        return check_vizier_setup()
    
    def set_method(self, method: Union[str, OptimizationMethod]):
        """Change the optimization method."""
        if isinstance(method, str):
            method = OptimizationMethod(method.lower())
        self.method = method
        
        # Reset tracking
        self.actual_method_used = None
        self.fallback_reason = None
    
    def get_optimizer_instance(self):
        """Get the active optimizer instance (for advanced users)."""
        if self.actual_method_used == OptimizationMethod.VIZIER:
            return self.vizier_optimizer
        else:
            return self.heuristic_optimizer


# Convenience functions for backward compatibility
def create_optimizer(
    method: str = "heuristic",
    available_wells: List[WellEconomics] = None,
    lease_limits: Dict[str, int] = None,
    capex_budget: float = None
) -> OptimizationManager:
    """Create an OptimizationManager with specified method."""
    return OptimizationManager(
        method=method,
        available_wells=available_wells,
        lease_limits=lease_limits,
        capex_budget=capex_budget
    )


def get_available_methods() -> Dict[str, Any]:
    """Get information about available optimization methods."""
    manager = OptimizationManager()
    return manager.get_method_info()["available_methods"]