"""
Drilling sequence optimization using OR-Tools CP-SAT solver.
Adapted from the task scheduling model in local/main.py.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta
from enum import Enum
import math

from ortools.sat.python import cp_model
import numpy as np

from .economics import WellEconomics, EconomicParameters, calculate_well_npv
from .data_model import Lease, DrillingParameters


def _get_drilling_order_start(item):
    """Helper function to get start month from drilling order tuple."""
    return item[1]


class OptimizationObjective(Enum):
    """Optimization objectives for drilling scheduling."""
    MAXIMIZE_NPV = "maximize_npv"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_PRODUCTION = "maximize_production"
    MINIMIZE_COST = "minimize_cost"


@dataclass
class DrillingConstraints:
    """Constraints for drilling optimization."""
    max_rigs_available: int
    total_capex_budget: float
    planning_horizon_months: int = 24
    max_wells_per_rig_per_month: int = 2
    min_production_target: Optional[float] = None  # boe/d
    lease_well_limits: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.max_rigs_available <= 0:
            raise ValueError("Number of rigs must be positive")
        if self.total_capex_budget <= 0:
            raise ValueError("Budget must be positive")
        if self.planning_horizon_months <= 0:
            raise ValueError("Planning horizon must be positive")


@dataclass
class DrillingDecisionVariables:
    """OR-Tools decision variables for drilling optimization."""
    wells_drilled: Dict[str, List[Any]]  # lease -> [monthly well count vars]
    drilling_start: Dict[str, Any]  # well -> start time var
    rig_assignments: Dict[str, Dict[int, Any]]  # well -> {month: rig assignment var}
    rig_active: Optional[List[List[Any]]] = None  # rig -> [monthly active vars]
    well_selected: Optional[Dict[str, Any]] = None  # well -> boolean selected var
    
    @staticmethod
    def create_for_leases(
        model: cp_model.CpModel,
        leases: List[Lease],
        horizon_months: int,
        max_rigs: Optional[int] = None
    ) -> 'DrillingDecisionVariables':
        """Create decision variables for given leases."""
        wells_drilled = {}
        for lease in leases:
            wells_drilled[lease.id] = []
            for month in range(horizon_months):
                var = model.NewIntVar(
                    0, 
                    min(lease.available_wells, 4),  # Max 4 wells per lease per month
                    f"wells_{lease.id}_m{month}"
                )
                wells_drilled[lease.id].append(var)
        
        # Rig tracking if specified
        rig_active = None
        if max_rigs:
            rig_active = []
            for rig in range(max_rigs):
                rig_months = []
                for month in range(horizon_months):
                    var = model.NewBoolVar(f"rig_{rig}_active_m{month}")
                    rig_months.append(var)
                rig_active.append(rig_months)
        
        return DrillingDecisionVariables(
            wells_drilled=wells_drilled,
            drilling_start={},
            rig_assignments={},
            rig_active=rig_active
        )


@dataclass
class DrillingSchedule:
    """Output of drilling optimization."""
    wells_drilled: List[WellEconomics]
    drilling_order: List[str]  # Well names in order
    start_dates: Dict[str, date]
    total_capex: float
    peak_rigs_used: int
    wells_per_month: Optional[Dict[int, int]] = None
    total_drill_days: Optional[int] = None
    batch_efficiency_gain: Optional[float] = None
    considers_operational_delays: bool = False
    
    def is_feasible(self) -> bool:
        """Check if schedule is feasible."""
        return len(self.wells_drilled) > 0 and self.total_capex > 0
    
    def calculate_metrics(self, econ_params: EconomicParameters) -> Dict[str, float]:
        """Calculate key metrics for the schedule."""
        total_npv = sum(
            calculate_well_npv(well, econ_params, i)
            for i, well in enumerate(self.wells_drilled)
        )
        
        # Peak production per well (average)
        if self.wells_drilled:
            peak_production_per_well = sum(w.ip_rate for w in self.wells_drilled) / len(self.wells_drilled)
        else:
            peak_production_per_well = 0
        
        return {
            'total_npv': total_npv,
            'total_capex': self.total_capex,
            'wells_drilled_count': len(self.wells_drilled),
            'peak_production_boed': peak_production_per_well,
            'capital_efficiency': total_npv / self.total_capex if self.total_capex > 0 else 0
        }


@dataclass
class RigAllocation:
    """Manages rig allocation and scheduling."""
    n_rigs: int
    wells: List[WellEconomics]
    drill_days_per_well: int = 25
    move_days_between_wells: int = 5
    move_days_between_pads: int = 10
    move_days_within_pad: int = 2
    enable_batch_drilling: bool = False
    
    def create_continuous_schedule(self) -> DrillingSchedule:
        """Create continuous drilling schedule."""
        rig_assignments = {}
        current_day = 0
        
        for i, well in enumerate(self.wells):
            rig_assignments[well.name] = self.drill_days_per_well
            if i < len(self.wells) - 1:  # Not the last well
                current_day += self.drill_days_per_well + self.move_days_between_wells
            else:
                current_day += self.drill_days_per_well
        
        return DrillingSchedule(
            wells_drilled=self.wells,
            drilling_order=[w.name for w in self.wells],
            start_dates={w.name: date.today() + timedelta(days=i*30) for i, w in enumerate(self.wells)},
            total_capex=sum(w.capex for w in self.wells),
            peak_rigs_used=self.n_rigs,
            total_drill_days=current_day,
            batch_efficiency_gain=0.0
        )
    
    def create_batch_schedule(self) -> DrillingSchedule:
        """Create batch/pad drilling schedule."""
        # Group wells by pad (simplified - by name prefix)
        pads = {}
        for well in self.wells:
            pad = well.name.split("_")[0]
            if pad not in pads:
                pads[pad] = []
            pads[pad].append(well)
        
        total_days = 0
        for pad_wells in pads.values():
            # First well takes full time
            total_days += self.drill_days_per_well
            # Subsequent wells on same pad are faster
            for i in range(1, len(pad_wells)):
                total_days += int(self.drill_days_per_well * self.enable_batch_drilling * 0.85)
                total_days += self.move_days_within_pad
            # Move to next pad
            total_days += self.move_days_between_pads
        
        # Remove last pad move
        total_days -= self.move_days_between_pads
        
        # Calculate efficiency gain
        continuous_days = len(self.wells) * self.drill_days_per_well + \
                         (len(self.wells) - 1) * self.move_days_between_wells
        efficiency_gain = (continuous_days - total_days) / continuous_days if continuous_days > 0 else 0
        
        return DrillingSchedule(
            wells_drilled=self.wells,
            drilling_order=[w.name for w in self.wells],
            start_dates={w.name: date.today() + timedelta(days=i*25) for i, w in enumerate(self.wells)},
            total_capex=sum(w.capex for w in self.wells),
            peak_rigs_used=self.n_rigs,
            total_drill_days=total_days,
            batch_efficiency_gain=efficiency_gain
        )


class DrillingScheduleOptimizer:
    """
    Optimizes drilling schedules using OR-Tools CP-SAT solver.
    Adapted from the task scheduling approach in local/main.py.
    """
    
    def __init__(
        self,
        wells: List[WellEconomics],
        constraints: DrillingConstraints,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_NPV,
        econ_params: Optional[EconomicParameters] = None
    ):
        self.wells = wells
        self.constraints = constraints
        self.objective = objective
        self.econ_params = econ_params or EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10
        )
        self.model = cp_model.CpModel()
        self.variables = None
        
    def _create_variables(self):
        """Create OR-Tools decision variables."""
        n_wells = len(self.wells)
        horizon = self.constraints.planning_horizon_months
        
        # Binary variable: is well i drilled?
        well_selected = {}
        for i, well in enumerate(self.wells):
            well_selected[well.name] = self.model.NewBoolVar(f"select_{well.name}")
        
        # Integer variable: when does drilling start for well i?
        drilling_start = {}
        for i, well in enumerate(self.wells):
            drilling_start[well.name] = self.model.NewIntVar(
                0, horizon - 1, f"start_{well.name}"
            )
        
        # Rig assignment variables
        rig_assignments = {}
        for i, well in enumerate(self.wells):
            rig_assignments[well.name] = {}
            for month in range(horizon):
                for rig in range(self.constraints.max_rigs_available):
                    var = self.model.NewBoolVar(f"well_{well.name}_month_{month}_rig_{rig}")
                    rig_assignments[well.name][(month, rig)] = var
        
        self.variables = {
            'well_selected': well_selected,
            'drilling_start': drilling_start,
            'rig_assignments': rig_assignments
        }
    
    def _add_constraints(self):
        """Add optimization constraints."""
        horizon = self.constraints.planning_horizon_months
        
        # Budget constraint
        total_cost = sum(
            self.variables['well_selected'][well.name] * int(well.capex)
            for well in self.wells
        )
        self.model.Add(total_cost <= int(self.constraints.total_capex_budget))
        
        # Rig availability constraints
        for month in range(horizon):
            for rig in range(self.constraints.max_rigs_available):
                # Each rig can drill at most one well per month
                wells_on_rig = []
                for well in self.wells:
                    if (month, rig) in self.variables['rig_assignments'][well.name]:
                        wells_on_rig.append(
                            self.variables['rig_assignments'][well.name][(month, rig)]
                        )
                self.model.Add(sum(wells_on_rig) <= 1)
        
        # Well selection implies drilling
        for well in self.wells:
            # If well is selected, it must be assigned to exactly one rig in one month
            selected = self.variables['well_selected'][well.name]
            
            # Total assignments for this well across all months and rigs
            total_assignments = []
            for month in range(horizon):
                for rig in range(self.constraints.max_rigs_available):
                    if (month, rig) in self.variables['rig_assignments'][well.name]:
                        total_assignments.append(
                            self.variables['rig_assignments'][well.name][(month, rig)]
                        )
            
            # If selected, must be assigned exactly once
            self.model.Add(sum(total_assignments) == selected)
        
        # Wells per month constraint
        for month in range(horizon):
            wells_in_month = []
            for well in self.wells:
                for rig in range(self.constraints.max_rigs_available):
                    if (month, rig) in self.variables['rig_assignments'][well.name]:
                        wells_in_month.append(
                            self.variables['rig_assignments'][well.name][(month, rig)]
                        )
            # Total wells in month <= rigs * wells per rig
            self.model.Add(
                sum(wells_in_month) <= 
                self.constraints.max_rigs_available * self.constraints.max_wells_per_rig_per_month
            )
        
        # Production target constraint if specified
        if self.constraints.min_production_target:
            total_production = sum(
                self.variables['well_selected'][well.name] * int(well.ip_rate)
                for well in self.wells
            )
            self.model.Add(total_production >= int(self.constraints.min_production_target))
        
        # Lease limits if specified
        if self.constraints.lease_well_limits:
            for lease_id, limit in self.constraints.lease_well_limits.items():
                lease_wells = [
                    self.variables['well_selected'][well.name]
                    for well in self.wells
                    if well.name.startswith(lease_id)
                ]
                if lease_wells:
                    self.model.Add(sum(lease_wells) <= limit)
    
    def _set_objective(self):
        """Set optimization objective."""
        if self.objective == OptimizationObjective.MAXIMIZE_NPV:
            # Calculate NPV for each well (simplified - ignoring time value)
            npv_values = []
            for i, well in enumerate(self.wells):
                npv = calculate_well_npv(well, self.econ_params, start_month=0)
                npv_values.append(int(npv / 1000))  # Scale down to avoid overflow
            
            objective = sum(
                self.variables['well_selected'][well.name] * npv_values[i]
                for i, well in enumerate(self.wells)
            )
            self.model.Maximize(objective)
            
        elif self.objective == OptimizationObjective.MINIMIZE_COST:
            total_cost = sum(
                self.variables['well_selected'][well.name] * int(well.capex / 1000)
                for well in self.wells
            )
            self.model.Minimize(total_cost)
    
    def solve(self, time_limit_seconds: float = 30.0) -> DrillingSchedule:
        """Solve the optimization problem."""
        # Create variables and constraints
        self._create_variables()
        self._add_constraints()
        self._set_objective()
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        status = solver.Solve(self.model)
        
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Return empty schedule if no solution found
            return DrillingSchedule(
                wells_drilled=[],
                drilling_order=[],
                start_dates={},
                total_capex=0,
                peak_rigs_used=0
            )
        
        # Extract solution
        selected_wells = []
        drilling_order = []
        start_dates = {}
        
        for well in self.wells:
            if solver.Value(self.variables['well_selected'][well.name]):
                selected_wells.append(well)
                start_month = solver.Value(self.variables['drilling_start'][well.name])
                drilling_order.append((well.name, start_month))
                start_dates[well.name] = date.today() + timedelta(days=30 * start_month)
        
        # Sort by start date
        drilling_order.sort(key=_get_drilling_order_start)
        drilling_order = [name for name, _ in drilling_order]
        
        # Calculate metrics
        total_capex = sum(w.capex for w in selected_wells)
        
        # Count peak rig usage from actual assignments
        rig_usage_by_month = {}
        for month in range(self.constraints.planning_horizon_months):
            wells_in_month = 0
            for well in self.wells:
                for rig in range(self.constraints.max_rigs_available):
                    if (month, rig) in self.variables['rig_assignments'][well.name]:
                        if solver.Value(self.variables['rig_assignments'][well.name][(month, rig)]):
                            wells_in_month += 1
            if wells_in_month > 0:
                rig_usage_by_month[month] = wells_in_month
        
        peak_rigs = max(rig_usage_by_month.values()) if rig_usage_by_month else 0
        
        return DrillingSchedule(
            wells_drilled=selected_wells,
            drilling_order=drilling_order,
            start_dates=start_dates,
            total_capex=total_capex,
            peak_rigs_used=peak_rigs,
            wells_per_month=rig_usage_by_month
        )


def solve_drilling_schedule(
    wells: List[WellEconomics],
    max_rigs: int,
    capex_budget: float,
    horizon_months: int = 24,
    min_production: Optional[float] = None,
    drilling_params: Optional[DrillingParameters] = None,
    econ_params: Optional[EconomicParameters] = None
) -> DrillingSchedule:
    """
    High-level function to solve drilling schedule optimization.
    
    Args:
        wells: List of wells to consider
        max_rigs: Maximum rigs available
        capex_budget: Total capital budget
        horizon_months: Planning horizon in months
        min_production: Minimum production target (boe/d)
        drilling_params: Operational parameters
        econ_params: Economic parameters
        
    Returns:
        Optimized drilling schedule
    """
    constraints = DrillingConstraints(
        max_rigs_available=max_rigs,
        total_capex_budget=capex_budget,
        planning_horizon_months=horizon_months,
        min_production_target=min_production
    )
    
    optimizer = DrillingScheduleOptimizer(
        wells=wells,
        constraints=constraints,
        objective=OptimizationObjective.MAXIMIZE_NPV,
        econ_params=econ_params
    )
    
    schedule = optimizer.solve()
    
    # Apply operational delays if parameters provided
    if drilling_params and schedule.is_feasible():
        total_days = len(schedule.wells_drilled) * drilling_params.drill_days_per_well
        total_days += (len(schedule.wells_drilled) - 1) * drilling_params.rig_move_days
        total_days += drilling_params.permit_delay_days
        total_days *= (1 + drilling_params.weather_delay_factor)
        
        schedule.total_drill_days = int(total_days)
        schedule.considers_operational_delays = True
    
    return schedule