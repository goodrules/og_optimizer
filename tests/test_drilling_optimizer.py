"""
Test suite for drilling sequence optimization using OR-Tools.
Following TDD approach - RED phase.
"""
import pytest
from datetime import date, timedelta
from ortools.sat.python import cp_model

# These imports will fail initially (RED phase)
from src.oil_gas_optimizer.drilling_optimizer import (
    DrillingScheduleOptimizer,
    DrillingConstraints,
    DrillingDecisionVariables,
    DrillingSchedule,
    RigAllocation,
    OptimizationObjective,
    solve_drilling_schedule
)
from src.oil_gas_optimizer.economics import WellEconomics, EconomicParameters
from src.oil_gas_optimizer.data_model import Lease, DrillingParameters


class TestDrillingConstraints:
    """Test drilling constraint definitions."""
    
    def test_drilling_constraints_validation(self):
        """Validate drilling constraint parameters."""
        constraints = DrillingConstraints(
            max_rigs_available=3,
            max_wells_per_rig_per_month=2,
            total_capex_budget=50_000_000,
            min_production_target=5000,  # boe/d
            planning_horizon_months=24
        )
        
        assert constraints.max_rigs_available == 3
        assert constraints.max_wells_per_rig_per_month == 2
        assert constraints.total_capex_budget == 50_000_000
    
    def test_drilling_constraints_invalid_rigs(self):
        """Should raise error for invalid rig count."""
        with pytest.raises(ValueError, match="rigs must be positive"):
            DrillingConstraints(max_rigs_available=0, total_capex_budget=1000000)
    
    def test_drilling_constraints_with_lease_limits(self):
        """Test constraints with per-lease well limits."""
        constraints = DrillingConstraints(
            max_rigs_available=2,
            total_capex_budget=100_000_000,
            lease_well_limits={
                "MIDLAND_A": 20,
                "REEVES_C": 32,
                "HOWARD_E": 12
            }
        )
        
        assert constraints.lease_well_limits["MIDLAND_A"] == 20
        assert constraints.lease_well_limits["REEVES_C"] == 32


class TestDrillingDecisionVariables:
    """Test OR-Tools decision variable creation."""
    
    def test_create_drilling_decision_variables(self):
        """Create decision variables for drilling optimization."""
        model = cp_model.CpModel()
        
        leases = [
            Lease("MIDLAND_A", available_wells=20),
            Lease("REEVES_C", available_wells=30)
        ]
        
        variables = DrillingDecisionVariables.create_for_leases(
            model=model,
            leases=leases,
            horizon_months=12
        )
        
        assert hasattr(variables, 'wells_drilled')
        assert hasattr(variables, 'drilling_start')
        assert hasattr(variables, 'rig_assignments')
        
        # Check dimensions
        assert len(variables.wells_drilled) == 2  # 2 leases
        assert len(variables.wells_drilled["MIDLAND_A"]) == 12  # 12 months
    
    def test_decision_variables_with_rig_tracking(self):
        """Track rig utilization across time."""
        model = cp_model.CpModel()
        leases = [Lease(f"LEASE_{i}", 20) for i in range(3)]
        
        variables = DrillingDecisionVariables.create_for_leases(
            model=model,
            leases=leases,
            horizon_months=6,
            max_rigs=2
        )
        
        # Should have rig assignment variables
        assert hasattr(variables, 'rig_active')
        assert len(variables.rig_active) == 2  # 2 rigs
        assert len(variables.rig_active[0]) == 6  # 6 months


class TestDrillingScheduleOptimizer:
    """Test drilling schedule optimization."""
    
    def test_simple_drilling_optimization(self):
        """Optimize drilling for small set of wells."""
        # Create test wells
        wells = [
            WellEconomics(
                name="MIDLAND_A_01",
                capex=7_500_000,
                ip_rate=1000,
                di=0.70,
                b=1.1
            ),
            WellEconomics(
                name="MIDLAND_A_02",
                capex=7_500_000,
                ip_rate=950,
                di=0.72,
                b=1.0
            ),
            WellEconomics(
                name="REEVES_C_01",
                capex=8_500_000,
                ip_rate=1200,
                di=0.68,
                b=1.2
            )
        ]
        
        # Create optimizer
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=1,
                total_capex_budget=25_000_000,  # Can drill all 3
                planning_horizon_months=6
            ),
            objective=OptimizationObjective.MAXIMIZE_NPV
        )
        
        # Solve
        schedule = optimizer.solve()
        
        assert isinstance(schedule, DrillingSchedule)
        assert len(schedule.wells_drilled) == 3
        assert schedule.total_capex <= 25_000_000
        assert schedule.peak_rigs_used <= 1
    
    def test_optimization_with_budget_constraint(self):
        """Test optimization with tight budget constraint."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000-i*50, 0.70, 1.1)
            for i in range(5)
        ]
        
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=2,
                total_capex_budget=20_000_000,  # Can only drill 2-3 wells
                planning_horizon_months=12
            )
        )
        
        schedule = optimizer.solve()
        
        # Should select best wells within budget
        assert 2 <= len(schedule.wells_drilled) <= 3
        assert schedule.total_capex <= 20_000_000
        # Best wells (highest IP) should be selected
        assert "WELL_0" in [w.name for w in schedule.wells_drilled]
    
    def test_optimization_with_rig_constraints(self):
        """Test optimization with limited rigs."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(10)
        ]
        
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=1,
                max_wells_per_rig_per_month=1,
                total_capex_budget=100_000_000,  # No budget constraint
                planning_horizon_months=6  # Can only drill 6 wells
            )
        )
        
        schedule = optimizer.solve()
        
        assert len(schedule.wells_drilled) <= 6  # Limited by time and rigs
        assert all(month <= 1 for month in schedule.wells_per_month.values())
    
    def test_optimization_with_production_target(self):
        """Test optimization with minimum production target."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 800+i*100, 0.70, 1.1)
            for i in range(6)
        ]
        
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=2,
                total_capex_budget=50_000_000,
                min_production_target=3000,  # Need multiple wells
                planning_horizon_months=12
            ),
            econ_params=EconomicParameters(oil_price=80.0, discount_rate=0.10)
        )
        
        schedule = optimizer.solve()
        
        # Should drill enough wells to meet production target
        total_ip = sum(w.ip_rate for w in schedule.wells_drilled)
        assert total_ip >= 3000
    
    def test_multi_lease_optimization(self):
        """Test optimization across multiple leases."""
        # Wells from different leases
        wells = []
        for lease in ["MIDLAND_A", "REEVES_C", "HOWARD_E"]:
            for i in range(5):
                wells.append(
                    WellEconomics(
                        name=f"{lease}_{i:02d}",
                        capex=7_500_000 + (1_000_000 if lease == "REEVES_C" else 0),
                        ip_rate=1000 + (200 if lease == "REEVES_C" else -100 if lease == "HOWARD_E" else 0),
                        di=0.70,
                        b=1.1
                    )
                )
        
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=2,
                total_capex_budget=40_000_000,
                lease_well_limits={
                    "MIDLAND_A": 3,
                    "REEVES_C": 2,
                    "HOWARD_E": 2
                },
                planning_horizon_months=12
            )
        )
        
        schedule = optimizer.solve()
        
        # Check lease limits are respected
        lease_counts = {}
        for well in schedule.wells_drilled:
            # Extract lease name (handle multi-part names like REEVES_C)
            parts = well.name.split("_")
            if len(parts) >= 3 and parts[1] in ["A", "C", "E"]:
                lease = "_".join(parts[:2])  # e.g., "REEVES_C"
            else:
                lease = parts[0]
            lease_counts[lease] = lease_counts.get(lease, 0) + 1
        
        assert lease_counts.get("MIDLAND_A", 0) <= 3
        assert lease_counts.get("REEVES_C", 0) <= 2
        assert lease_counts.get("HOWARD_E", 0) <= 2
        
        # REEVES_C wells should be prioritized (highest NPV)
        assert lease_counts.get("REEVES_C", 0) >= 1


class TestRigAllocation:
    """Test rig allocation and scheduling."""
    
    def test_rig_allocation_continuous_drilling(self):
        """Test continuous drilling with rig moves."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(6)
        ]
        
        allocation = RigAllocation(
            n_rigs=1,
            wells=wells,
            drill_days_per_well=20,
            move_days_between_wells=5
        )
        
        schedule = allocation.create_continuous_schedule()
        
        # Check continuous utilization
        assert len(schedule.wells_drilled) == 6
        assert schedule.total_drill_days is not None
        # Each well takes 20 days + 5 days move (except last well)
        expected_days = 20 * 6 + 5 * 5  # 145 days
        assert schedule.total_drill_days == expected_days
    
    def test_rig_allocation_batch_drilling(self):
        """Test batch/pad drilling optimization."""
        # Wells from same pad
        pad_wells = [
            WellEconomics(f"PAD_A_{i:02d}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(4)
        ]
        
        allocation = RigAllocation(
            n_rigs=1,
            wells=pad_wells,
            drill_days_per_well=20,
            move_days_between_pads=10,
            move_days_within_pad=2,
            enable_batch_drilling=True
        )
        
        schedule = allocation.create_batch_schedule()
        
        # Should drill all pad wells consecutively
        assert schedule.batch_efficiency_gain > 0
        # Total time should be less than continuous drilling
        assert schedule.total_drill_days < 20 * 4 + 10 * 3


class TestDrillingSchedule:
    """Test drilling schedule output and metrics."""
    
    def test_drilling_schedule_metrics(self):
        """Test schedule metrics calculation."""
        wells = [
            WellEconomics("W1", 7_500_000, 1000, 0.70, 1.1),
            WellEconomics("W2", 8_000_000, 1100, 0.68, 1.2),
            WellEconomics("W3", 7_000_000, 900, 0.72, 1.0)
        ]
        
        schedule = DrillingSchedule(
            wells_drilled=wells,
            drilling_order=["W1", "W2", "W3"],
            start_dates={
                "W1": date(2024, 1, 1),
                "W2": date(2024, 2, 1),
                "W3": date(2024, 3, 1)
            },
            total_capex=22_500_000,
            peak_rigs_used=1
        )
        
        metrics = schedule.calculate_metrics(
            econ_params=EconomicParameters(oil_price=80.0, discount_rate=0.10)
        )
        
        assert 'total_npv' in metrics
        assert 'peak_production_boed' in metrics
        assert 'wells_drilled_count' in metrics
        assert 'capital_efficiency' in metrics
        
        assert metrics['wells_drilled_count'] == 3
        assert metrics['total_capex'] == 22_500_000
        assert metrics['peak_production_boed'] > 2500  # Multiple wells producing


class TestSolveDrillingSchedule:
    """Test high-level solving function."""
    
    def test_solve_drilling_schedule_simple(self):
        """Test simple drilling schedule solving."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000-i*50, 0.70, 1.1)
            for i in range(5)
        ]
        
        schedule = solve_drilling_schedule(
            wells=wells,
            max_rigs=1,
            capex_budget=25_000_000,
            horizon_months=12
        )
        
        assert isinstance(schedule, DrillingSchedule)
        assert len(schedule.wells_drilled) <= 3  # Budget constraint
        assert schedule.is_feasible()
    
    def test_solve_with_drilling_parameters(self):
        """Test solving with detailed drilling parameters."""
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(8)
        ]
        
        drill_params = DrillingParameters(
            drill_days_per_well=25,
            rig_move_days=5,
            batch_drilling_efficiency=0.85,
            permit_delay_days=30,
            weather_delay_factor=0.1
        )
        
        schedule = solve_drilling_schedule(
            wells=wells,
            max_rigs=2,
            capex_budget=50_000_000,
            horizon_months=12,
            drilling_params=drill_params
        )
        
        # Should account for delays and efficiency
        assert schedule.total_drill_days > len(schedule.wells_drilled) * 25
        assert schedule.considers_operational_delays