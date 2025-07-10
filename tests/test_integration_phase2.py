"""
Integration tests for Phase 2 - Multi-well drilling optimization scenarios.
"""
import pytest
from datetime import date

from src.oil_gas_optimizer.economics import WellEconomics, EconomicParameters
from src.oil_gas_optimizer.drilling_optimizer import (
    DrillingConstraints,
    DrillingScheduleOptimizer,
    OptimizationObjective,
    solve_drilling_schedule
)
from src.oil_gas_optimizer.data_model import DrillingParameters, Lease
from src.oil_gas_optimizer.monte_carlo import MonteCarloParameters, run_monte_carlo_npv


class TestMultiWellScenarios:
    """Test multi-well drilling optimization scenarios."""
    
    def test_three_well_budget_constrained(self):
        """Test optimization with 3-5 wells under budget constraint."""
        # Create Texas wells with varying quality
        wells = [
            WellEconomics("MIDLAND_A_01", 7_500_000, 1100, 0.68, 1.2),  # Best well
            WellEconomics("MIDLAND_A_02", 7_500_000, 1000, 0.70, 1.1),
            WellEconomics("MIDLAND_A_03", 7_500_000, 900, 0.72, 1.0),
            WellEconomics("MIDLAND_A_04", 7_500_000, 850, 0.75, 0.9),
            WellEconomics("MIDLAND_A_05", 7_500_000, 800, 0.78, 0.8)   # Worst well
        ]
        
        # Budget allows only 3 wells
        constraints = DrillingConstraints(
            max_rigs_available=1,
            total_capex_budget=23_000_000,  # ~3 wells
            planning_horizon_months=12
        )
        
        optimizer = DrillingScheduleOptimizer(
            wells=wells,
            constraints=constraints,
            objective=OptimizationObjective.MAXIMIZE_NPV
        )
        
        schedule = optimizer.solve()
        
        # Should select the 3 best wells
        assert len(schedule.wells_drilled) == 3
        assert schedule.total_capex <= 23_000_000
        
        # Best wells should be selected
        selected_names = [w.name for w in schedule.wells_drilled]
        assert "MIDLAND_A_01" in selected_names  # Best IP
        assert "MIDLAND_A_02" in selected_names  # Second best
        assert "MIDLAND_A_05" not in selected_names  # Worst should not be selected
    
    def test_rig_constrained_scheduling(self):
        """Test scheduling with limited rigs over time."""
        # 6 good wells but only 1 rig
        wells = [
            WellEconomics(f"WELL_{i:02d}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(6)
        ]
        
        schedule = solve_drilling_schedule(
            wells=wells,
            max_rigs=1,
            capex_budget=50_000_000,  # Not budget constrained
            horizon_months=4  # Only 4 months - can drill max 4 wells
        )
        
        # Limited by time and rigs
        assert len(schedule.wells_drilled) <= 4
        assert schedule.peak_rigs_used <= 1
        
        # Wells should be spread across months
        month_counts = list(schedule.wells_per_month.values()) if schedule.wells_per_month else []
        assert all(count <= 2 for count in month_counts)  # Max 2 wells per rig per month
    
    def test_multi_lease_field_development(self):
        """Test field development across multiple Texas leases."""
        # Create wells from 5 Texas leases
        texas_leases = [
            ("MIDLAND_A", 28, 1000, 0.70),
            ("MARTIN_B", 15, 900, 0.72),
            ("REEVES_C", 32, 1200, 0.68),  # Best wells
            ("LOVING_D", 22, 1100, 0.69),
            ("HOWARD_E", 12, 800, 0.75)    # Worst wells
        ]
        
        wells = []
        for lease, max_wells, ip, di in texas_leases:
            # Create 5 wells per lease
            for i in range(5):
                wells.append(
                    WellEconomics(
                        name=f"{lease}_{i:02d}",
                        capex=7_500_000 + (1_000_000 if "REEVES" in lease else 0),
                        ip_rate=ip - i * 20,  # Declining quality within lease
                        di=di,
                        b=1.1
                    )
                )
        
        constraints = DrillingConstraints(
            max_rigs_available=2,
            total_capex_budget=60_000_000,  # Can drill ~7-8 wells
            planning_horizon_months=12,
            lease_well_limits={
                "MIDLAND_A": 3,
                "MARTIN_B": 2,
                "REEVES_C": 3,
                "LOVING_D": 2,
                "HOWARD_E": 1
            }
        )
        
        optimizer = DrillingScheduleOptimizer(wells=wells, constraints=constraints)
        schedule = optimizer.solve()
        
        # Count wells per lease
        lease_counts = {}
        for well in schedule.wells_drilled:
            lease = "_".join(well.name.split("_")[:2])
            lease_counts[lease] = lease_counts.get(lease, 0) + 1
        
        # Check lease limits respected
        assert lease_counts.get("MIDLAND_A", 0) <= 3
        assert lease_counts.get("REEVES_C", 0) <= 3
        assert lease_counts.get("HOWARD_E", 0) <= 1
        
        # REEVES_C should be prioritized (best economics)
        assert lease_counts.get("REEVES_C", 0) >= 2
        
        # Should drill 7-8 wells total
        assert 6 <= len(schedule.wells_drilled) <= 8
    
    def test_production_target_optimization(self):
        """Test meeting production targets with minimum wells."""
        # Mix of high and low IP wells
        wells = []
        for i in range(10):
            ip_rate = 1500 if i < 3 else 800  # 3 high IP, 7 low IP
            wells.append(
                WellEconomics(
                    name=f"WELL_{i:02d}",
                    capex=8_000_000 if i < 3 else 6_500_000,
                    ip_rate=ip_rate,
                    di=0.65 if i < 3 else 0.75,
                    b=1.2 if i < 3 else 0.9
                )
            )
        
        # Need 4000 boe/d production
        constraints = DrillingConstraints(
            max_rigs_available=2,
            total_capex_budget=50_000_000,
            min_production_target=4000,  # Forces mix of wells
            planning_horizon_months=12
        )
        
        optimizer = DrillingScheduleOptimizer(wells=wells, constraints=constraints)
        schedule = optimizer.solve()
        
        # Calculate total initial production
        total_ip = sum(w.ip_rate for w in schedule.wells_drilled)
        assert total_ip >= 4000
        
        # Should include at least 2 high-IP wells
        high_ip_count = sum(1 for w in schedule.wells_drilled if w.ip_rate >= 1500)
        assert high_ip_count >= 2
        
        # But also need some lower cost wells
        assert len(schedule.wells_drilled) >= 4
    
    def test_staged_development_strategy(self):
        """Test staged field development with capital recycling."""
        # Create two development phases
        phase1_wells = [
            WellEconomics(f"PHASE1_{i:02d}", 7_000_000, 1100, 0.68, 1.2)
            for i in range(3)
        ]
        
        phase2_wells = [
            WellEconomics(f"PHASE2_{i:02d}", 8_000_000, 900, 0.72, 1.0)
            for i in range(5)
        ]
        
        all_wells = phase1_wells + phase2_wells
        
        # First optimize with limited initial capital
        phase1_schedule = solve_drilling_schedule(
            wells=all_wells,
            max_rigs=1,
            capex_budget=22_000_000,  # Enough for phase 1
            horizon_months=6
        )
        
        # Should prioritize phase 1 (better economics)
        phase1_names = [w.name for w in phase1_schedule.wells_drilled if "PHASE1" in w.name]
        assert len(phase1_names) >= 2
        
        # Calculate cash generation from phase 1
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            royalty=0.1875,
            working_interest=0.80
        )
        
        phase1_metrics = phase1_schedule.calculate_metrics(econ_params)
        assert phase1_metrics['capital_efficiency'] > 2.0  # Good capital efficiency
    
    def test_monte_carlo_multi_well_risk(self):
        """Test Monte Carlo risk assessment for multi-well program."""
        # 5-well drilling program
        wells = [
            WellEconomics(
                name=f"MC_WELL_{i:02d}",
                capex=7_500_000,
                ip_rate=1000 - i * 50,  # Declining quality
                di=0.70 + i * 0.02,     # Increasing decline
                b=1.1 - i * 0.05
            )
            for i in range(5)
        ]
        
        # Run Monte Carlo
        mc_params = MonteCarloParameters(
            n_simulations=200,
            oil_price_volatility=0.30,
            cost_uncertainty=0.15,
            scenarios={
                'low_price': {'oil_price_base': 50.0, 'probability': 0.3},
                'mid_price': {'oil_price_base': 75.0, 'probability': 0.5},
                'high_price': {'oil_price_base': 100.0, 'probability': 0.2}
            }
        )
        
        results = run_monte_carlo_npv(wells, mc_params)
        
        # Multi-well program should have good risk profile
        assert results.probability_positive > 0.85  # High success rate
        assert results.mean_npv > 50_000_000  # Substantial NPV
        
        # Check scenario impacts
        low_npv = results.scenario_results['low_price']['mean_npv']
        high_npv = results.scenario_results['high_price']['mean_npv']
        assert high_npv > low_npv * 2.5  # Significant upside
        
        # Risk metrics
        assert results.var_95 > results.mean_npv * 0.6  # Limited downside
    
    def test_operational_constraints_impact(self):
        """Test impact of operational constraints on scheduling."""
        wells = [
            WellEconomics(f"OP_WELL_{i:02d}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(8)
        ]
        
        # Drilling parameters with delays
        drill_params = DrillingParameters(
            drill_days_per_well=25,
            rig_move_days=5,
            batch_drilling_efficiency=0.85,
            permit_delay_days=30,
            weather_delay_factor=0.10
        )
        
        # Compare with and without operational constraints
        ideal_schedule = solve_drilling_schedule(
            wells=wells,
            max_rigs=2,
            capex_budget=60_000_000,
            horizon_months=6
        )
        
        realistic_schedule = solve_drilling_schedule(
            wells=wells,
            max_rigs=2,
            capex_budget=60_000_000,
            horizon_months=6,
            drilling_params=drill_params
        )
        
        # Both should be feasible
        assert ideal_schedule.is_feasible()
        assert realistic_schedule.is_feasible()
        
        # Realistic schedule should account for delays
        assert realistic_schedule.considers_operational_delays
        assert realistic_schedule.total_drill_days > ideal_schedule.wells_drilled.__len__() * 25