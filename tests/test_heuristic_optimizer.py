"""
Test suite for heuristic optimization of oil & gas field development.
Following TDD approach - RED phase.
"""
import pytest
from typing import Dict, List

# These imports will fail initially (RED phase)
from src.oil_gas_optimizer.heuristic_optimizer import (
    OptimizationKnobs,
    DrillingScenario,
    ScenarioEvaluator,
    HeuristicOptimizer,
    perturb_knobs,
    evaluate_scenario,
    worst_case_knobs,
    TrialHistory,
    OptimizationMetrics
)
from src.oil_gas_optimizer.economics import WellEconomics, EconomicParameters
from src.oil_gas_optimizer.drilling_optimizer import DrillingConstraints


class TestOptimizationKnobs:
    """Test optimization knobs for oil & gas parameters."""
    
    def test_optimization_knobs_creation(self):
        """Create knobs for field development optimization."""
        knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 5, "REEVES_C": 8},
            rig_count=2,
            drilling_mode="batch",  # batch or continuous
            contingency_percent=0.15,
            hurdle_rate=0.15,
            oil_price_forecast=80.0,
            price_volatility=0.25,
            permit_strategy="aggressive",  # aggressive, conservative, balanced
            development_pace="moderate"  # slow, moderate, fast
        )
        
        assert knobs.wells_per_lease["MIDLAND_A"] == 5
        assert knobs.rig_count == 2
        assert knobs.drilling_mode == "batch"
        assert knobs.contingency_percent == 0.15
    
    def test_knobs_validation(self):
        """Validate knob ranges."""
        with pytest.raises(ValueError, match="Rig count must be positive"):
            OptimizationKnobs(wells_per_lease={}, rig_count=0)
        
        with pytest.raises(ValueError, match="Contingency must be between"):
            OptimizationKnobs(wells_per_lease={}, rig_count=1, contingency_percent=2.0)
    
    def test_knobs_to_dict(self):
        """Convert knobs to dictionary for storage."""
        knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 5},
            rig_count=2,
            oil_price_forecast=75.0
        )
        
        knob_dict = knobs.to_dict()
        assert isinstance(knob_dict, dict)
        assert knob_dict["rig_count"] == 2
        assert knob_dict["oil_price_forecast"] == 75.0


class TestPerturbKnobs:
    """Test knob perturbation for exploration."""
    
    def test_perturb_knobs_small_changes(self):
        """Test small perturbations around current best."""
        base_knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 10, "REEVES_C": 15},
            rig_count=2,
            contingency_percent=0.15,
            oil_price_forecast=80.0,
            price_volatility=0.25
        )
        
        # Small perturbation
        perturbed = perturb_knobs(base_knobs, scale=0.1)
        
        assert perturbed != base_knobs  # Should be different
        # Wells should change by small amount
        assert abs(perturbed.wells_per_lease.get("MIDLAND_A", 10) - 10) <= 3
        # Rig count might change by 1
        assert abs(perturbed.rig_count - 2) <= 1
        # Price should be close
        assert abs(perturbed.oil_price_forecast - 80.0) <= 10
    
    def test_perturb_knobs_large_exploration(self):
        """Test large perturbations for exploration."""
        base_knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 10},
            rig_count=2,
            drilling_mode="continuous",
            permit_strategy="conservative"
        )
        
        # Large perturbation
        perturbed = perturb_knobs(base_knobs, scale=1.0)
        
        # Should see bigger changes
        assert abs(perturbed.wells_per_lease.get("MIDLAND_A", 10) - 10) <= 10
        # Mode might flip
        possible_modes = ["continuous", "batch"]
        assert perturbed.drilling_mode in possible_modes
    
    def test_perturb_respects_constraints(self):
        """Perturbations should respect valid ranges."""
        base_knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 30},  # Near max
            rig_count=5,  # Near max
            contingency_percent=0.25,
            hurdle_rate=0.20
        )
        
        # Perturb many times
        for _ in range(20):
            perturbed = perturb_knobs(base_knobs, scale=0.5)
            # Check constraints
            assert all(0 <= wells <= 32 for wells in perturbed.wells_per_lease.values())
            assert 1 <= perturbed.rig_count <= 5
            assert 0 <= perturbed.contingency_percent <= 0.30
            assert 0.10 <= perturbed.hurdle_rate <= 0.30


class TestDrillingScenario:
    """Test drilling scenario creation from knobs."""
    
    def test_create_drilling_scenario(self):
        """Create a drilling scenario from optimization knobs."""
        knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 5, "REEVES_C": 3},
            rig_count=2,
            drilling_mode="batch",
            oil_price_forecast=80.0
        )
        
        # Create wells for the leases
        all_wells = []
        for lease in ["MIDLAND_A", "REEVES_C"]:
            for i in range(10):  # 10 potential wells per lease
                all_wells.append(
                    WellEconomics(
                        name=f"{lease}_{i:02d}",
                        capex=7_500_000,
                        ip_rate=1000,
                        di=0.70,
                        b=1.1
                    )
                )
        
        scenario = DrillingScenario.from_knobs(knobs, all_wells)
        
        assert len(scenario.selected_wells) == 8  # 5 + 3
        assert scenario.constraints.max_rigs_available == 2
        assert scenario.econ_params.oil_price == 80.0
        
        # Check well selection
        midland_wells = [w for w in scenario.selected_wells if "MIDLAND_A" in w.name]
        reeves_wells = [w for w in scenario.selected_wells if "REEVES_C" in w.name]
        assert len(midland_wells) == 5
        assert len(reeves_wells) == 3
    
    def test_scenario_with_constraints(self):
        """Test scenario creation with operational constraints."""
        knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 10},
            rig_count=1,
            contingency_percent=0.20,
            permit_strategy="conservative"
        )
        
        wells = [WellEconomics(f"MIDLAND_A_{i:02d}", 7_500_000, 1000, 0.70, 1.1) 
                for i in range(20)]
        
        scenario = DrillingScenario.from_knobs(
            knobs, 
            wells,
            base_capex_budget=100_000_000
        )
        
        # Budget should include contingency
        expected_budget = 100_000_000 * (1 - 0.20)  # 80M after contingency
        assert scenario.constraints.total_capex_budget == expected_budget
        
        # Conservative permitting should add delays
        assert scenario.drilling_params.permit_delay_days > 30


class TestScenarioEvaluator:
    """Test scenario evaluation."""
    
    def test_evaluate_scenario(self):
        """Evaluate a drilling scenario."""
        # Create simple scenario
        wells = [
            WellEconomics(f"WELL_{i}", 7_500_000, 1000-i*50, 0.70, 1.1)
            for i in range(5)
        ]
        
        scenario = DrillingScenario(
            selected_wells=wells,
            constraints=DrillingConstraints(
                max_rigs_available=2,
                total_capex_budget=40_000_000
            ),
            econ_params=EconomicParameters(
                oil_price=80.0,
                discount_rate=0.10
            )
        )
        
        evaluator = ScenarioEvaluator()
        metrics = evaluator.evaluate(scenario)
        
        assert isinstance(metrics, OptimizationMetrics)
        assert metrics.total_npv > 0
        assert metrics.total_capex > 0
        assert metrics.npv_per_dollar > 0
        assert metrics.peak_production > 0
        assert 0 <= metrics.risk_score <= 1
    
    def test_evaluate_with_monte_carlo(self):
        """Evaluate scenario with uncertainty analysis."""
        wells = [WellEconomics(f"W{i}", 7_500_000, 1000, 0.70, 1.1) for i in range(3)]
        
        scenario = DrillingScenario(
            selected_wells=wells,
            constraints=DrillingConstraints(max_rigs_available=1, total_capex_budget=25_000_000),
            econ_params=EconomicParameters(oil_price=80.0, discount_rate=0.10),
            run_monte_carlo=True,
            mc_simulations=100
        )
        
        evaluator = ScenarioEvaluator()
        metrics = evaluator.evaluate(scenario)
        
        # Should have uncertainty metrics
        assert hasattr(metrics, 'p10_npv')
        assert hasattr(metrics, 'p90_npv')
        assert hasattr(metrics, 'probability_positive')
        assert metrics.p10_npv < metrics.total_npv < metrics.p90_npv


class TestHeuristicOptimizer:
    """Test heuristic optimization process."""
    
    def test_heuristic_optimizer_initialization(self):
        """Initialize heuristic optimizer."""
        wells = [WellEconomics(f"W{i}", 7_500_000, 1000, 0.70, 1.1) for i in range(10)]
        
        optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits={"DEFAULT": 10},
            capex_budget=50_000_000,
            n_trials=20,
            improvement_threshold=0.95
        )
        
        assert optimizer.n_trials == 20
        assert optimizer.best_knobs is None
        assert optimizer.best_score == float('-inf')
    
    def test_run_optimization(self):
        """Run heuristic optimization."""
        # Create wells from multiple leases
        wells = []
        for lease in ["MIDLAND_A", "REEVES_C"]:
            for i in range(5):
                wells.append(
                    WellEconomics(
                        name=f"{lease}_{i:02d}",
                        capex=7_500_000 if lease == "MIDLAND_A" else 8_500_000,
                        ip_rate=1000 if lease == "MIDLAND_A" else 1200,
                        di=0.70,
                        b=1.1
                    )
                )
        
        optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits={"MIDLAND_A": 3, "REEVES_C": 2},
            capex_budget=40_000_000,
            n_trials=10
        )
        
        best_knobs, best_metrics, history = optimizer.optimize()
        
        assert best_knobs is not None
        assert best_metrics.total_npv > 0
        assert len(history.trials) == 10
        
        # Should have found reasonable solution
        total_wells = sum(best_knobs.wells_per_lease.values())
        assert 3 <= total_wells <= 5  # Budget constrained
        
        # Best trial should be tracked
        assert history.best_trial_idx >= 0
        assert history.trials[history.best_trial_idx]["metrics"].total_npv == best_metrics.total_npv
    
    def test_convergence_behavior(self):
        """Test that optimization improves over trials."""
        wells = [WellEconomics(f"W{i}", 7_500_000, 900+i*20, 0.70, 1.1) for i in range(8)]
        
        optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits={"DEFAULT": 8},
            capex_budget=35_000_000,
            n_trials=15
        )
        
        _, _, history = optimizer.optimize()
        
        # Should see improvement
        first_half_best = max(t["metrics"].total_npv for t in history.trials[:7])
        second_half_best = max(t["metrics"].total_npv for t in history.trials[7:])
        
        # Second half should be better (or at least not worse)
        assert second_half_best >= first_half_best * 0.95


class TestWorstCaseKnobs:
    """Test worst-case scenario generation."""
    
    def test_worst_case_knobs(self):
        """Generate worst-case knobs."""
        worst = worst_case_knobs()
        
        assert worst.rig_count == 1  # Minimum rigs
        assert worst.contingency_percent >= 0.25  # High contingency
        assert worst.oil_price_forecast <= 50  # Low oil price
        assert worst.price_volatility >= 0.35  # High volatility
        assert worst.permit_strategy == "conservative"  # Slow permitting
        assert worst.development_pace == "slow"  # Slow development
        
        # Should have minimal wells
        total_wells = sum(worst.wells_per_lease.values())
        assert total_wells <= 5


class TestTrialHistory:
    """Test trial history tracking."""
    
    def test_trial_history_tracking(self):
        """Track optimization trials."""
        history = TrialHistory()
        
        # Add trials
        knobs1 = OptimizationKnobs(wells_per_lease={"A": 5}, rig_count=2)
        metrics1 = OptimizationMetrics(total_npv=50_000_000, total_capex=30_000_000)
        history.add_trial(knobs1, metrics1)
        
        knobs2 = OptimizationKnobs(wells_per_lease={"A": 6}, rig_count=2)
        metrics2 = OptimizationMetrics(total_npv=55_000_000, total_capex=35_000_000)
        history.add_trial(knobs2, metrics2)
        
        assert len(history.trials) == 2
        assert history.best_trial_idx == 1  # Second trial is better
        assert history.get_best_trial()["metrics"].total_npv == 55_000_000
    
    def test_history_statistics(self):
        """Calculate history statistics."""
        history = TrialHistory()
        
        for i in range(5):
            knobs = OptimizationKnobs(wells_per_lease={"A": i+3}, rig_count=2)
            metrics = OptimizationMetrics(
                total_npv=40_000_000 + i*5_000_000,
                total_capex=25_000_000 + i*3_000_000
            )
            history.add_trial(knobs, metrics)
        
        stats = history.get_statistics()
        assert stats["mean_npv"] == 50_000_000
        assert stats["best_npv"] == 60_000_000
        assert stats["improvement_ratio"] > 1.0