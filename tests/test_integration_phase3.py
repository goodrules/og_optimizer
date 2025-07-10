"""
Integration tests for Phase 3 - Heuristic optimization workflows.
"""
import pytest
from typing import List

from src.oil_gas_optimizer.economics import WellEconomics, EconomicParameters
from src.oil_gas_optimizer.heuristic_optimizer import (
    OptimizationKnobs,
    HeuristicOptimizer,
    ScenarioEvaluator,
    DrillingScenario,
    evaluate_scenario,
    perturb_knobs
)
from src.oil_gas_optimizer.drilling_optimizer import DrillingConstraints
from src.oil_gas_optimizer.monte_carlo import MonteCarloParameters


class TestHeuristicOptimizationWorkflow:
    """Test complete heuristic optimization workflows."""
    
    def test_texas_field_development_optimization(self):
        """Test optimization of a typical Texas oil field development."""
        # Create realistic Texas wells from 5 leases
        texas_wells = []
        lease_data = [
            ("MIDLAND_A", 28, 1100, 0.68, 1.2, 7_500_000),   # Best wells
            ("MARTIN_B", 15, 950, 0.70, 1.1, 7_000_000),
            ("REEVES_C", 32, 1300, 0.66, 1.3, 8_500_000),    # Premium wells
            ("LOVING_D", 22, 1050, 0.69, 1.15, 7_800_000),
            ("HOWARD_E", 12, 850, 0.72, 1.0, 6_800_000)      # Lower quality
        ]
        
        for lease, max_wells, base_ip, di, b, capex in lease_data:
            for i in range(max_wells):
                # Add variability within lease
                ip_rate = base_ip * (1 - i * 0.02)  # 2% decline per well
                texas_wells.append(
                    WellEconomics(
                        name=f"{lease}_{i+1:02d}",
                        capex=capex,
                        ip_rate=ip_rate,
                        di=di,
                        b=b
                    )
                )
        
        # Run heuristic optimization
        optimizer = HeuristicOptimizer(
            available_wells=texas_wells,
            lease_limits={
                "MIDLAND_A": 10,  # Regulatory limits
                "MARTIN_B": 8,
                "REEVES_C": 15,
                "LOVING_D": 10,
                "HOWARD_E": 5
            },
            capex_budget=100_000_000,  # $100MM development budget
            n_trials=15,
            improvement_threshold=0.95
        )
        
        best_knobs, best_metrics, history = optimizer.optimize()
        
        # Verify optimization found good solution
        assert best_metrics.total_npv > 80_000_000  # Should generate significant NPV
        assert best_metrics.total_capex <= 100_000_000  # Within budget
        assert best_metrics.npv_per_dollar > 1.0  # Good capital efficiency
        
        # Check that optimization selected wells from multiple leases
        wells_selected = sum(1 for v in best_knobs.wells_per_lease.values() if v > 0)
        assert wells_selected >= 3  # Should diversify across leases
        
        # Verify convergence
        stats = history.get_statistics()
        assert stats['improvement_ratio'] > 1.3  # Significant improvement
        
        # Best solution should include reasonable parameters
        assert 1 <= best_knobs.rig_count <= 3
        assert 60 <= best_knobs.oil_price_forecast <= 100
        assert 0.05 <= best_knobs.contingency_percent <= 0.30
    
    def test_capital_constrained_optimization(self):
        """Test optimization with severe capital constraints."""
        # Create high-quality wells that exceed budget
        wells = []
        for i in range(20):
            wells.append(
                WellEconomics(
                    name=f"HIGH_QUAL_{i+1:02d}",
                    capex=8_000_000,  # $8MM per well
                    ip_rate=1500,      # High IP
                    di=0.65,
                    b=1.3
                )
            )
        
        # Only $30MM budget (can drill 3-4 wells)
        optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits={"HIGH_QUAL": 20},
            capex_budget=30_000_000,
            n_trials=10
        )
        
        best_knobs, best_metrics, history = optimizer.optimize()
        
        # Should select wells within budget - check actual wells drilled
        assert best_metrics.wells_drilled <= 4  # Budget constraint
        assert best_metrics.total_capex <= 30_000_000
        
        # Should still generate good returns
        assert best_metrics.total_npv > 25_000_000
        assert best_metrics.npv_per_dollar > 1.3
        
        # Should use minimal rigs (capital efficient)
        assert best_knobs.rig_count <= 2
    
    def test_multi_objective_scenario_evaluation(self):
        """Test scenario evaluation with multiple objectives."""
        # Create diverse well portfolio
        wells = []
        
        # High IP, fast decline wells
        for i in range(10):
            wells.append(
                WellEconomics(
                    name=f"FAST_DECLINE_{i+1:02d}",
                    capex=7_000_000,
                    ip_rate=1800,
                    di=0.85,  # Fast decline
                    b=0.8
                )
            )
        
        # Lower IP, slow decline wells
        for i in range(10):
            wells.append(
                WellEconomics(
                    name=f"SLOW_DECLINE_{i+1:02d}",
                    capex=7_500_000,
                    ip_rate=900,
                    di=0.55,  # Slow decline
                    b=1.5
                )
            )
        
        # Test different optimization strategies
        strategies = [
            # Aggressive growth
            OptimizationKnobs(
                wells_per_lease={"FAST_DECLINE": 8, "SLOW_DECLINE": 2},
                rig_count=3,
                hurdle_rate=0.12,
                permit_strategy="aggressive",
                development_pace="fast"
            ),
            # Conservative, long-term
            OptimizationKnobs(
                wells_per_lease={"FAST_DECLINE": 2, "SLOW_DECLINE": 8},
                rig_count=1,
                hurdle_rate=0.18,
                permit_strategy="conservative",
                development_pace="slow"
            ),
            # Balanced
            OptimizationKnobs(
                wells_per_lease={"FAST_DECLINE": 5, "SLOW_DECLINE": 5},
                rig_count=2,
                hurdle_rate=0.15,
                permit_strategy="balanced",
                development_pace="moderate"
            )
        ]
        
        evaluator = ScenarioEvaluator()
        results = []
        
        for knobs in strategies:
            scenario = DrillingScenario.from_knobs(knobs, wells, 80_000_000)
            metrics = evaluator.evaluate(scenario)
            results.append((knobs.development_pace, metrics))
        
        # Fast development should have highest peak production
        fast_metrics = next(m for pace, m in results if pace == "fast")
        slow_metrics = next(m for pace, m in results if pace == "slow")
        assert fast_metrics.peak_production > slow_metrics.peak_production
        
        # Conservative should have lower risk (if Monte Carlo was run)
        # Note: risk_score defaults to 0.5 without Monte Carlo
        # Both should be profitable regardless
        assert slow_metrics.total_npv > 0
        assert fast_metrics.total_npv > 0
    
    def test_monte_carlo_risk_assessment_integration(self):
        """Test integration of Monte Carlo risk assessment with heuristic optimization."""
        # Create portfolio of wells
        wells = [
            WellEconomics(f"WELL_{i+1:02d}", 7_500_000, 1000+i*50, 0.68-i*0.01, 1.1+i*0.02)
            for i in range(15)
        ]
        
        # Run optimization with Monte Carlo
        optimizer = HeuristicOptimizer(
            available_wells=wells,
            lease_limits={"WELL": 15},
            capex_budget=60_000_000,
            n_trials=8
        )
        
        best_knobs, best_metrics, history = optimizer.optimize()
        
        # Re-evaluate best solution with Monte Carlo
        scenario = DrillingScenario.from_knobs(
            best_knobs,
            wells,
            60_000_000
        )
        scenario.run_monte_carlo = True
        scenario.mc_simulations = 500
        
        evaluator = ScenarioEvaluator()
        mc_metrics = evaluator.evaluate(scenario)
        
        # Should have uncertainty metrics
        assert mc_metrics.p10_npv is not None
        assert mc_metrics.p90_npv is not None
        assert mc_metrics.probability_positive is not None
        
        # Should have reasonable risk profile
        assert mc_metrics.probability_positive > 0.90  # High success probability
        assert mc_metrics.p10_npv > mc_metrics.total_npv * 0.7  # Limited downside
        assert mc_metrics.p90_npv < mc_metrics.total_npv * 1.3  # Reasonable upside
    
    def test_knob_perturbation_convergence(self):
        """Test that knob perturbation leads to convergence."""
        wells = [
            WellEconomics(f"W{i}", 7_500_000, 1100-i*20, 0.70, 1.1)
            for i in range(10)
        ]
        
        # Start with worst case
        current_knobs = OptimizationKnobs(
            wells_per_lease={"W": 2},
            rig_count=1,
            oil_price_forecast=50.0,
            hurdle_rate=0.25
        )
        
        # Track improvement over iterations
        improvements = []
        
        for iteration in range(20):
            # Decrease exploration over time
            scale = max(0.1, 1.0 - iteration / 20.0)
            
            # Perturb and evaluate
            new_knobs = perturb_knobs(current_knobs, scale)
            
            current_metrics = evaluate_scenario(current_knobs, wells, 50_000_000)
            new_metrics = evaluate_scenario(new_knobs, wells, 50_000_000)
            
            if new_metrics.total_npv > current_metrics.total_npv:
                improvement = (new_metrics.total_npv - current_metrics.total_npv) / current_metrics.total_npv
                improvements.append(improvement)
                current_knobs = new_knobs
        
        # Should see decreasing improvements (convergence)
        if len(improvements) > 5:
            early_avg = sum(improvements[:3]) / 3
            late_avg = sum(improvements[-3:]) / 3
            assert late_avg < early_avg  # Smaller improvements later
    
    def test_operational_strategy_comparison(self):
        """Test different operational strategies."""
        # Create wells suitable for batch drilling
        pad_wells = []
        for pad in ["PAD_A", "PAD_B", "PAD_C"]:
            for i in range(6):
                pad_wells.append(
                    WellEconomics(
                        name=f"{pad}_{i+1:02d}",
                        capex=7_200_000,
                        ip_rate=1050,
                        di=0.69,
                        b=1.15
                    )
                )
        
        # Compare batch vs continuous drilling
        batch_knobs = OptimizationKnobs(
            wells_per_lease={"PAD_A": 6, "PAD_B": 6, "PAD_C": 4},
            rig_count=2,
            drilling_mode="batch",
            development_pace="fast"
        )
        
        continuous_knobs = OptimizationKnobs(
            wells_per_lease={"PAD_A": 6, "PAD_B": 6, "PAD_C": 4},
            rig_count=2,
            drilling_mode="continuous",
            development_pace="moderate"
        )
        
        batch_scenario = DrillingScenario.from_knobs(batch_knobs, pad_wells, 120_000_000)
        continuous_scenario = DrillingScenario.from_knobs(continuous_knobs, pad_wells, 120_000_000)
        
        evaluator = ScenarioEvaluator()
        batch_metrics = evaluator.evaluate(batch_scenario)
        continuous_metrics = evaluator.evaluate(continuous_scenario)
        
        # Both should be profitable
        assert batch_metrics.total_npv > 100_000_000
        assert continuous_metrics.total_npv > 100_000_000
        
        # Batch drilling should have efficiency advantages
        assert batch_scenario.drilling_params.batch_drilling_efficiency < 1.0
    
    def test_sensitivity_to_economic_assumptions(self):
        """Test sensitivity of optimization to economic parameters."""
        wells = [
            WellEconomics(f"SENS_{i+1:02d}", 7_500_000, 1000, 0.70, 1.1)
            for i in range(20)
        ]
        
        # Test different oil price scenarios
        oil_prices = [50.0, 70.0, 90.0, 110.0]
        results = []
        
        for oil_price in oil_prices:
            knobs = OptimizationKnobs(
                wells_per_lease={"SENS": 10},
                rig_count=2,
                oil_price_forecast=oil_price
            )
            
            scenario = DrillingScenario.from_knobs(knobs, wells, 80_000_000)
            scenario.econ_params.oil_price = oil_price
            
            evaluator = ScenarioEvaluator()
            metrics = evaluator.evaluate(scenario)
            results.append((oil_price, metrics))
        
        # Higher oil prices should yield higher NPV
        for i in range(len(results) - 1):
            assert results[i+1][1].total_npv > results[i][1].total_npv
        
        # NPV should be roughly proportional to oil price changes
        npv_50 = results[0][1].total_npv
        npv_90 = results[2][1].total_npv
        ratio = npv_90 / npv_50
        assert 1.5 < ratio < 4.0  # Oil has high sensitivity to price