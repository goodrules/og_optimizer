"""
Test suite for Monte Carlo simulation of oil & gas uncertainties.
Following TDD approach - RED phase.
"""
import pytest
import numpy as np
from datetime import date

# These imports will fail initially (RED phase)
from src.oil_gas_optimizer.monte_carlo import (
    run_monte_carlo_npv,
    OilPriceModel,
    CostUncertaintyModel,
    OperationalRiskModel,
    MonteCarloParameters,
    MonteCarloResults
)
from src.oil_gas_optimizer.economics import EconomicParameters, WellEconomics


class TestOilPriceModel:
    """Test oil price uncertainty modeling."""
    
    def test_oil_price_model_mean_reversion(self):
        """Oil price should follow mean-reverting process."""
        model = OilPriceModel(
            base_price=80.0,
            volatility=0.25,
            mean_reversion_rate=0.2,
            long_term_mean=75.0
        )
        
        # Generate price paths
        prices = model.generate_price_path(months=60, n_simulations=1000)
        
        assert prices.shape == (1000, 60)
        assert np.all(prices > 0)  # Prices should be positive
        
        # Average should be close to base price initially
        assert 75 < np.mean(prices[:, 0]) < 85
        
        # Long-term average should approach long-term mean
        assert 70 < np.mean(prices[:, -1]) < 80
    
    def test_oil_price_model_volatility(self):
        """Higher volatility should produce wider price ranges."""
        low_vol_model = OilPriceModel(base_price=80.0, volatility=0.10)
        high_vol_model = OilPriceModel(base_price=80.0, volatility=0.40)
        
        low_vol_prices = low_vol_model.generate_price_path(months=12, n_simulations=1000)
        high_vol_prices = high_vol_model.generate_price_path(months=12, n_simulations=1000)
        
        # High volatility should have wider range
        low_vol_range = np.max(low_vol_prices) - np.min(low_vol_prices)
        high_vol_range = np.max(high_vol_prices) - np.min(high_vol_prices)
        
        assert high_vol_range > low_vol_range * 1.5


class TestCostUncertainty:
    """Test cost uncertainty modeling."""
    
    def test_cost_uncertainty_model(self):
        """Drilling costs should vary around base cost."""
        model = CostUncertaintyModel(
            base_cost=7_500_000,
            uncertainty=0.15,
            correlation=0.3  # Some correlation between wells
        )
        
        # Generate costs for 10 wells
        costs = model.generate_well_costs(n_wells=10, n_simulations=1000)
        
        assert costs.shape == (1000, 10)
        assert np.all(costs > 0)
        
        # Average should be close to base cost
        mean_cost = np.mean(costs)
        assert 7_400_000 < mean_cost < 7_600_000
        
        # Should see some correlation between wells
        well_corr = np.corrcoef(costs[:, 0], costs[:, 1])[0, 1]
        assert 0.1 < well_corr < 0.5
    
    def test_cost_escalation(self):
        """Costs should escalate over time."""
        model = CostUncertaintyModel(
            base_cost=7_500_000,
            uncertainty=0.15,
            annual_escalation=0.03  # 3% annual
        )
        
        # Generate costs for wells drilled over 2 years
        drill_months = [0, 6, 12, 18, 24]
        costs = model.generate_well_costs_with_timing(
            drill_months=drill_months,
            n_simulations=100
        )
        
        # Later wells should cost more on average
        # Account for uncertainty - compare medians which are more stable
        # With random noise, require at least 1% inflation over 2 years
        assert np.median(costs[:, -1]) > np.median(costs[:, 0]) * 1.01


class TestOperationalRisk:
    """Test operational risk modeling."""
    
    def test_mechanical_failure_risk(self):
        """Model mechanical failures and weather delays."""
        model = OperationalRiskModel(
            mechanical_failure_rate=0.07,
            weather_delay_days=12,
            permit_delay_range=(0, 45)
        )
        
        # Generate operational delays for 20 wells
        delays = model.generate_delays(n_wells=20, n_simulations=1000)
        
        assert delays.shape == (1000, 20)
        assert np.all(delays >= 0)
        
        # Average delay should match expected values
        mean_delay = np.mean(delays)
        assert 5 < mean_delay < 20  # Days
        
        # Some wells should have significant delays
        max_delays = np.max(delays, axis=0)
        assert np.any(max_delays > 30)


class TestMonteCarloNPV:
    """Test Monte Carlo NPV simulation."""
    
    def test_run_monte_carlo_npv_single_well(self):
        """Run Monte Carlo for single well NPV."""
        well = WellEconomics(
            name="MC_TEST_WELL",
            capex=7_500_000,
            ip_rate=1000,
            di=0.70,
            b=1.1,
            months=360
        )
        
        mc_params = MonteCarloParameters(
            n_simulations=100,
            oil_price_volatility=0.25,
            cost_uncertainty=0.15,
            mechanical_failure_rate=0.07
        )
        
        results = run_monte_carlo_npv([well], mc_params)
        
        assert isinstance(results, MonteCarloResults)
        assert len(results.npv_distribution) == 100
        assert results.p10_npv < results.p50_npv < results.p90_npv
        assert results.probability_positive > 0.5  # Should be profitable most of the time
        assert 15_000_000 < results.mean_npv < 35_000_000
    
    def test_monte_carlo_with_correlation(self):
        """Test correlation between price and costs."""
        wells = [
            WellEconomics(
                name=f"WELL_{i}",
                capex=7_500_000,
                ip_rate=1000,
                di=0.70,
                b=1.1,
                months=360
            )
            for i in range(5)
        ]
        
        mc_params = MonteCarloParameters(
            n_simulations=200,
            oil_price_volatility=0.30,
            cost_uncertainty=0.20,
            price_cost_correlation=0.4  # Positive correlation
        )
        
        results = run_monte_carlo_npv(wells, mc_params)
        
        # Should see wider distribution due to correlation
        npv_range = results.p90_npv - results.p10_npv
        assert npv_range > 10_000_000  # Wide range for 5 wells
        # Verify substantial variation
        assert results.std_npv > 3_000_000
    
    def test_monte_carlo_value_at_risk(self):
        """Calculate Value at Risk metrics."""
        well = WellEconomics(
            name="VAR_TEST",
            capex=7_500_000,
            ip_rate=800,  # Lower IP
            di=0.75,  # Higher decline
            b=1.0,
            months=240
        )
        
        mc_params = MonteCarloParameters(
            n_simulations=500,
            oil_price_volatility=0.35,
            cost_uncertainty=0.20
        )
        
        results = run_monte_carlo_npv([well], mc_params)
        
        assert hasattr(results, 'var_95')  # Value at Risk at 95% confidence
        assert hasattr(results, 'cvar_95')  # Conditional Value at Risk
        assert results.var_95 < results.mean_npv
        assert results.cvar_95 < results.var_95  # CVaR is more conservative


class TestMonteCarloResults:
    """Test Monte Carlo results and statistics."""
    
    def test_monte_carlo_results_statistics(self):
        """Results should include comprehensive statistics."""
        well = WellEconomics(
            name="STATS_TEST",
            capex=7_500_000,
            ip_rate=1000,
            di=0.70,
            b=1.1
        )
        
        mc_params = MonteCarloParameters(n_simulations=1000)
        results = run_monte_carlo_npv([well], mc_params)
        
        # Check all statistics are present
        assert hasattr(results, 'mean_npv')
        assert hasattr(results, 'std_npv')
        assert hasattr(results, 'p10_npv')
        assert hasattr(results, 'p50_npv')
        assert hasattr(results, 'p90_npv')
        assert hasattr(results, 'probability_positive')
        assert hasattr(results, 'probability_exceeds_hurdle')
        
        # Statistics should be consistent
        assert results.p10_npv <= results.p50_npv <= results.p90_npv
        assert 0 <= results.probability_positive <= 1
    
    def test_monte_carlo_results_scenarios(self):
        """Results should include scenario analysis."""
        wells = [WellEconomics(f"W{i}", 7_500_000, 1000, 0.70, 1.1) for i in range(3)]
        
        mc_params = MonteCarloParameters(
            n_simulations=200,
            scenarios={
                'low_price': {'oil_price_base': 50.0, 'probability': 0.2},
                'base_case': {'oil_price_base': 80.0, 'probability': 0.6},
                'high_price': {'oil_price_base': 100.0, 'probability': 0.2}
            }
        )
        
        results = run_monte_carlo_npv(wells, mc_params)
        
        assert hasattr(results, 'scenario_results')
        assert 'low_price' in results.scenario_results
        assert 'base_case' in results.scenario_results
        assert 'high_price' in results.scenario_results
        
        # High price scenario should have higher NPV
        assert (results.scenario_results['high_price']['mean_npv'] > 
                results.scenario_results['low_price']['mean_npv'])


class TestMonteCarloParameters:
    """Test Monte Carlo parameter validation."""
    
    def test_monte_carlo_parameters_validation(self):
        """Parameters should validate inputs."""
        params = MonteCarloParameters(
            n_simulations=1000,
            oil_price_volatility=0.25,
            cost_uncertainty=0.15
        )
        
        assert params.n_simulations == 1000
        assert params.oil_price_volatility == 0.25
    
    def test_monte_carlo_parameters_invalid_simulations(self):
        """Should raise error for invalid simulation count."""
        with pytest.raises(ValueError, match="simulations must be positive"):
            MonteCarloParameters(n_simulations=-10)
    
    def test_monte_carlo_parameters_invalid_volatility(self):
        """Should raise error for invalid volatility."""
        with pytest.raises(ValueError, match="Volatility must be between 0 and 1"):
            MonteCarloParameters(oil_price_volatility=1.5)