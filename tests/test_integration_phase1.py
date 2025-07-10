"""
Integration tests for Phase 1 - Single well scenarios.
"""
import pytest
from datetime import date

from src.oil_gas_optimizer.decline_curves import DeclineCurveParameters, calculate_cumulative_production
from src.oil_gas_optimizer.economics import (
    EconomicParameters, 
    WellEconomics, 
    calculate_well_npv,
    calculate_irr
)
from src.oil_gas_optimizer.monte_carlo import (
    MonteCarloParameters,
    run_monte_carlo_npv
)


class TestSingleWellScenarios:
    """Test complete single well scenarios integrating all Phase 1 components."""
    
    def test_texas_horizontal_well_base_case(self):
        """Test typical Texas horizontal oil well economics."""
        # Create a typical Permian well
        well = WellEconomics(
            name="MIDLAND_A_01",
            capex=7_500_000,  # $7.5M horizontal well
            ip_rate=1000,  # 1000 boe/d
            di=0.70,  # 70% annual decline (typical Permian)
            b=1.1,  # Hyperbolic b factor
            months=360,  # 30 year life
            eur_mboe=400  # 400,000 boe EUR
        )
        
        # Texas economic parameters
        econ_params = EconomicParameters(
            oil_price=80.0,  # $80/bbl WTI
            discount_rate=0.15,  # 15% hurdle rate for shale
            opex_per_boe=12.0,  # Texas operating costs
            royalty=0.1875,  # Standard Texas royalty
            working_interest=0.80,  # 80% WI
            severance_tax=0.046,  # Texas severance tax
            ad_valorem_tax=0.02  # Property tax
        )
        
        # Calculate deterministic NPV
        npv = calculate_well_npv(well, econ_params)
        cash_flows = well.get_cash_flows(econ_params)
        irr = calculate_irr(cash_flows)
        
        # Verify economics are reasonable
        assert npv > 15_000_000  # Should be profitable
        assert irr > 0.50  # High IRR for good wells
        
        # Calculate EUR
        decline_params = well.get_decline_params()
        eur = calculate_cumulative_production(decline_params, well.months)
        assert 350_000 < eur < 450_000  # Should be close to specified EUR
    
    def test_marginal_well_low_oil_price(self):
        """Test marginal well economics at low oil price."""
        # Lower quality well
        well = WellEconomics(
            name="MARGINAL_01",
            capex=6_000_000,  # Cheaper vertical/short lateral
            ip_rate=400,  # Lower IP
            di=0.85,  # Steeper decline
            b=0.8,
            months=240  # 20 year life
        )
        
        # Low oil price scenario
        econ_params = EconomicParameters(
            oil_price=45.0,  # Low oil price
            discount_rate=0.15,
            opex_per_boe=15.0,  # Higher opex for marginal well
            royalty=0.1875,
            working_interest=0.75
        )
        
        npv = calculate_well_npv(well, econ_params)
        
        # Should be unprofitable or marginally profitable
        assert npv < 1_000_000
    
    def test_high_quality_delaware_well(self):
        """Test high-quality Delaware basin well."""
        # Premium Delaware well
        well = WellEconomics(
            name="REEVES_C_01",
            capex=9_000_000,  # More expensive but better well
            ip_rate=1500,  # High IP
            di=0.65,  # Moderate decline
            b=1.2,
            months=360,
            eur_mboe=600  # High EUR
        )
        
        econ_params = EconomicParameters(
            oil_price=85.0,
            discount_rate=0.12,  # Lower discount rate for proven area
            opex_per_boe=10.0,  # Efficient operations
            royalty=0.20,  # Slightly higher royalty
            working_interest=0.85
        )
        
        npv = calculate_well_npv(well, econ_params)
        irr = calculate_irr(well.get_cash_flows(econ_params))
        
        assert npv > 30_000_000  # Very profitable
        assert irr > 1.0  # >100% IRR
    
    def test_monte_carlo_single_well_risk(self):
        """Test Monte Carlo simulation for single well risk assessment."""
        # Standard well
        well = WellEconomics(
            name="MC_TEST_01",
            capex=7_500_000,
            ip_rate=800,
            di=0.72,
            b=1.0,
            months=300
        )
        
        # Run Monte Carlo with typical uncertainties
        mc_params = MonteCarloParameters(
            n_simulations=500,
            oil_price_volatility=0.30,  # 30% price volatility
            cost_uncertainty=0.15,  # 15% cost uncertainty
            mechanical_failure_rate=0.07  # 7% failure rate
        )
        
        results = run_monte_carlo_npv([well], mc_params)
        
        # Check risk metrics
        assert results.probability_positive > 0.70  # Should be profitable >70% of time
        assert results.p10_npv < results.p50_npv < results.p90_npv
        assert results.std_npv > 1_000_000  # Significant uncertainty
        
        # Value at Risk should show downside
        assert results.var_95 < results.mean_npv * 0.95  # 5% downside
    
    def test_scenario_analysis_single_well(self):
        """Test scenario analysis for strategic planning."""
        well = WellEconomics(
            name="SCENARIO_01",
            capex=7_000_000,
            ip_rate=900,
            di=0.70,
            b=1.1,
            months=360
        )
        
        # Define price scenarios
        mc_params = MonteCarloParameters(
            n_simulations=200,
            oil_price_volatility=0.20,
            scenarios={
                'bear_case': {'oil_price_base': 50.0, 'probability': 0.25},
                'base_case': {'oil_price_base': 75.0, 'probability': 0.50},
                'bull_case': {'oil_price_base': 100.0, 'probability': 0.25}
            }
        )
        
        results = run_monte_carlo_npv([well], mc_params)
        
        # Verify scenario results
        assert results.scenario_results is not None
        bear_npv = results.scenario_results['bear_case']['mean_npv']
        base_npv = results.scenario_results['base_case']['mean_npv']
        bull_npv = results.scenario_results['bull_case']['mean_npv']
        
        assert bear_npv < base_npv < bull_npv
        assert bull_npv > bear_npv * 2  # Significant upside in bull case
    
    def test_texas_lease_well_comparison(self):
        """Compare wells from different Texas counties."""
        # Define wells from different areas
        wells = [
            WellEconomics("MIDLAND_01", 7_500_000, 1000, 0.70, 1.1),
            WellEconomics("MARTIN_01", 7_000_000, 900, 0.72, 1.0),
            WellEconomics("REEVES_01", 8_500_000, 1200, 0.68, 1.2),
            WellEconomics("LOVING_01", 8_000_000, 1100, 0.69, 1.15),
            WellEconomics("HOWARD_01", 6_500_000, 800, 0.75, 0.9)
        ]
        
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.15,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        # Calculate NPV for each well
        npvs = {}
        for well in wells:
            npvs[well.name] = calculate_well_npv(well, econ_params)
        
        # Reeves should be best (highest IP, lowest decline)
        assert npvs["REEVES_01"] == max(npvs.values())
        # Howard should be worst (lowest IP, highest decline)
        assert npvs["HOWARD_01"] == min(npvs.values())
        
        # All should be profitable at $80 oil
        assert all(npv > 0 for npv in npvs.values())