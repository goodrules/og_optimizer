"""
Test suite for oil & gas economic calculations (NPV, IRR).
Following TDD approach - RED phase.
"""
import pytest
import numpy as np
from datetime import date

# These imports will fail initially (RED phase)
from src.oil_gas_optimizer.economics import (
    calculate_well_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_field_economics,
    EconomicParameters,
    WellEconomics,
    CashFlow
)


class TestWellNPV:
    """Test NPV calculations for individual wells."""
    
    def test_calculate_well_npv_positive_case(self):
        """Calculate NPV for a profitable well."""
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80,
            severance_tax=0.046,
            ad_valorem_tax=0.02
        )
        
        well = WellEconomics(
            name="MIDLAND_A_01",
            capex=7_500_000,
            ip_rate=1000,
            di=0.70,
            b=1.1,
            months=360
        )
        
        npv = calculate_well_npv(well, econ_params)
        
        # With $80 oil, this well should be profitable
        assert npv > 0
        assert 20_000_000 < npv < 35_000_000  # Reasonable range for high IP well
    
    def test_calculate_well_npv_negative_case(self):
        """Calculate NPV for an unprofitable well."""
        econ_params = EconomicParameters(
            oil_price=30.0,  # Low oil price
            discount_rate=0.15,  # High discount rate
            opex_per_boe=20.0,  # High operating costs
            royalty=0.25,
            working_interest=0.60,
            severance_tax=0.046,
            ad_valorem_tax=0.02
        )
        
        well = WellEconomics(
            name="POOR_WELL_01",
            capex=10_000_000,  # High capex
            ip_rate=500,  # Low IP
            di=0.85,  # High decline
            b=1.0,
            months=360
        )
        
        npv = calculate_well_npv(well, econ_params)
        
        # This well should be unprofitable
        assert npv < 0
    
    def test_calculate_well_npv_with_delayed_start(self):
        """NPV should decrease with delayed start due to discounting."""
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        well = WellEconomics(
            name="DELAYED_WELL",
            capex=7_500_000,
            ip_rate=1000,
            di=0.70,
            b=1.1,
            months=360
        )
        
        npv_immediate = calculate_well_npv(well, econ_params, start_month=0)
        npv_delayed_6m = calculate_well_npv(well, econ_params, start_month=6)
        npv_delayed_12m = calculate_well_npv(well, econ_params, start_month=12)
        
        # NPV should decrease with delay due to time value of money
        assert npv_immediate > npv_delayed_6m > npv_delayed_12m


class TestIRR:
    """Test Internal Rate of Return calculations."""
    
    def test_calculate_irr_positive_project(self):
        """Calculate IRR for a profitable project."""
        cash_flows = CashFlow(
            dates=[date(2024, 1, 1) + timedelta(days=30*i) for i in range(37)],
            values=[-7_500_000] + [250_000] * 36  # Initial investment + 36 months of positive cash
        )
        
        irr = calculate_irr(cash_flows)
        
        assert irr > 0
        assert 0.10 < irr < 0.50  # Reasonable IRR range (10-50% annually)
    
    def test_calculate_irr_negative_project(self):
        """IRR should be None or negative for unprofitable project."""
        cash_flows = CashFlow(
            dates=[date(2024, 1, 1) + timedelta(days=30*i) for i in range(37)],
            values=[-10_000_000] + [100_000] * 36  # Won't recover investment
        )
        
        irr = calculate_irr(cash_flows)
        
        # IRR might not exist or be negative
        assert irr is None or irr < 0
    
    def test_calculate_irr_from_well_economics(self):
        """Calculate IRR from complete well economics."""
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        well = WellEconomics(
            name="IRR_TEST_WELL",
            capex=7_500_000,
            ip_rate=1000,
            di=0.70,
            b=1.1,
            months=60  # 5 years
        )
        
        irr = calculate_irr(well.get_cash_flows(econ_params))
        
        assert irr > 0.15  # Should exceed 15% hurdle rate
        assert irr < 3.0   # Sanity check - very high for good wells


class TestPaybackPeriod:
    """Test payback period calculations."""
    
    def test_calculate_payback_period_simple(self):
        """Calculate payback period for simple cash flows."""
        cash_flows = CashFlow(
            dates=[date(2024, 1, 1) + timedelta(days=30*i) for i in range(37)],
            values=[-7_500_000] + [500_000] * 36
        )
        
        payback_months = calculate_payback_period(cash_flows)
        
        # Should pay back in 15 months (7.5M / 0.5M per month)
        assert payback_months == 15
    
    def test_calculate_payback_period_never_payback(self):
        """Payback period should be None if investment never recovered."""
        cash_flows = CashFlow(
            dates=[date(2024, 1, 1) + timedelta(days=30*i) for i in range(37)],
            values=[-10_000_000] + [100_000] * 36  # Only recovers 3.6M
        )
        
        payback_months = calculate_payback_period(cash_flows)
        
        assert payback_months is None
    
    def test_calculate_payback_period_with_varying_cash_flows(self):
        """Calculate payback with declining cash flows."""
        # Simulate declining production
        values = [-7_500_000]
        monthly_flow = 800_000
        for i in range(36):
            values.append(monthly_flow)
            monthly_flow *= 0.95  # 5% monthly decline
        
        cash_flows = CashFlow(
            dates=[date(2024, 1, 1) + timedelta(days=30*i) for i in range(37)],
            values=values
        )
        
        payback_months = calculate_payback_period(cash_flows)
        
        assert payback_months is not None
        assert 10 < payback_months < 20  # Reasonable range


class TestFieldEconomics:
    """Test field-level economic aggregation."""
    
    def test_calculate_field_economics_multiple_wells(self):
        """Aggregate economics for multiple wells."""
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        wells = [
            WellEconomics(
                name=f"WELL_{i:02d}",
                capex=7_500_000,
                ip_rate=1000 - i*50,  # Varying quality
                di=0.70,
                b=1.1,
                months=360
            )
            for i in range(5)
        ]
        
        # Drilling schedule: 1 well per month
        drilling_schedule = {
            date(2024, i+1, 1): [wells[i]]
            for i in range(5)
        }
        
        field_metrics = calculate_field_economics(
            drilling_schedule,
            econ_params
        )
        
        assert 'total_npv' in field_metrics
        assert 'total_capex' in field_metrics
        assert 'peak_production_boed' in field_metrics
        assert 'total_eur_mboe' in field_metrics
        assert 'average_irr' in field_metrics
        
        assert field_metrics['total_npv'] > 0
        assert field_metrics['total_capex'] == 5 * 7_500_000
        assert field_metrics['peak_production_boed'] > 3000  # Multiple wells producing
    
    def test_calculate_field_economics_empty_schedule(self):
        """Handle empty drilling schedule."""
        econ_params = EconomicParameters(oil_price=80.0, discount_rate=0.10)
        
        field_metrics = calculate_field_economics({}, econ_params)
        
        assert field_metrics['total_npv'] == 0
        assert field_metrics['total_capex'] == 0
        assert field_metrics['peak_production_boed'] == 0
        assert field_metrics['total_eur_mboe'] == 0
    
    def test_calculate_field_economics_with_constraints(self):
        """Field economics with capital constraints."""
        econ_params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        # Create more wells than budget allows
        wells = [
            WellEconomics(
                name=f"WELL_{i:02d}",
                capex=7_500_000,
                ip_rate=1000,
                di=0.70,
                b=1.1
            )
            for i in range(10)
        ]
        
        # Budget constraint of $30M (only 4 wells)
        field_metrics = calculate_field_economics(
            wells,
            econ_params,
            capital_budget=30_000_000
        )
        
        assert field_metrics['total_capex'] <= 30_000_000
        assert field_metrics['wells_drilled'] == 4


class TestEconomicParameters:
    """Test EconomicParameters validation."""
    
    def test_economic_parameters_validation(self):
        """Parameters should validate input ranges."""
        params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            opex_per_boe=12.0,
            royalty=0.1875,
            working_interest=0.80
        )
        
        assert params.oil_price == 80.0
        assert params.discount_rate == 0.10
    
    def test_economic_parameters_invalid_oil_price(self):
        """Should raise error for invalid oil price."""
        with pytest.raises(ValueError, match="Oil price must be positive"):
            EconomicParameters(oil_price=-10.0, discount_rate=0.10)
    
    def test_economic_parameters_invalid_discount_rate(self):
        """Should raise error for invalid discount rate."""
        with pytest.raises(ValueError, match="Discount rate must be between 0 and 1"):
            EconomicParameters(oil_price=80.0, discount_rate=1.5)
    
    def test_economic_parameters_net_revenue_interest(self):
        """Calculate net revenue interest correctly."""
        params = EconomicParameters(
            oil_price=80.0,
            discount_rate=0.10,
            royalty=0.1875,
            working_interest=0.80
        )
        
        nri = params.net_revenue_interest
        expected_nri = 0.80 * (1 - 0.1875)
        assert abs(nri - expected_nri) < 0.001


# Import for timedelta
from datetime import timedelta