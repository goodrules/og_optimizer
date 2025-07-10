"""
Test suite for oil & gas decline curve calculations.
Following TDD approach - RED phase.
"""
import pytest
import numpy as np
from datetime import date, timedelta

# These imports will fail initially (RED phase)
from src.oil_gas_optimizer.decline_curves import (
    hyperbolic_decline,
    calculate_monthly_production,
    calculate_cumulative_production,
    DeclineCurveParameters
)


class TestHyperbolicDecline:
    """Test hyperbolic decline curve calculations."""
    
    def test_hyperbolic_decline_at_time_zero(self):
        """Initial production rate should equal IP rate at time 0."""
        params = DeclineCurveParameters(
            ip_rate=1000,  # boe/d
            di=0.70,  # 70% annual decline
            b=1.1
        )
        
        production = hyperbolic_decline(params, time_months=0)
        assert production == 1000
    
    def test_hyperbolic_decline_after_one_year(self):
        """Production should decline appropriately after one year."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,  # 70% annual decline
            b=1.1
        )
        
        production = hyperbolic_decline(params, time_months=12)
        # With hyperbolic decline, production after 1 year should be less than IP
        assert production < 1000
        assert production > 0
        # Approximate expected value based on hyperbolic formula
        assert 500 < production < 700  # Expected range for these parameters
    
    def test_hyperbolic_decline_with_b_equals_one(self):
        """When b=1, should behave as harmonic decline."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.50,
            b=1.0  # Harmonic decline
        )
        
        production = hyperbolic_decline(params, time_months=12)
        # Harmonic decline: q = qi / (1 + Di*t)
        expected = 1000 / (1 + 0.50)  # Simplified for annual
        assert abs(production - expected) < 50  # Allow for monthly conversion
    
    def test_hyperbolic_decline_with_b_equals_zero(self):
        """When b=0, should behave as exponential decline."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.30,
            b=0.0  # Exponential decline
        )
        
        production = hyperbolic_decline(params, time_months=12)
        # Exponential decline: q = qi * exp(-Di*t)
        expected = 1000 * np.exp(-0.30)  # Annual
        assert abs(production - expected) < 100  # Allow for monthly conversion


class TestMonthlyProduction:
    """Test monthly production profile generation."""
    
    def test_calculate_monthly_production_single_well(self):
        """Generate monthly production for a single well."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,
            b=1.1
        )
        
        monthly_production = calculate_monthly_production(
            params, 
            months=24,
            start_date=date(2024, 1, 1)
        )
        
        assert len(monthly_production) == 24
        assert all('date' in month for month in monthly_production)
        assert all('production_boe' in month for month in monthly_production)
        assert all('day_rate_boed' in month for month in monthly_production)
        
        # Production should decline over time
        assert monthly_production[0]['day_rate_boed'] > monthly_production[-1]['day_rate_boed']
        # First month should be close to IP (using midpoint of month for calculation)
        assert 950 < monthly_production[0]['day_rate_boed'] < 1000
    
    def test_calculate_monthly_production_with_zero_months(self):
        """Should return empty list for zero months."""
        params = DeclineCurveParameters(ip_rate=1000, di=0.70, b=1.1)
        
        monthly_production = calculate_monthly_production(params, months=0)
        assert monthly_production == []
    
    def test_calculate_monthly_production_accounts_for_days_in_month(self):
        """Monthly production should account for varying days in month."""
        params = DeclineCurveParameters(ip_rate=1000, di=0.70, b=1.1)
        
        # Start in February to test 28/29 days
        monthly_production = calculate_monthly_production(
            params,
            months=2,
            start_date=date(2024, 2, 1)  # 2024 is a leap year
        )
        
        feb_production = monthly_production[0]['production_boe']
        mar_production = monthly_production[1]['production_boe']
        
        # February has 29 days (leap year), March has 31
        # Even with decline, March might have more total production due to more days
        assert feb_production > 0
        assert mar_production > 0


class TestCumulativeProduction:
    """Test cumulative production calculations."""
    
    def test_calculate_cumulative_production(self):
        """Calculate cumulative production over time."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,
            b=1.1
        )
        
        cumulative = calculate_cumulative_production(params, months=360)  # 30 years
        
        assert cumulative > 0
        # For these parameters, EUR should be in reasonable range
        assert 1_500_000 < cumulative < 2_000_000  # boe
    
    def test_cumulative_production_increases_with_time(self):
        """Cumulative production should increase with time."""
        params = DeclineCurveParameters(ip_rate=1000, di=0.70, b=1.1)
        
        cum_12_months = calculate_cumulative_production(params, months=12)
        cum_24_months = calculate_cumulative_production(params, months=24)
        cum_36_months = calculate_cumulative_production(params, months=36)
        
        assert cum_12_months > 0
        assert cum_24_months > cum_12_months
        assert cum_36_months > cum_24_months
    
    def test_cumulative_production_approaches_eur(self):
        """Cumulative production should approach EUR over long time."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,
            b=1.1,
            eur_mboe=400  # 400,000 boe EUR
        )
        
        # After many years, should approach EUR
        cumulative = calculate_cumulative_production(params, months=600)  # 50 years
        
        assert cumulative <= 400_000  # Should not exceed EUR
        assert cumulative > 350_000  # Should be close to EUR after 50 years


class TestDeclineCurveParameters:
    """Test DeclineCurveParameters dataclass."""
    
    def test_decline_curve_parameters_validation(self):
        """Parameters should validate input ranges."""
        # Valid parameters
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,
            b=1.1
        )
        assert params.ip_rate == 1000
        assert params.di == 0.70
        assert params.b == 1.1
    
    def test_decline_curve_parameters_invalid_ip_rate(self):
        """Should raise error for invalid IP rate."""
        with pytest.raises(ValueError, match="IP rate must be positive"):
            DeclineCurveParameters(ip_rate=-100, di=0.70, b=1.1)
    
    def test_decline_curve_parameters_invalid_di(self):
        """Should raise error for invalid decline rate."""
        with pytest.raises(ValueError, match="Decline rate must be between 0 and 1"):
            DeclineCurveParameters(ip_rate=1000, di=1.5, b=1.1)
    
    def test_decline_curve_parameters_invalid_b_factor(self):
        """Should raise error for invalid b factor."""
        with pytest.raises(ValueError, match="b factor must be between 0 and 2"):
            DeclineCurveParameters(ip_rate=1000, di=0.70, b=2.5)
    
    def test_decline_curve_parameters_with_eur(self):
        """Should accept optional EUR parameter."""
        params = DeclineCurveParameters(
            ip_rate=1000,
            di=0.70,
            b=1.1,
            eur_mboe=450
        )
        assert params.eur_mboe == 450