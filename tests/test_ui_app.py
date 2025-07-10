"""
Test suite for UI application components.
"""
import pytest
from unittest.mock import Mock, MagicMock

from src.oil_gas_optimizer.ui_app import (
    create_well_portfolio,
    add_trial,
    update_metrics_card,
    TEXAS_LEASES
)
from src.oil_gas_optimizer.heuristic_optimizer import OptimizationKnobs, OptimizationMetrics
from src.oil_gas_optimizer.economics import WellEconomics


class TestUIComponents:
    """Test UI helper functions."""
    
    def test_create_well_portfolio(self):
        """Test creation of Texas well portfolio."""
        wells = create_well_portfolio()
        
        # Should create wells for all leases
        assert len(wells) > 0
        
        # Count wells per lease
        lease_counts = {}
        for well in wells:
            lease = "_".join(well.name.split("_")[:2])
            lease_counts[lease] = lease_counts.get(lease, 0) + 1
        
        # Verify expected counts
        assert lease_counts["MIDLAND_A"] == 28
        assert lease_counts["REEVES_C"] == 32
        assert lease_counts["HOWARD_E"] == 12
        
        # All wells should have Texas-appropriate parameters
        for well in wells:
            assert 6_000_000 <= well.capex <= 10_000_000  # Wider range for premium wells
            assert 400 <= well.ip_rate <= 1500  # Wide range due to 2% decline per well
            assert 0.6 <= well.di <= 0.85
    
    def test_add_trial(self):
        """Test adding optimization trial to history."""
        # Mock client storage
        client = {
            "trials": [],
            "stats_html": Mock(content=""),
            "timeline_chart": Mock(figure=None, update=Mock()),
            "production_chart": Mock(figure=None, update=Mock()),
            "economics_chart": Mock(figure=None, update=Mock()),
            "history_plot": Mock(figure=None, update=Mock())
        }
        
        knobs = OptimizationKnobs(
            wells_per_lease={"MIDLAND_A": 5},
            rig_count=2,
            oil_price_forecast=80.0
        )
        
        metrics = {
            "total_npv": 100_000_000,
            "total_capex": 50_000_000,
            "npv_per_dollar": 2.0,
            "peak_production": 5000,
            "wells_drilled": 5,
            "risk_score": 0.3
        }
        
        # Add trial
        add_trial(knobs, metrics, client)
        
        # Verify trial was added
        assert len(client["trials"]) == 1
        assert client["trials"][0]["knobs"] == knobs
        assert client["trials"][0]["metrics"] == metrics
        
        # Verify UI updates were called
        assert client["timeline_chart"].update.called
        assert client["production_chart"].update.called
    
    def test_update_metrics_card(self):
        """Test metrics card update."""
        client = {
            "trials": [{
                "knobs": OptimizationKnobs(wells_per_lease={"A": 5}, rig_count=2),
                "metrics": {
                    "total_npv": 120_000_000,
                    "total_capex": 60_000_000,
                    "npv_per_dollar": 2.0,
                    "peak_production": 6000,
                    "wells_drilled": 8,
                    "risk_score": 0.25
                }
            }],
            "stats_html": Mock(content=""),
            "initial_npv": 100_000_000
        }
        
        # Update metrics for first trial
        update_metrics_card(0, client)
        
        # Check that HTML was updated
        html = client["stats_html"].content
        assert "$120.0MM" in html  # NPV
        assert "$60.0MM" in html   # CAPEX
        assert "$20.0MM" in html   # Improvement
        assert "2.00" in html      # NPV per dollar
        assert "6000" in html      # Peak production
        assert "25%" in html       # Risk score
    
    def test_texas_lease_configuration(self):
        """Test Texas lease configurations."""
        # Verify all expected leases are defined
        assert "MIDLAND_A" in TEXAS_LEASES
        assert "REEVES_C" in TEXAS_LEASES
        assert "HOWARD_E" in TEXAS_LEASES
        
        # Verify Permian/Delaware classification
        assert TEXAS_LEASES["MIDLAND_A"]["basin"] == "Permian"
        assert TEXAS_LEASES["REEVES_C"]["basin"] == "Delaware"
        
        # Verify realistic well counts
        for lease, config in TEXAS_LEASES.items():
            assert 5 <= config["wells"] <= 32
            assert 800 <= config["ip_base"] <= 1400
            assert 6_000_000 <= config["capex"] <= 9_000_000