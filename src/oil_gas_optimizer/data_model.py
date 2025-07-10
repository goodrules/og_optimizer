"""
Data models for oil & gas field development.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List
import random


@dataclass
class Lease:
    """Represents an oil & gas lease with drilling locations."""
    id: str
    available_wells: int
    basin: Optional[str] = None
    county: Optional[str] = None
    royalty: float = 0.1875  # Standard Texas royalty
    working_interest: float = 0.80
    
    def __post_init__(self):
        if self.available_wells < 0:
            raise ValueError("Available wells must be non-negative")
        if not 0 <= self.royalty <= 1:
            raise ValueError("Royalty must be between 0 and 1")
        if not 0 <= self.working_interest <= 1:
            raise ValueError("Working interest must be between 0 and 1")


@dataclass 
class DrillingParameters:
    """Parameters for drilling operations."""
    drill_days_per_well: int = 25
    rig_move_days: int = 5
    batch_drilling_efficiency: float = 0.85  # Time savings for pad drilling
    permit_delay_days: int = 0
    weather_delay_factor: float = 0.05  # 5% weather downtime
    
    def __post_init__(self):
        if self.drill_days_per_well <= 0:
            raise ValueError("Drill days must be positive")
        if not 0 <= self.batch_drilling_efficiency <= 1:
            raise ValueError("Batch efficiency must be between 0 and 1")


def generate_texas_wells(
    lease_id: str,
    n_wells: int,
    base_ip: float = 1000,
    base_capex: float = 7_500_000,
    ip_variation: float = 0.02,
    capex_variation: float = 0.10
) -> List:
    """
    Generate a list of wells for a Texas lease with realistic variation.
    
    Args:
        lease_id: Lease identifier (e.g., "MIDLAND_A")
        n_wells: Number of wells to generate
        base_ip: Base initial production rate (boe/d)
        base_capex: Base capital cost per well
        ip_variation: Variation in IP per well position (0.02 = 2% decline)
        capex_variation: Random variation in capex (0.10 = ±10%)
        
    Returns:
        List of WellEconomics objects
    """
    # Import here to avoid circular import
    from .economics import WellEconomics
    
    wells = []
    
    # Texas-specific decline parameters by basin
    basin_params = {
        "Permian": {"di": 0.70, "b": 1.1, "eur": 400},
        "Delaware": {"di": 0.68, "b": 1.2, "eur": 450},
        "Eagle Ford": {"di": 0.75, "b": 1.0, "eur": 350},
        "Barnett": {"di": 0.80, "b": 0.9, "eur": 300}
    }
    
    # Determine basin from lease name
    if "REEVES" in lease_id or "LOVING" in lease_id:
        basin = "Delaware"
    elif "EAGLE" in lease_id:
        basin = "Eagle Ford"
    elif "BARNETT" in lease_id:
        basin = "Barnett"
    else:
        basin = "Permian"  # Default
    
    params = basin_params[basin]
    
    for i in range(n_wells):
        # IP declines with well position (best locations drilled first)
        ip_factor = 1 - (i * ip_variation)
        ip_rate = base_ip * ip_factor * random.uniform(0.95, 1.05)
        
        # Capex varies randomly
        capex = base_capex * random.uniform(1 - capex_variation, 1 + capex_variation)
        
        # Create well
        well = WellEconomics(
            name=f"{lease_id}_{i+1:02d}",
            capex=capex,
            ip_rate=ip_rate,
            di=params["di"] * random.uniform(0.95, 1.05),
            b=params["b"] * random.uniform(0.95, 1.05),
            eur_mboe=params["eur"] * ip_factor
        )
        wells.append(well)
    
    return wells