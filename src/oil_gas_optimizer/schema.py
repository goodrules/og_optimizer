"""
JSON Schema definitions for oil & gas optimization parameters.
Used for structured output from LLMs like Gemini.
"""

# Oil & Gas Field Development Optimization Parameters Schema
OPTIMIZATION_PARAMS_SCHEMA = {
    "description": "Oil and gas field development optimization parameters extracted from user requirements or analysis",
    "type": "OBJECT",
    "properties": {
        "economic_parameters": {
            "description": "Economic assumptions for the field development",
            "type": "OBJECT",
            "properties": {
                "oil_price_per_barrel": {
                    "description": "Expected oil price in USD per barrel (range: 40-120)",
                    "type": "NUMBER"
                },
                "discount_rate_percent": {
                    "description": "Discount rate as a percentage for NPV calculations (range: 8-20)",
                    "type": "NUMBER"
                },
                "contingency_percent": {
                    "description": "Contingency reserve as a percentage of budget (range: 10-30)",
                    "type": "NUMBER"
                },
                "capex_budget_millions": {
                    "description": "Total capital expenditure budget in millions USD (range: 50-200)",
                    "type": "NUMBER"
                }
            },
            "required": []
        },
        "drilling_execution": {
            "description": "Parameters related to drilling operations and execution",
            "type": "OBJECT",
            "properties": {
                "horizontal_rigs_available": {
                    "description": "Number of horizontal drilling rigs available (range: 1-5)",
                    "type": "INTEGER"
                },
                "drilling_mode": {
                    "description": "Drilling execution mode",
                    "type": "STRING",
                    "enum": ["continuous", "batch"]
                },
                "permit_delay_days": {
                    "description": "Expected permit delay in days (range: 0-60, increment: 5)",
                    "type": "INTEGER"
                }
            },
            "required": []
        },
        "well_selection": {
            "description": "Number of wells to drill per lease",
            "type": "OBJECT",
            "properties": {
                "MIDLAND_A": {
                    "description": "Number of wells to drill in MIDLAND_A lease (Permian basin, max: 28)",
                    "type": "INTEGER"
                },
                "MARTIN_B": {
                    "description": "Number of wells to drill in MARTIN_B lease (Permian basin, max: 15)",
                    "type": "INTEGER"
                },
                "REEVES_C": {
                    "description": "Number of wells to drill in REEVES_C lease (Delaware basin, max: 32)",
                    "type": "INTEGER"
                },
                "LOVING_D": {
                    "description": "Number of wells to drill in LOVING_D lease (Delaware basin, max: 22)",
                    "type": "INTEGER"
                },
                "HOWARD_E": {
                    "description": "Number of wells to drill in HOWARD_E lease (Permian basin, max: 12)",
                    "type": "INTEGER"
                }
            },
            "required": []
        },
        "optimization_guidance": {
            "description": "Optional guidance for the optimization process",
            "type": "OBJECT",
            "properties": {
                "primary_objective": {
                    "description": "Primary optimization objective",
                    "type": "STRING",
                    "enum": ["maximize_npv", "minimize_risk", "maximize_production", "balance_portfolio"]
                },
                "risk_tolerance": {
                    "description": "Risk tolerance level",
                    "type": "STRING",
                    "enum": ["conservative", "moderate", "aggressive"]
                },
                "development_pace": {
                    "description": "Preferred development pace",
                    "type": "STRING",
                    "enum": ["slow", "moderate", "fast"]
                }
            },
            "required": []
        }
    },
    "required": []
}


def validate_optimization_params(params: dict) -> tuple[bool, list[str]]:
    """
    Validate optimization parameters against schema constraints.
    
    Args:
        params: Dictionary of optimization parameters
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate economic parameters
    if "economic_parameters" in params:
        econ = params["economic_parameters"]
        
        if "oil_price_per_barrel" in econ:
            if not 40 <= econ["oil_price_per_barrel"] <= 120:
                errors.append("oil_price_per_barrel must be between 40 and 120")
                
        if "discount_rate_percent" in econ:
            if not 8 <= econ["discount_rate_percent"] <= 20:
                errors.append("discount_rate_percent must be between 8 and 20")
                
        if "contingency_percent" in econ:
            if not 10 <= econ["contingency_percent"] <= 30:
                errors.append("contingency_percent must be between 10 and 30")
                
        if "capex_budget_millions" in econ:
            if not 50 <= econ["capex_budget_millions"] <= 200:
                errors.append("capex_budget_millions must be between 50 and 200")
    
    # Validate drilling execution
    if "drilling_execution" in params:
        drill = params["drilling_execution"]
        
        if "horizontal_rigs_available" in drill:
            if not 1 <= drill["horizontal_rigs_available"] <= 5:
                errors.append("horizontal_rigs_available must be between 1 and 5")
                
        if "drilling_mode" in drill:
            if drill["drilling_mode"] not in ["continuous", "batch"]:
                errors.append("drilling_mode must be 'continuous' or 'batch'")
                
        if "permit_delay_days" in drill:
            if not 0 <= drill["permit_delay_days"] <= 60:
                errors.append("permit_delay_days must be between 0 and 60")
            if drill["permit_delay_days"] % 5 != 0:
                errors.append("permit_delay_days must be in increments of 5")
    
    # Validate well selection
    if "well_selection" in params:
        wells = params["well_selection"]
        lease_limits = {
            "MIDLAND_A": 28,
            "MARTIN_B": 15,
            "REEVES_C": 32,
            "LOVING_D": 22,
            "HOWARD_E": 12
        }
        
        for lease, limit in lease_limits.items():
            if lease in wells:
                if not 0 <= wells[lease] <= limit:
                    errors.append(f"{lease} must be between 0 and {limit}")
    
    # Validate optimization guidance
    if "optimization_guidance" in params:
        guidance = params["optimization_guidance"]
        
        if "primary_objective" in guidance:
            valid_objectives = ["maximize_npv", "minimize_risk", "maximize_production", "balance_portfolio"]
            if guidance["primary_objective"] not in valid_objectives:
                errors.append(f"primary_objective must be one of: {', '.join(valid_objectives)}")
                
        if "risk_tolerance" in guidance:
            valid_risk = ["conservative", "moderate", "aggressive"]
            if guidance["risk_tolerance"] not in valid_risk:
                errors.append(f"risk_tolerance must be one of: {', '.join(valid_risk)}")
                
        if "development_pace" in guidance:
            valid_pace = ["slow", "moderate", "fast"]
            if guidance["development_pace"] not in valid_pace:
                errors.append(f"development_pace must be one of: {', '.join(valid_pace)}")
    
    return len(errors) == 0, errors


def apply_params_to_ui(params: dict, ui_elements: dict) -> dict:
    """
    Apply validated parameters to UI elements.
    
    Args:
        params: Validated optimization parameters
        ui_elements: Dictionary containing UI element references
        
    Returns:
        Dictionary of applied changes
    """
    applied = {}
    
    # Apply economic parameters
    if "economic_parameters" in params:
        econ = params["economic_parameters"]
        
        if "oil_price_per_barrel" in econ and "oil_price" in ui_elements:
            ui_elements["oil_price"].value = econ["oil_price_per_barrel"]
            applied["oil_price"] = econ["oil_price_per_barrel"]
            
        if "discount_rate_percent" in econ and "discount_rate" in ui_elements:
            ui_elements["discount_rate"].value = econ["discount_rate_percent"]
            applied["discount_rate"] = econ["discount_rate_percent"]
            
        if "contingency_percent" in econ and "contingency" in ui_elements:
            ui_elements["contingency"].value = econ["contingency_percent"]
            applied["contingency"] = econ["contingency_percent"]
            
        if "capex_budget_millions" in econ and "budget" in ui_elements:
            ui_elements["budget"].value = econ["capex_budget_millions"]
            applied["budget"] = econ["capex_budget_millions"]
    
    # Apply drilling execution
    if "drilling_execution" in params:
        drill = params["drilling_execution"]
        
        if "horizontal_rigs_available" in drill and "rigs" in ui_elements:
            ui_elements["rigs"].value = drill["horizontal_rigs_available"]
            applied["rigs"] = drill["horizontal_rigs_available"]
            
        if "drilling_mode" in drill and "drilling_mode" in ui_elements:
            ui_elements["drilling_mode"].value = drill["drilling_mode"]
            applied["drilling_mode"] = drill["drilling_mode"]
            
        if "permit_delay_days" in drill and "permit_delay" in ui_elements:
            ui_elements["permit_delay"].value = drill["permit_delay_days"]
            applied["permit_delay"] = drill["permit_delay_days"]
    
    # Apply well selection
    if "well_selection" in params and "well_sliders" in ui_elements:
        wells = params["well_selection"]
        
        for lease, count in wells.items():
            if lease in ui_elements["well_sliders"]:
                ui_elements["well_sliders"][lease].value = count
                applied[f"wells_{lease}"] = count
    
    return applied