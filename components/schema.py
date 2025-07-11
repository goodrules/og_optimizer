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
        "risk_analysis": {
            "description": "Risk analysis configuration including Monte Carlo simulation",
            "type": "OBJECT",
            "properties": {
                "use_monte_carlo": {
                    "description": "Enable Monte Carlo simulation for probabilistic risk analysis",
                    "type": "BOOLEAN"
                },
                "monte_carlo_simulations": {
                    "description": "Number of Monte Carlo simulation scenarios to run (range: 10-1000)",
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
        },
        "parameter_locks": {
            "description": "Lock or unlock specific parameters to control optimizer behavior",
            "type": "OBJECT",
            "properties": {
                "lock_oil_price": {
                    "description": "Lock the oil price slider to prevent optimizer from changing it",
                    "type": "BOOLEAN"
                },
                "lock_discount_rate": {
                    "description": "Lock the discount rate slider",
                    "type": "BOOLEAN"
                },
                "lock_contingency": {
                    "description": "Lock the contingency percentage slider",
                    "type": "BOOLEAN"
                },
                "lock_rigs": {
                    "description": "Lock the number of rigs slider",
                    "type": "BOOLEAN"
                },
                "lock_drilling_mode": {
                    "description": "Lock the drilling mode toggle",
                    "type": "BOOLEAN"
                },
                "lock_permit_delay": {
                    "description": "Lock the permit delay slider",
                    "type": "BOOLEAN"
                },
                "lock_wells": {
                    "description": "Lock specific lease well counts. Use lease IDs as keys with boolean values",
                    "type": "OBJECT",
                    "properties": {
                        "MIDLAND_A": {"type": "BOOLEAN"},
                        "MARTIN_B": {"type": "BOOLEAN"},
                        "REEVES_C": {"type": "BOOLEAN"},
                        "LOVING_D": {"type": "BOOLEAN"},
                        "HOWARD_E": {"type": "BOOLEAN"}
                    }
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
    
    # Validate risk analysis
    if "risk_analysis" in params:
        risk = params["risk_analysis"]
        
        if "monte_carlo_simulations" in risk:
            if not 10 <= risk["monte_carlo_simulations"] <= 1000:
                errors.append("monte_carlo_simulations must be between 10 and 1000")
    
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
    
    # Apply risk analysis
    if "risk_analysis" in params:
        risk = params["risk_analysis"]
        
        if "use_monte_carlo" in risk and "monte_carlo_toggle" in ui_elements:
            ui_elements["monte_carlo_toggle"].value = risk["use_monte_carlo"]
            applied["use_monte_carlo"] = risk["use_monte_carlo"]
            
            # Enable/disable simulations input based on toggle
            if "n_simulations" in ui_elements:
                ui_elements["n_simulations"].set_enabled(risk["use_monte_carlo"])
            
        if "monte_carlo_simulations" in risk and "n_simulations" in ui_elements:
            ui_elements["n_simulations"].value = risk["monte_carlo_simulations"]
            applied["monte_carlo_simulations"] = risk["monte_carlo_simulations"]
    
    # Apply well selection
    if "well_selection" in params and "well_sliders" in ui_elements:
        wells = params["well_selection"]
        
        for lease, count in wells.items():
            if lease in ui_elements["well_sliders"]:
                ui_elements["well_sliders"][lease].value = count
                applied[f"wells_{lease}"] = count
    
    # Apply parameter locks
    if "parameter_locks" in params:
        locks = params["parameter_locks"]
        
        # Helper function to apply lock state
        def apply_lock(lock_icon, control, should_lock):
            current_state = lock_icon.name == 'lock'
            if current_state != should_lock:
                if should_lock:
                    lock_icon.name = 'lock'
                    lock_icon.classes(replace='cursor-pointer text-orange-600')
                    control._locked = True
                    if hasattr(control, 'props'):
                        control.props(add='color=orange')
                else:
                    lock_icon.name = 'lock_open'
                    lock_icon.classes(replace='cursor-pointer text-gray-400')
                    control._locked = False
                    if hasattr(control, 'props'):
                        control.props(remove='color=orange')
        
        if "lock_oil_price" in locks and "oil_price_lock" in ui_elements and "oil_price" in ui_elements:
            apply_lock(ui_elements["oil_price_lock"], ui_elements["oil_price"], locks["lock_oil_price"])
            applied["lock_oil_price"] = locks["lock_oil_price"]
            
        if "lock_discount_rate" in locks and "discount_rate_lock" in ui_elements and "discount_rate" in ui_elements:
            apply_lock(ui_elements["discount_rate_lock"], ui_elements["discount_rate"], locks["lock_discount_rate"])
            applied["lock_discount_rate"] = locks["lock_discount_rate"]
            
        if "lock_contingency" in locks and "contingency_lock" in ui_elements and "contingency" in ui_elements:
            apply_lock(ui_elements["contingency_lock"], ui_elements["contingency"], locks["lock_contingency"])
            applied["lock_contingency"] = locks["lock_contingency"]
            
        if "lock_rigs" in locks and "rigs_lock" in ui_elements and "rigs" in ui_elements:
            apply_lock(ui_elements["rigs_lock"], ui_elements["rigs"], locks["lock_rigs"])
            applied["lock_rigs"] = locks["lock_rigs"]
            
        if "lock_drilling_mode" in locks and "drilling_mode_lock" in ui_elements and "drilling_mode" in ui_elements:
            apply_lock(ui_elements["drilling_mode_lock"], ui_elements["drilling_mode"], locks["lock_drilling_mode"])
            applied["lock_drilling_mode"] = locks["lock_drilling_mode"]
            
        if "lock_permit_delay" in locks and "permit_delay_lock" in ui_elements and "permit_delay" in ui_elements:
            apply_lock(ui_elements["permit_delay_lock"], ui_elements["permit_delay"], locks["lock_permit_delay"])
            applied["lock_permit_delay"] = locks["lock_permit_delay"]
            
        # Handle well locks - need special handling for well sliders
        if "lock_wells" in locks and "well_locks" in ui_elements and "well_sliders" in ui_elements:
            for lease, should_lock in locks["lock_wells"].items():
                if lease in ui_elements["well_locks"] and lease in ui_elements["well_sliders"]:
                    lock_icon = ui_elements["well_locks"][lease]
                    slider = ui_elements["well_sliders"][lease]
                    
                    current_state = lock_icon.name == 'lock'
                    if current_state != should_lock:
                        if should_lock:
                            lock_icon.name = 'lock'
                            lock_icon.classes(replace='cursor-pointer text-orange-600')
                            slider._locked = True
                            slider.disable()
                            slider.props(add='color=orange')
                            # Update label color if available
                            if "well_labels" in ui_elements and lease in ui_elements["well_labels"]:
                                ui_elements["well_labels"][lease].classes(replace='text-sm font-semibold w-20 text-right text-orange-600')
                        else:
                            lock_icon.name = 'lock_open'
                            lock_icon.classes(replace='cursor-pointer text-gray-400')
                            slider._locked = False
                            slider.enable()
                            slider.props(remove='color=orange')
                            # Update label color if available
                            if "well_labels" in ui_elements and lease in ui_elements["well_labels"]:
                                ui_elements["well_labels"][lease].classes(replace='text-sm font-semibold w-20 text-right text-gray-700')
                    
                    applied[f"lock_wells_{lease}"] = should_lock
    
    return applied