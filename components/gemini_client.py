"""
Gemini 2.5 Pro/Flash integration for Oil & Gas Field Development Optimizer
Handles controlled generation and chat interactions
Supports model selection between Pro (accuracy) and Flash (speed)
"""
import os
import json
import asyncio
import queue
import threading
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .system_prompt import SYSTEM_PROMPT, RUNTIME_CONTEXT_TEMPLATE
from .schema import OPTIMIZATION_PARAMS_SCHEMA, validate_optimization_params

# Load environment variables from .env file
load_dotenv()

tools = [
    types.Tool(google_search=types.GoogleSearch())
    ]

@dataclass
class ChatContext:
    """Maintains context for the chat session"""
    trial_data: Optional[Dict[str, Any]] = None
    current_params: Optional[Dict[str, Any]] = None
    history: list = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class GeminiOptimizationClient:
    """Client for interacting with Gemini 2.5 Pro for optimization guidance via Vertex AI"""
    
    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize Gemini client with Vertex AI configuration"""
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.region = region or os.getenv("GCP_REGION", "us-central1")
        
        if not self.project_id:
            raise ValueError("GCP Project ID not found. Set GCP_PROJECT_ID environment variable.")
        
        # Initialize Vertex AI client
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.region
        )
        
        # Model configuration - default to pro model
        self.model_name = model_name or "gemini-2.5-pro"
        
        # Generation config for structured output
        self.structured_config = {
            "response_mime_type": "application/json",
            "response_schema": OPTIMIZATION_PARAMS_SCHEMA,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 65535
        }
        
        # Generation config for general chat
        self.chat_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 65535,
            "tools": tools
        }
        
        self.context = ChatContext()
    
    def update_context(self, trial_data: Dict[str, Any], current_params: Dict[str, Any]) -> None:
        """Update the chat context with current trial information"""
        self.context.trial_data = trial_data
        self.context.current_params = current_params
    
    def _build_runtime_context(self) -> str:
        """Build runtime context string from current data"""
        if not self.context.trial_data:
            return "No trial data available yet. Run optimization to see results."
        
        trial = self.context.trial_data
        params = self.context.current_params or {}
        
        # Format wells per lease details
        wells_details = []
        for lease_id, count in params.get("wells_per_lease", {}).items():
            wells_details.append(f"- {lease_id}: {count} wells")
        
        return RUNTIME_CONTEXT_TEMPLATE.format(
            trial_number=trial.get("trial_number", 0),
            npv_mm=trial.get("npv_mm", 0),
            capex_mm=trial.get("capex_mm", 0),
            wells_selected=trial.get("wells_selected", 0),
            risk_score=trial.get("risk_score", 50),
            avg_production=trial.get("avg_production", 0),
            npv_per_dollar=trial.get("npv_per_dollar", 0),
            oil_price=params.get("oil_price", 80),
            discount_rate=params.get("discount_rate", 15),
            contingency=params.get("contingency", 20),
            budget_mm=params.get("budget", 100),
            rigs=params.get("rigs", 2),
            drilling_mode=params.get("drilling_mode", "continuous"),
            permit_delay=params.get("permit_delay", 30),
            use_monte_carlo="Enabled" if params.get("use_monte_carlo", False) else "Disabled",
            monte_carlo_simulations=params.get("monte_carlo_simulations", 100),
            wells_per_lease_details="\n".join(wells_details) if wells_details else "No wells selected"
        )
    
    async def process_message_stream(self, user_message: str, update_callback=None):
        """
        Process a user message with streaming response
        
        Args:
            user_message: The user's input message
            update_callback: Optional async callback function to handle streaming updates
            
        Yields:
            Streaming text chunks or final (response_text, parameter_updates) tuple
        """
        # Add runtime context to the message
        context_message = self._build_runtime_context()
        
        # Build the full prompt with system instruction
        full_prompt = f"{SYSTEM_PROMPT}\n\n{context_message}\n\nUser: {user_message}"
        
        # Determine if this is likely a parameter adjustment request
        param_keywords = [
            "set", "change", "adjust", "update", "configure",
            "conservative", "aggressive", "moderate",
            "increase", "decrease", "optimize for",
            "monte carlo", "enable", "disable", "simulation", "simulations"
        ]
        
        is_param_request = any(keyword in user_message.lower() for keyword in param_keywords)
        
        try:
            if is_param_request:
                # Use controlled generation for parameter updates (no streaming for JSON)
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=full_prompt,
                    config=self.structured_config
                )
                
                # Parse the JSON response
                if response.text:
                    try:
                        param_updates = json.loads(response.text)
                        # Validate the parameters
                        is_valid, errors = validate_optimization_params(param_updates)
                        
                        if is_valid:
                            # Generate a descriptive message about what was changed
                            description = self._describe_parameter_changes(param_updates)
                            if update_callback:
                                await update_callback(description, param_updates, is_final=True)
                            return description, param_updates
                        else:
                            error_msg = f"I encountered some issues with the parameters: {', '.join(errors)}"
                            if update_callback:
                                await update_callback(error_msg, None, is_final=True)
                            return error_msg, None
                    except json.JSONDecodeError:
                        error_msg = "I had trouble generating valid parameters. Please try rephrasing your request."
                        if update_callback:
                            await update_callback(error_msg, None, is_final=True)
                        return error_msg, None
            else:
                # Use streaming generation for analysis/questions
                accumulated_text = ""
                
                # Create a queue for thread-safe communication
                chunk_queue = queue.Queue()
                stream_done = False
                
                # Define a function to process stream in thread
                def process_stream():
                    nonlocal stream_done
                    try:
                        # Get streaming response
                        for chunk in self.client.models.generate_content_stream(
                            model=self.model_name,
                            contents=full_prompt,
                            config=self.chat_config
                        ):
                            if chunk.text:
                                chunk_queue.put(chunk.text)
                    finally:
                        stream_done = True
                        chunk_queue.put(None)  # Sentinel to indicate completion
                
                # Start streaming in background thread
                stream_thread = threading.Thread(target=process_stream)
                stream_thread.start()
                
                # Process chunks as they arrive
                while True:
                    try:
                        # Check for new chunks with a short timeout
                        chunk_text = chunk_queue.get(timeout=0.1)
                        
                        if chunk_text is None:  # Sentinel value
                            break
                            
                        accumulated_text += chunk_text
                        if update_callback:
                            await update_callback(accumulated_text, None, is_final=False)
                            
                    except queue.Empty:
                        # No chunk available yet, check if stream is done
                        if stream_done and chunk_queue.empty():
                            break
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.05)
                
                # Wait for thread to complete
                stream_thread.join()
                
                # Final callback with complete text
                if update_callback:
                    await update_callback(accumulated_text, None, is_final=True)
                
                return accumulated_text, None
                    
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            if update_callback:
                await update_callback(error_msg, None, is_final=True)
            return error_msg, None
    
    def _describe_parameter_changes(self, params: Dict[str, Any]) -> str:
        """Generate a human-readable description of parameter changes"""
        descriptions = []
        
        if "economic_parameters" in params:
            econ = params["economic_parameters"]
            if "oil_price_per_barrel" in econ:
                descriptions.append(f"oil price to ${econ['oil_price_per_barrel']}/bbl")
            if "discount_rate_percent" in econ:
                descriptions.append(f"discount rate to {econ['discount_rate_percent']}%")
            if "contingency_percent" in econ:
                descriptions.append(f"contingency to {econ['contingency_percent']}%")
            if "capex_budget_millions" in econ:
                descriptions.append(f"CAPEX budget to ${econ['capex_budget_millions']}MM")
        
        if "drilling_execution" in params:
            drill = params["drilling_execution"]
            if "horizontal_rigs_available" in drill:
                descriptions.append(f"available rigs to {drill['horizontal_rigs_available']}")
            if "drilling_mode" in drill:
                descriptions.append(f"drilling mode to {drill['drilling_mode']}")
            if "permit_delay_days" in drill:
                descriptions.append(f"permit delay to {drill['permit_delay_days']} days")
        
        if "risk_analysis" in params:
            risk = params["risk_analysis"]
            if "use_monte_carlo" in risk:
                if risk["use_monte_carlo"]:
                    descriptions.append("enabled Monte Carlo simulation")
                else:
                    descriptions.append("disabled Monte Carlo simulation")
            if "monte_carlo_simulations" in risk:
                descriptions.append(f"Monte Carlo simulations to {risk['monte_carlo_simulations']}")
        
        if "well_selection" in params:
            wells = params["well_selection"]
            total_wells = sum(wells.values())
            descriptions.append(f"well selection ({total_wells} total wells)")
        
        if "parameter_locks" in params:
            locks = params["parameter_locks"]
            lock_descriptions = []
            
            # Check individual parameter locks
            param_names = {
                "lock_oil_price": "oil price",
                "lock_discount_rate": "discount rate",
                "lock_contingency": "contingency",
                "lock_rigs": "rigs",
                "lock_drilling_mode": "drilling mode",
                "lock_permit_delay": "permit delay"
            }
            
            for key, name in param_names.items():
                if key in locks:
                    if locks[key]:
                        lock_descriptions.append(f"locked {name}")
                    else:
                        lock_descriptions.append(f"unlocked {name}")
            
            # Check well locks
            if "lock_wells" in locks:
                for lease, is_locked in locks["lock_wells"].items():
                    if is_locked:
                        lock_descriptions.append(f"locked {lease} wells")
                    else:
                        lock_descriptions.append(f"unlocked {lease} wells")
            
            if lock_descriptions:
                descriptions.append(f"parameter locks ({', '.join(lock_descriptions)})")
        
        if descriptions:
            return f"I've updated the following: {', '.join(descriptions)}. Click 'Optimize Development' to see the impact of these changes."
        else:
            return "No parameters were changed."
    
    def clear_history(self) -> None:
        """Clear the chat history"""
        self.context.history = []
    
    def set_model(self, model_name: str) -> None:
        """Change the model being used"""
        if model_name in ["gemini-2.5-pro", "gemini-2.5-flash"]:
            self.model_name = model_name
        else:
            raise ValueError(f"Invalid model name: {model_name}. Must be 'gemini-2.5-pro' or 'gemini-2.5-flash'")


# Singleton instance for the application
_gemini_client = None


def get_gemini_client(model_name: Optional[str] = None) -> GeminiOptimizationClient:
    """Get or create the Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiOptimizationClient(model_name=model_name)
    elif model_name and _gemini_client.model_name != model_name:
        # Update the model if requested
        _gemini_client.set_model(model_name)
    return _gemini_client