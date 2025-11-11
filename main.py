#!/usr/bin/env python3
"""
Main entry point for Oil & Gas Field Development Optimizer
"""
from components import ui_app
from nicegui import ui
import os

PROJECT_ID = os.environ.get("PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

if __name__ in {"__main__", "__mp_main__"}:    # Import will register the page routes
    ui.run(
        title="Oil & Gas Field Development Optimizer",
        port=8081,
        reload=True,
    )