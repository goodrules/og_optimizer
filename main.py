#!/usr/bin/env python3
"""
Main entry point for Oil & Gas Field Development Optimizer
"""
from components import ui_app
from nicegui import ui

if __name__ == "__main__":
    # Import will register the page routes
    ui.run(
        title="Oil & Gas Field Development Optimizer",
        port=8080,
        reload=False,
        show=True
    )