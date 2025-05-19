"""
UI utilities for the Knowledge Graph Builder application.

This module provides functions for managing UI components and styling.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional

# Constants
STATIC_DIR = Path("static")


def load_css(css_file: str = "styles.css") -> None:
    """
    Load CSS from a file and inject it into the Streamlit app.
    
    Args:
        css_file (str): The CSS file name in the static directory
    """
    css_path = STATIC_DIR / css_file
    
    # Ensure the file exists
    if not css_path.exists():
        st.warning(f"CSS file not found: {css_path}")
        return
    
    # Read the CSS file
    with open(css_path, "r") as f:
        css = f.read()
    
    # Inject the CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def display_logo(logo_path: Optional[str] = None) -> None:
    """
    Display the application logo.
    
    Args:
        logo_path (Optional[str]): Path to the logo image file
    """
    if logo_path and os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        # Display a text logo if image is not available
        st.markdown(
            """
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; font-weight: 700; color: #4f46e5;">🧠</div>
                <div style="font-size: 1.5rem; font-weight: 600; margin-left: 0.5rem;">AI Research Assistant</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_card(title: str, content: str, icon: str = "ℹ️") -> None:
    """
    Create a styled card with title and content.
    
    Args:
        title (str): The card title
        content (str): The card content (can include HTML)
        icon (str): An emoji icon for the card
    """
    st.markdown(
        f"""
        <div class="modern-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div>
                <div class="card-title">{title}</div>
            </div>
            <div>{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_error(message: str) -> None:
    """
    Display an error message in a styled container.
    
    Args:
        message (str): The error message to display
    """
    st.markdown(
        f"""
        <div class="error-message">
            <strong>Error:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_success(message: str) -> None:
    """
    Display a success message in a styled container.
    
    Args:
        message (str): The success message to display
    """
    st.markdown(
        f"""
        <div style="background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.375rem; color: #065f46; margin-bottom: 1rem;">
            <strong>Success:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_info(message: str) -> None:
    """
    Display an info message in a styled container.
    
    Args:
        message (str): The info message to display
    """
    st.markdown(
        f"""
        <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 0.375rem; color: #1e40af; margin-bottom: 1rem;">
            <strong>Info:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_warning(message: str) -> None:
    """
    Display a warning message in a styled container.
    
    Args:
        message (str): The warning message to display
    """
    st.markdown(
        f"""
        <div style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 0.375rem; color: #92400e; margin-bottom: 1rem;">
            <strong>Warning:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )
