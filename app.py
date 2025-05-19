"""
AI Research Assistant - Knowledge Graph Builder Application

This Streamlit application allows users to research topics and generate
knowledge graphs from the research results.

Author: AI Research Team
Version: 2.0
"""

import numpy as np
import streamlit as st
import os
import json
import time
import re
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import logging
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import io
import uuid
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our custom modules
try:
    # Core functionality
    from workflows.langgraph_router import run_research_workflow

    # Utilities
    from utils.config import set_api_keys, check_api_keys, get_api_keys, get_app_settings, update_app_settings
    from utils.ui import load_css, display_logo, create_card, display_error, display_success, display_info, display_warning
    from utils.graph_utils import create_network_graph, export_graph_to_file
except ImportError as e:
    # Fallback imports for development
    logging.warning(f"Import error: {e}. Using fallback implementations.")

    # Mock functions for development
    def run_research_workflow(query: str) -> Dict[str, Any]:
        """Mock implementation of the research workflow."""
        logger.info(f"Running mock research workflow for query: {query}")
        time.sleep(2)  # Simulate processing time

        # Generate a more realistic mock result
        topic = query.split()[0] if query else "Topic"
        subtopics = [f"{topic} Aspect {i}" for i in range(1, 4)]
        related_concepts = ["Technology", "Method", "Application"]

        # Create nodes
        nodes = [{"name": topic, "category": "Topic", "importance": 3}]

        # Add subtopic nodes
        for i, subtopic in enumerate(subtopics):
            nodes.append({
                "name": subtopic,
                "category": "Concept",
                "importance": 2
            })

        # Add related concept nodes
        for i, concept in enumerate(related_concepts):
            nodes.append({
                "name": f"{concept} {i+1}",
                "category": "Technology",
                "importance": 1
            })

        # Create edges
        edges = []

        # Connect topic to subtopics
        for subtopic in subtopics:
            edges.append({
                "source": topic,
                "target": subtopic,
                "relationship": "includes"
            })

        # Connect subtopics to related concepts
        for i, subtopic in enumerate(subtopics):
            if i < len(related_concepts):
                edges.append({
                    "source": subtopic,
                    "target": f"{related_concepts[i]} {i+1}",
                    "relationship": "relates_to"
                })

        return {
            "query": query,
            "research_results": [
                {
                    "title": f"Understanding {topic}: A Comprehensive Guide",
                    "link": "https://example.com/guide",
                    "snippet": f"This guide provides a detailed overview of {topic} and its applications."
                },
                {
                    "title": f"Latest Developments in {topic}",
                    "link": "https://example.com/latest",
                    "snippet": f"Recent advancements in {topic} have led to significant breakthroughs."
                },
                {
                    "title": f"{topic} Applications in Industry",
                    "link": "https://example.com/applications",
                    "snippet": f"How {topic} is transforming various industries and creating new opportunities."
                }
            ],
            "summarized_content": f"""
                {topic} is a complex field with multiple aspects and applications.

                Key aspects include:
                - {subtopics[0]}: Fundamental principles and core concepts
                - {subtopics[1]}: Methodologies and approaches
                - {subtopics[2]}: Practical implementations

                Related technologies:
                - {related_concepts[0]}: Enabling technologies
                - {related_concepts[1]}: Complementary methodologies
                - {related_concepts[2]}: Real-world applications

                Recent developments have shown significant progress in integrating {topic} with other fields,
                leading to innovative solutions for complex problems.
            """,
            "knowledge_graph": {
                "nodes": nodes,
                "edges": edges
            },
            "metadata": {
                "workflow_id": f"mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "start_timestamp": datetime.now().isoformat(),
                "end_timestamp": datetime.now().isoformat(),
                "query": query,
                "success": True,
                "results_count": 3
            }
        }

    def set_api_keys(serp_api_key: str, euri_api_key: str) -> bool:
        """Mock implementation of setting API keys."""
        logger.info("Setting mock API keys")
        return True

    def check_api_keys() -> bool:
        """Mock implementation of checking API keys."""
        logger.info("Checking mock API keys")
        return True

    def get_api_keys() -> Tuple[str, str]:
        """Mock implementation of getting API keys."""
        return "mock_serp_key", "mock_euri_key"

    def get_app_settings() -> Dict[str, Any]:
        """Mock implementation of getting app settings."""
        return {
            "max_results": 10,
            "default_graph_layout": "spring",
            "show_categories": True,
            "show_relationships": True
        }

    def update_app_settings(settings: Dict[str, Any]) -> bool:
        """Mock implementation of updating app settings."""
        logger.info(f"Updating mock app settings: {settings}")
        return True

    def load_css(css_file: str = "styles.css") -> None:
        """Mock implementation of loading CSS."""
        logger.info(f"Loading mock CSS from {css_file}")
        # Minimal inline CSS for development
        st.markdown("""
        <style>
        .main { padding: 2rem; }
        .card { background-color: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
        .stButton>button { background-color: #4f46e5; color: white; border-radius: 0.375rem; }
        </style>
        """, unsafe_allow_html=True)

    def display_logo() -> None:
        """Mock implementation of displaying logo."""
        st.markdown("# 🧠 AI Research Assistant", unsafe_allow_html=True)

    def create_card(title: str, content: str, icon: str = "ℹ️") -> None:
        """Mock implementation of creating a card."""
        st.markdown(f"### {icon} {title}")
        st.markdown(content)

    def display_error(message: str) -> None:
        """Mock implementation of displaying an error."""
        st.error(message)

    def display_success(message: str) -> None:
        """Mock implementation of displaying a success message."""
        st.success(message)

    def display_info(message: str) -> None:
        """Mock implementation of displaying an info message."""
        st.info(message)

    def display_warning(message: str) -> None:
        """Mock implementation of displaying a warning message."""
        st.warning(message)

    def create_network_graph(knowledge_graph: Dict[str, Any], layout: str = "spring") -> Optional[go.Figure]:
        """Mock implementation of creating a network graph."""
        # Use the original implementation for simplicity
        from app import create_network_graph as original_create_network_graph
        return original_create_network_graph(knowledge_graph)

    def export_graph_to_file(knowledge_graph: Dict[str, Any], query: str) -> Optional[str]:
        """Mock implementation of exporting a graph to a file."""
        logger.info(f"Exporting mock graph for query: {query}")
        return "mock_graph_path.svg"

# App configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state


def init_session_state() -> None:
    """Initialize the Streamlit session state with default values."""
    # Core session state variables
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # API and configuration state
    if "api_keys_set" not in st.session_state:
        st.session_state.api_keys_set = check_api_keys()

    # UI state
    if "graph_layout" not in st.session_state:
        st.session_state.graph_layout = "spring"
    if "show_categories" not in st.session_state:
        st.session_state.show_categories = True
    if "show_relationships" not in st.session_state:
        st.session_state.show_relationships = True
    if "max_results" not in st.session_state:
        st.session_state.max_results = 10

    # Load settings from config if available
    try:
        settings = get_app_settings()
        st.session_state.graph_layout = settings.get(
            "default_graph_layout", st.session_state.graph_layout)
        st.session_state.show_categories = settings.get(
            "show_categories", st.session_state.show_categories)
        st.session_state.show_relationships = settings.get(
            "show_relationships", st.session_state.show_relationships)
        st.session_state.max_results = settings.get(
            "max_results", st.session_state.max_results)
    except Exception as e:
        logger.warning(f"Failed to load settings from config: {e}")

# Agent Progress tracking


class AgentProgress:
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    MAP = "map"
    COMPLETE = "complete"

    @staticmethod
    def get_step_number(step):
        steps = {
            AgentProgress.RESEARCH: 1,
            AgentProgress.SUMMARIZE: 2,
            AgentProgress.MAP: 3,
            AgentProgress.COMPLETE: 4
        }
        return steps.get(step, 0)

    @staticmethod
    def get_step_name(step):
        names = {
            AgentProgress.RESEARCH: "Research Agent (Web Search)",
            AgentProgress.SUMMARIZE: "Summarizer Agent (Content Analysis)",
            AgentProgress.MAP: "Graph Mapper Agent (Visualization)",
            AgentProgress.COMPLETE: "Process Complete"
        }
        return names.get(step, "Unknown Step")

# Save and load results


def save_result(query, result):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"results/result_{timestamp}_{uuid.uuid4().hex[:8]}.json"

    os.makedirs("results", exist_ok=True)

    with open(filename, "w") as f:
        json.dump({
            "query": query,
            "result": result,
            "timestamp": timestamp
        }, f, indent=2)

    return filename


def load_result(filename):
    with open(filename, "r") as f:
        return json.load(f)

# Display the network graph using Plotly


@st.cache_data
def create_network_graph(knowledge_graph, layout="spring"):
    """
    Create a network graph visualization from knowledge graph data.

    Args:
        knowledge_graph (Dict[str, Any]): The knowledge graph data with nodes and edges
        layout (str): The layout algorithm to use (spring, circular, kamada_kawai)

    Returns:
        Optional[go.Figure]: A Plotly figure object or None if graph creation fails
    """
    if not knowledge_graph or not knowledge_graph.get("nodes") or not knowledge_graph.get("edges"):
        return None

    # Create a networkx graph
    G = nx.DiGraph()

    # Node category colors
    category_colors = {
        "Topic": "#4299E1",
        "Concept": "#48BB78",
        "Technology": "#805AD5",
        "Phase": "#F6AD55",
        "Tool": "#FC8181",
        "TimeEstimate": "#F6E05E",
        "Outcome": "#B794F4",
        "default": "#CBD5E0"
    }

    # Edge relationship styles
    relationship_colors = {
        "requires": "#E53E3E",
        "leads_to": "#38A169",
        "includes": "#3182CE",
        "part_of": "#805AD5",
        "followed_by": "#DD6B20",
        "mentioned_together": "#718096",
        "relates_to": "#A0AEC0",
        "default": "#A0AEC0"
    }

    # Add nodes with attributes
    for node in knowledge_graph["nodes"]:
        category = node.get("category", "default")

        G.add_node(
            node["name"],
            category=category,
            importance=node.get("importance", 1),
            color=node.get("color", category_colors.get(
                category, category_colors["default"]))
        )

    # Add edges with attributes
    for edge in knowledge_graph["edges"]:
        relationship = edge.get("relationship", "default")

        G.add_edge(
            edge["source"],
            edge["target"],
            relationship=relationship,
            weight=edge.get("weight", 1),
            color=edge.get("color", relationship_colors.get(
                relationship, relationship_colors["default"]))
        )

    # Choose layout algorithm based on the layout parameter
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # default to spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Create edge trace
    edge_traces = []
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        weight = edge[2].get('weight', 1)
        color = edge[2].get('color', '#9ca3af')

        # Create arrows using shapes
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight*1.5, color=color),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_traces = []

    # Group nodes by category for the legend
    categories = set(nx.get_node_attributes(G, 'category').values())

    for category in categories:
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G.nodes(data=True):
            if node[1].get('category') == category:
                x, y = pos[node[0]]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node[0])
                # Scale node size by importance (1-3)
                node_size.append(node[1].get('importance', 1) * 20)
                node_color.append(node[1].get('color', '#4f46e5'))

        if not node_x:  # Skip empty categories
            continue

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(
                family="Arial",
                size=12,
                color="#1f2937"
            ),
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='#ffffff'),
            ),
            hoverinfo='text',
            hovertext=node_text,
            name=category
        )
        node_traces.append(node_trace)

    # Create figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=5, l=5, r=5, t=5),
            plot_bgcolor='rgba(255,255,255,0.8)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                x=0,
                y=1,
                title="Node Categories",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#e5e7eb",
                borderwidth=1
            ),
            height=600,
        )
    )

    # Add annotations for relationships
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        relationship = edge[2].get('relationship', 'relates_to')

        # Only show label if it's not a default relationship
        if relationship != 'default' and relationship != 'relates_to':
            # Calculate midpoint of the edge for label placement
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            # Add small offset to prevent overlap with the edge
            offset_x = -(y1 - y0) * 0.1
            offset_y = (x1 - x0) * 0.1

            fig.add_annotation(
                x=mid_x + offset_x,
                y=mid_y + offset_y,
                text=relationship.replace('_', ' '),
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=9,
                    color="#6b7280"
                ),
                bgcolor="rgba(255, 255, 255, 0.7)",
                borderpad=2
            )

    # Add arrows for directed edges
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Calculate the angle of the edge
        dx, dy = x1 - x0, y1 - y0
        angle = np.arctan2(dy, dx)

        # Calculate the position of the arrowhead
        # Place it at 90% of the edge length
        scalar = 0.9
        end_x = x0 + scalar * dx
        end_y = y0 + scalar * dy

        # Arrow dimensions
        arrow_length = 0.1
        arrow_width = 0.05

        # Calculate arrow points
        p1_x = end_x - arrow_length * \
            np.cos(angle) + arrow_width * np.sin(angle)
        p1_y = end_y - arrow_length * \
            np.sin(angle) - arrow_width * np.cos(angle)
        p2_x = end_x - arrow_length * \
            np.cos(angle) - arrow_width * np.sin(angle)
        p2_y = end_y - arrow_length * \
            np.sin(angle) + arrow_width * np.cos(angle)

        # Add the arrow shape
        arrow_color = edge[2].get('color', '#9ca3af')
        fig.add_shape(
            type="path",
            path=f"M {end_x} {end_y} L {p1_x} {p1_y} L {p2_x} {p2_y} Z",
            fillcolor=arrow_color,
            line=dict(color=arrow_color),
            opacity=0.8
        )

    return fig

# Display search results in a nice format


def display_search_results(results):
    if not results:
        st.info("No search results available.")
        return

    for i, result in enumerate(results):
        with st.container():
            st.markdown(
                f"""
                <div class="search-result">
                    <div class="search-result-title">{result.get('title', 'Untitled')}</div>
                    <div class="search-result-link">{result.get('link', '#')}</div>
                    <div class="search-result-snippet">{result.get('snippet', 'No description available.')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Display summary with nice formatting


def display_summary(summary, query=None):
    """
    Display a detailed summary with proper formatting.

    Args:
        summary (str): The summary content to display
        query (str, optional): The original query for context
    """
    if not summary:
        st.info("No summary available.")
        return

    # Format the summary with proper line breaks
    formatted_summary = summary.replace('\n', '<br>')

    # Create a more structured display with the query context
    if query:
        st.markdown(f"### 🔍 Topic: {query}")
        st.markdown("---")

    # Display the summary in a nicely formatted container
    st.markdown(
        f"""
        <div class="summary-content">
            {formatted_summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add a section for key insights if the summary is long enough
    if len(summary) > 200:
        try:
            # Extract key points from the summary
            lines = summary.strip().split('\n')
            key_points = [line.strip() for line in lines if line.strip() and line.strip()[
                0] in ['-', '*'] or ':' in line]

            if key_points:
                st.markdown("### 💡 Key Insights")
                for point in key_points[:5]:  # Limit to top 5 points
                    st.markdown(f"- {point.lstrip('-* ')}")
        except Exception as e:
            logger.warning(f"Error extracting key points: {e}")
            # Continue without key points if there's an error

# Display detailed topic explanation with graph and summary


def display_topic_explanation(result, query):
    """
    Display a comprehensive explanation of the topic with graph visualization.

    Args:
        result (Dict[str, Any]): The research result data
        query (str): The original research query
    """
    if not result:
        st.info("No research results available.")
        return

    # Create a two-column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Display the knowledge graph
        if result.get("knowledge_graph"):
            st.markdown("### 📊 Knowledge Graph Visualization")
            fig = create_network_graph(
                result["knowledge_graph"],
                layout=st.session_state.graph_layout
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Add a brief explanation of the graph
                st.markdown("""
                **Understanding the Graph:**
                - **Nodes** represent key concepts, topics, and entities
                - **Edges** show relationships between nodes
                - **Colors** indicate different categories
                - **Size** reflects importance or relevance
                """)
            else:
                st.info("Unable to create knowledge graph visualization.")
        else:
            st.info("No knowledge graph available.")

    with col2:
        # Display the detailed explanation
        st.markdown("### 📝 Topic Explanation")

        # Show the query as a title
        st.markdown(f"**Research Question:**")
        st.markdown(
            f"<div style='background-color: #f0f9ff; padding: 10px; border-radius: 5px; border-left: 4px solid #3b82f6;'>{query}</div>", unsafe_allow_html=True)

        # Display metadata about the research
        if result.get("metadata"):
            metadata = result.get("metadata", {})
            st.markdown("**Research Overview:**")

            # Format timestamps if available
            start_time = metadata.get("start_timestamp", "N/A")
            if start_time != "N/A":
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass

            # Show research stats
            st.markdown(f"""
            - **Sources analyzed:** {metadata.get("results_count", 0)}
            - **Research conducted:** {start_time}
            - **Confidence level:** {'High' if metadata.get("success", False) else 'Medium'}
            """)

        # Display key concepts from the knowledge graph
        if result.get("knowledge_graph") and result["knowledge_graph"].get("nodes"):
            st.markdown("**Key Concepts:**")

            # Extract and sort nodes by importance
            nodes = result["knowledge_graph"]["nodes"]
            important_nodes = sorted(
                nodes,
                key=lambda x: x.get("importance", 1),
                reverse=True
            )[:5]  # Top 5 most important nodes

            for node in important_nodes:
                category = node.get("category", "Concept")
                name = node.get("name", "")
                st.markdown(f"- **{name}** ({category})")

    # Display the full summary below the columns
    st.markdown("### 📚 Detailed Summary")
    if result.get("summarized_content"):
        # Use our enhanced display_summary function without repeating the query
        display_summary(result["summarized_content"])
    else:
        st.info("No detailed summary available.")

    # Add a section for related topics if available
    if result.get("knowledge_graph") and result["knowledge_graph"].get("nodes"):
        # Find nodes that might be related topics (not the main topic)
        nodes = result["knowledge_graph"]["nodes"]
        related_topics = [
            node["name"] for node in nodes
            if node.get("category") in ["Topic", "Concept", "Technology"]
            and node.get("importance", 1) < 3  # Not the main topic
        ][:5]  # Limit to 5 related topics

        if related_topics:
            st.markdown("### 🔗 Related Topics")
            cols = st.columns(min(len(related_topics), 3))
            for i, topic in enumerate(related_topics):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='background-color: #f5f3ff; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;'>
                        <strong>{topic}</strong>
                    </div>
                    """, unsafe_allow_html=True)

# Display workflow steps with status


def display_workflow_steps(current_step, errors=None):
    steps = [
        AgentProgress.RESEARCH,
        AgentProgress.SUMMARIZE,
        AgentProgress.MAP,
        AgentProgress.COMPLETE
    ]

    for step in steps:
        step_number = AgentProgress.get_step_number(step)
        step_name = AgentProgress.get_step_name(step)

        if step == current_step:
            status = "in-progress"
            icon = "🔄"
        elif AgentProgress.get_step_number(current_step) > step_number:
            status = "complete"
            icon = "✅"
        else:
            status = "pending"
            icon = "⏳"

        # Check for errors in this step
        has_error = False
        error_message = ""
        if errors:
            for error in errors:
                if error.get("agent") == step:
                    has_error = True
                    status = "error"
                    icon = "❌"
                    error_message = error.get("error", "Unknown error")

        error_html = f'<div class="error-message">{error_message}</div>' if has_error else ''

        st.markdown(
            f"""
            <div class="agent-step {status}">
                <div class="agent-step-header">
                    <div class="agent-step-title">{icon} Step {step_number}: {step_name}</div>
                    <div class="agent-step-status">{status.replace('-', ' ').title()}</div>
                </div>
                <div class="agent-step-content">
                    {error_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Export graph to file


def export_graph_to_file(knowledge_graph, query):
    """
    Export the graph data to a file that can be used with Graphviz.

    Args:
        knowledge_graph: The knowledge graph data
        query: The search query used to generate the graph

    Returns:
        The path to the exported file
    """
    try:
        # Import the export function
        try:
            from utils.graphviz_exporter import export_to_svg
        except ImportError:
            # Fallback to basic export
            from graphviz import Digraph

            def sanitize(text):
                clean = str(text).replace('"', "'").replace("\n", " ").strip()
                clean = re.sub(r"[<>\\]", "", clean)
                return clean[:120] + "..." if len(clean) > 120 else clean

            def export_to_svg(graph_data, file_name="output_graph"):
                os.makedirs("data/outputs", exist_ok=True)

                dot = Digraph(format="svg")
                dot.attr(bgcolor="white")
                dot.attr("graph", rankdir="TB", ranksep="2.5", nodesep="1.5")
                dot.attr("node", shape="box", style="filled,setlinewidth(3)", width="2", height="1", fontsize="24",
                         fillcolor="#FFF8DC", fontname="Helvetica-Bold", color="#FFB300", fontcolor="#000000")
                dot.attr("edge", color="#999999",
                         arrowsize="1.4", penwidth="2.0")

                seen = set()
                for node in graph_data["nodes"]:
                    name = sanitize(node["name"])
                    if name and name not in seen:
                        dot.node(name)
                        seen.add(name)

                for edge in graph_data["edges"]:
                    source = sanitize(edge["source"])
                    target = sanitize(edge["target"])
                    if source and target:
                        dot.edge(source, target)

                output_path = os.path.join("data/outputs", file_name)
                dot.render(output_path, format="svg", cleanup=True)
                return f"{output_path}.svg"

        # Create a safe filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"graph_{safe_query}_{timestamp}"

        # Export the graph
        return export_to_svg(knowledge_graph, file_name=file_name)

    except Exception as e:
        logging.exception(f"Error exporting graph: {e}")
        return None

# Generate shareable URL for results


def get_shareable_link(result_path):
    """
    Generate a shareable link for the results.
    This is a simplified example - in production, you'd implement proper storage/sharing.

    Args:
        result_path: Path to the result file

    Returns:
        A shareable URL (simulated in this example)
    """
    # In a real application, you would upload the file to cloud storage
    # or generate a proper shareable URL

    # For now, we'll just return a simulated link
    return f"https://ai-research-assistant.example.com/shared/{os.path.basename(result_path)}"

# Main app function


def main() -> None:
    """Main application function."""
    # Initialize session state
    init_session_state()

    # Load custom CSS
    load_css()

    # Sidebar
    with st.sidebar:
        display_logo()
        st.markdown(
            "<p style='color: #94a3b8;'>Powered by LangGraph</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3 style='color: white;'>⚙️ Settings</h3>",
                    unsafe_allow_html=True)

        # API keys input
        with st.expander("API Keys", expanded=not st.session_state.api_keys_set):
            serp_api_key = st.text_input(
                "SERP API Key",
                type="password",
                value=os.environ.get("SERP_API_KEY", ""),
                help="Enter your SERP API key for web search functionality"
            )
            euri_api_key = st.text_input(
                "EURI API Key",
                type="password",
                value=os.environ.get("EURI_API_KEY", ""),
                help="Enter your EURI API key for enhanced research capabilities"
            )

            if st.button("Save API Keys", key="save_api_keys"):
                try:
                    # Set API keys in environment and configuration
                    success = set_api_keys(serp_api_key, euri_api_key)
                    if success:
                        st.session_state.api_keys_set = check_api_keys()
                        display_success("API keys saved successfully!")
                    else:
                        display_error(
                            "Failed to save API keys. Please try again.")
                except Exception as e:
                    logger.exception("Error saving API keys")
                    display_error(f"Error saving API keys: {str(e)}")

        # Advanced settings
        with st.expander("Advanced Settings"):
            # Use session state for persistence
            max_results = st.slider(
                "Max Search Results",
                min_value=5,
                max_value=50,
                value=st.session_state.max_results,
                help="Maximum number of search results to display"
            )

            graph_layout = st.selectbox(
                "Graph Layout",
                options=["spring", "circular", "kamada_kawai"],
                format_func=lambda x: x.replace("_", " ").title(),
                index=["spring", "circular", "kamada_kawai"].index(
                    st.session_state.graph_layout)
                if st.session_state.graph_layout in ["spring", "circular", "kamada_kawai"] else 0,
                help="Select the layout algorithm for the knowledge graph"
            )

            show_categories = st.checkbox(
                "Show Node Categories",
                value=st.session_state.show_categories,
                help="Display node categories in the graph legend"
            )

            show_relationships = st.checkbox(
                "Show Relationship Labels",
                value=st.session_state.show_relationships,
                help="Display relationship labels on graph edges"
            )

            # Save settings if changed
            if (max_results != st.session_state.max_results or
                graph_layout != st.session_state.graph_layout or
                show_categories != st.session_state.show_categories or
                    show_relationships != st.session_state.show_relationships):

                st.session_state.max_results = max_results
                st.session_state.graph_layout = graph_layout
                st.session_state.show_categories = show_categories
                st.session_state.show_relationships = show_relationships

                # Update settings in config
                try:
                    update_app_settings({
                        "max_results": max_results,
                        "default_graph_layout": graph_layout,
                        "show_categories": show_categories,
                        "show_relationships": show_relationships
                    })
                except Exception as e:
                    logger.warning(f"Failed to save settings: {e}")

        # Search history
        st.markdown("<h3 style='color: white;'>📜 Search History</h3>",
                    unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown(
                "<p style='color: #94a3b8;'>No previous searches</p>", unsafe_allow_html=True)
        else:
            # Display history with timestamps
            for i, item in enumerate(st.session_state.history):
                timestamp = datetime.fromisoformat(
                    item.get("timestamp", datetime.now().isoformat()))
                formatted_time = timestamp.strftime("%m/%d/%Y %H:%M")
                query_text = item['query'][:25] + \
                    "..." if len(item['query']) > 25 else item['query']

                if st.button(f"{query_text} ({formatted_time})", key=f"history_{i}"):
                    st.session_state.current_query = item["query"]
                    st.session_state.current_results = item["results"]
                    st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.8rem;'>© 2025 AI Research Assistant</p>",
            unsafe_allow_html=True
        )

    # Main content
    st.markdown(
        """
        <div class="header-container">
            <h1 class="app-header">🧠 AI Research Agent</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Search input
    with st.container():
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])

        with col1:
            query = st.text_input(
                "Enter your research question or topic",
                value=st.session_state.current_query if st.session_state.current_query else "",
                placeholder="e.g., What are the latest developments in quantum computing?",
                key="query_input",
                help="Enter a research question, topic, or concept you want to explore"
            )

        with col2:
            search_button = st.button(
                "Research",
                disabled=st.session_state.processing or not st.session_state.api_keys_set,
                key="search_button",
                help="Start the research process"
            )

        if not st.session_state.api_keys_set:
            display_warning(
                "Please set your API keys in the sidebar to use the research assistant.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Process query
    if search_button and query:
        st.session_state.processing = True
        st.session_state.current_query = query
        st.session_state.current_step = AgentProgress.RESEARCH

        with st.spinner("Processing your query..."):
            try:
                # Update UI to show progress
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    display_workflow_steps(st.session_state.current_step)

                # Run research agent
                st.session_state.current_step = AgentProgress.RESEARCH
                progress_placeholder.empty()
                with progress_placeholder.container():
                    display_workflow_steps(st.session_state.current_step)

                # Execute the workflow
                result = run_research_workflow(query)

                # Update UI based on result
                if result.get("errors"):
                    st.session_state.current_step = list(
                        set([error.get("agent") for error in result.get("errors")]))[0]
                    progress_placeholder.empty()
                    with progress_placeholder.container():
                        display_workflow_steps(
                            st.session_state.current_step, result.get("errors"))
                else:
                    # Workflow completed successfully
                    st.session_state.current_step = AgentProgress.COMPLETE
                    progress_placeholder.empty()
                    with progress_placeholder.container():
                        display_workflow_steps(st.session_state.current_step)

                # Store the results
                st.session_state.current_results = result

                # Save to history
                st.session_state.history.insert(0, {
                    "query": query,
                    "results": result,
                    "timestamp": datetime.now().isoformat()
                })

                # Keep history to a reasonable size
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[:10]

                # Save result to file
                save_result(query, result)

            except Exception as e:
                display_error(f"An error occurred: {str(e)}")
                logger.exception("Error processing query")

            finally:
                st.session_state.processing = False
                st.rerun()

    # Display results if available
    if st.session_state.current_results:
        result = st.session_state.current_results

        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)

            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["� Topic Explanation", "�📊 Knowledge Graph", " Research Results", "ℹ️ Details"])

            with tab1:
                # Display the comprehensive topic explanation with graph and summary
                display_topic_explanation(result, st.session_state.current_query)

                # Add a button to copy the summary
                if result.get("summarized_content"):
                    if st.button("Copy Summary to Clipboard", key="copy_summary"):
                        st.code(result["summarized_content"], language=None)
                        display_success("Summary content copied to clipboard. Use Ctrl+C to copy.")

            with tab2:
                if result.get("knowledge_graph"):
                    # Use the cached graph creation function with the selected layout
                    fig = create_network_graph(
                        result["knowledge_graph"],
                        layout=st.session_state.graph_layout
                    )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # Export options
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("Export Graph", key="export_graph"):
                                graph_path = export_graph_to_file(
                                    result["knowledge_graph"],
                                    result.get("query", "research")
                                )

                                if graph_path:
                                    display_success(f"Graph exported successfully to {graph_path}")

                                    # Provide download link
                                    with open(graph_path, "rb") as file:
                                        svg_content = file.read()
                                        st.download_button(
                                            label="Download SVG",
                                            data=svg_content,
                                            file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d')}.svg",
                                            mime="image/svg+xml",
                                            key="download_svg"
                                        )
                                else:
                                    display_error("Failed to export graph")

                        with col2:
                            # Add layout selection in the graph view as well
                            layout_options = ["spring", "circular", "kamada_kawai"]
                            selected_layout = st.selectbox(
                                "Change Layout",
                                options=layout_options,
                                format_func=lambda x: x.replace("_", " ").title(),
                                index=layout_options.index(st.session_state.graph_layout)
                                      if st.session_state.graph_layout in layout_options else 0,
                                key="change_layout"
                            )

                            if selected_layout != st.session_state.graph_layout:
                                st.session_state.graph_layout = selected_layout
                                st.rerun()
                    else:
                        display_info("Unable to create knowledge graph visualization.")
                else:
                    display_info("No knowledge graph available.")

            with tab3:
                if result.get("research_results"):
                    display_search_results(result["research_results"])
                else:
                    display_info("No research results available.")

            with tab4:
                st.subheader("Metadata")
                metadata = result.get("metadata", {})

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Query**")
                    st.write(metadata.get("query", "N/A"))

                    st.markdown("**Workflow ID**")
                    st.write(metadata.get("workflow_id", "N/A"))

                    st.markdown("**Success**")
                    st.write("✅ Yes" if metadata.get(
                        "success", False) else "❌ No")

                with col2:
                    st.markdown("**Start Time**")
                    start_time = metadata.get("start_timestamp", "N/A")
                    if start_time != "N/A":
                        try:
                            # Format the timestamp if it's valid
                            start_dt = datetime.fromisoformat(start_time)
                            start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            pass
                    st.write(start_time)

                    st.markdown("**End Time**")
                    end_time = metadata.get("end_timestamp", "N/A")
                    if end_time != "N/A":
                        try:
                            # Format the timestamp if it's valid
                            end_dt = datetime.fromisoformat(end_time)
                            end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            pass
                    st.write(end_time)

                    if "results_count" in metadata:
                        st.markdown("**Results Count**")
                        st.write(metadata.get("results_count", 0))

                if result.get("errors"):
                    st.subheader("Errors")
                    for error in result["errors"]:
                        error_time = error.get("timestamp", "Unknown")
                        if error_time != "Unknown":
                            try:
                                # Format the timestamp if it's valid
                                error_dt = datetime.fromisoformat(error_time)
                                error_time = error_dt.strftime(
                                    "%Y-%m-%d %H:%M:%S")
                            except (ValueError, TypeError):
                                pass

                        display_error(
                            f"""
                            **Agent:** {error.get("agent", "Unknown")}\n
                            **Error:** {error.get("error", "Unknown error")}\n
                            **Time:** {error_time}
                            """
                        )

                # Share results
                st.subheader("Share Results")
                if st.button("Generate Shareable Link", key="share_link"):
                    # Save current result to file
                    result_path = save_result(query, result)
                    sharable_link = get_shareable_link(result_path)
                    display_success("Shareable link generated!")
                    st.code(sharable_link)

            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #6b7280; font-size: 0.8rem;">
            Powered by LangGraph, Streamlit and AI Research Agents
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    try:
        # Ensure required directories exist
        os.makedirs("results", exist_ok=True)
        os.makedirs("data/outputs", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("config", exist_ok=True)

        # Check if CSS file exists, if not create a minimal version
        css_path = os.path.join("static", "styles.css")
        if not os.path.exists(css_path):
            logger.info(
                f"CSS file not found at {css_path}, creating a minimal version")
            try:
                from utils.ui import load_css
            except ImportError:
                # If utils.ui is not available, use the mock implementation
                logger.warning("utils.ui module not found, using minimal CSS")

        # Run the app
        logger.info("Starting AI Research Assistant application")
        main()

    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        st.error(f"Application startup error: {str(e)}")
        st.info(
            "Please check the logs for more information and ensure all required directories exist.")
