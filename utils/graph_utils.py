"""
Graph utilities for the Knowledge Graph Builder application.

This module provides functions for creating and manipulating knowledge graphs.
"""

import os
import re
import logging
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Node category colors
DEFAULT_CATEGORY_COLORS = {
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
DEFAULT_RELATIONSHIP_COLORS = {
    "requires": "#E53E3E",
    "leads_to": "#38A169",
    "includes": "#3182CE",
    "part_of": "#805AD5",
    "followed_by": "#DD6B20",
    "mentioned_together": "#718096",
    "relates_to": "#A0AEC0",
    "default": "#A0AEC0"
}


@st.cache_data
def create_network_graph(knowledge_graph: Dict[str, Any], layout: str = "spring") -> Optional[go.Figure]:
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
    
    try:
        # Create a networkx graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in knowledge_graph["nodes"]:
            category = node.get("category", "default")
            
            G.add_node(
                node["name"],
                category=category,
                importance=node.get("importance", 1),
                color=node.get("color", DEFAULT_CATEGORY_COLORS.get(
                    category, DEFAULT_CATEGORY_COLORS["default"]))
            )
        
        # Add edges with attributes
        for edge in knowledge_graph["edges"]:
            relationship = edge.get("relationship", "default")
            
            G.add_edge(
                edge["source"],
                edge["target"],
                relationship=relationship,
                weight=edge.get("weight", 1),
                color=edge.get("color", DEFAULT_RELATIONSHIP_COLORS.get(
                    relationship, DEFAULT_RELATIONSHIP_COLORS["default"]))
            )
        
        # Choose layout algorithm
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
            p1_x = end_x - arrow_length * np.cos(angle) + arrow_width * np.sin(angle)
            p1_y = end_y - arrow_length * np.sin(angle) - arrow_width * np.cos(angle)
            p2_x = end_x - arrow_length * np.cos(angle) - arrow_width * np.sin(angle)
            p2_y = end_y - arrow_length * np.sin(angle) + arrow_width * np.cos(angle)
            
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
    
    except Exception as e:
        logger.exception(f"Error creating network graph: {e}")
        return None


def export_graph_to_file(knowledge_graph: Dict[str, Any], query: str) -> Optional[str]:
    """
    Export the graph data to a file that can be used with Graphviz.
    
    Args:
        knowledge_graph (Dict[str, Any]): The knowledge graph data
        query (str): The search query used to generate the graph
        
    Returns:
        Optional[str]: The path to the exported file or None if export fails
    """
    try:
        # Import graphviz
        try:
            from graphviz import Digraph
        except ImportError:
            logger.error("Graphviz not installed. Cannot export graph.")
            return None
        
        def sanitize(text):
            """Sanitize text for use in Graphviz."""
            clean = str(text).replace('"', "'").replace("\n", " ").strip()
            clean = re.sub(r"[<>\\]", "", clean)
            return clean[:120] + "..." if len(clean) > 120 else clean
        
        # Create a safe filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"graph_{safe_query}_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs("data/outputs", exist_ok=True)
        output_path = os.path.join("data/outputs", file_name)
        
        # Create the graph
        dot = Digraph(format="svg")
        dot.attr(bgcolor="white")
        dot.attr("graph", rankdir="TB", ranksep="2.5", nodesep="1.5")
        dot.attr("node", shape="box", style="filled,setlinewidth(3)", width="2", height="1", fontsize="24",
                 fillcolor="#FFF8DC", fontname="Helvetica-Bold", color="#FFB300", fontcolor="#000000")
        dot.attr("edge", color="#999999", arrowsize="1.4", penwidth="2.0")
        
        # Add nodes
        seen = set()
        for node in knowledge_graph["nodes"]:
            name = sanitize(node["name"])
            if name and name not in seen:
                dot.node(name)
                seen.add(name)
        
        # Add edges
        for edge in knowledge_graph["edges"]:
            source = sanitize(edge["source"])
            target = sanitize(edge["target"])
            if source and target:
                dot.edge(source, target)
        
        # Render the graph
        dot.render(output_path, format="svg", cleanup=True)
        return f"{output_path}.svg"
    
    except Exception as e:
        logger.exception(f"Error exporting graph: {e}")
        return None
