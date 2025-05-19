import os
import re
import json
import math
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from graphviz import Digraph, Graph

# Configure Graphviz path
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin" + \
    os.pathsep + os.environ["PATH"]

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for styling
NODE_CATEGORIES = {
    "Topic": {"color": "#4299E1", "shape": "box", "fontcolor": "#000000"},
    "Concept": {"color": "#48BB78", "shape": "ellipse", "fontcolor": "#000000"},
    "Technology": {"color": "#805AD5", "shape": "box", "fontcolor": "#FFFFFF"},
    "Phase": {"color": "#F6AD55", "shape": "box", "fontcolor": "#000000"},
    "Tool": {"color": "#FC8181", "shape": "component", "fontcolor": "#000000"},
    "TimeEstimate": {"color": "#F6E05E", "shape": "box", "fontcolor": "#000000"},
    "Outcome": {"color": "#B794F4", "shape": "ellipse", "fontcolor": "#000000"},
    "default": {"color": "#CBD5E0", "shape": "box", "fontcolor": "#000000"}
}

RELATIONSHIP_STYLES = {
    "requires": {"color": "#E53E3E", "style": "solid", "penwidth": "2.0"},
    "leads_to": {"color": "#38A169", "style": "solid", "penwidth": "2.0"},
    "includes": {"color": "#3182CE", "style": "solid", "penwidth": "2.0"},
    "part_of": {"color": "#805AD5", "style": "dashed", "penwidth": "1.5"},
    "followed_by": {"color": "#DD6B20", "style": "solid", "penwidth": "2.0"},
    "mentioned_together": {"color": "#718096", "style": "dotted", "penwidth": "1.0"},
    "relates_to": {"color": "#A0AEC0", "style": "solid", "penwidth": "1.5"},
    "default": {"color": "#A0AEC0", "style": "solid", "penwidth": "1.5"}
}


class GraphExporter:
    """
    Comprehensive graph exporter for knowledge graphs with extensive styling and layout options.

    Features:
    - Multiple output formats (SVG, PNG, PDF, DOT)
    - Advanced node and edge styling based on properties
    - Clustering and subgraph support
    - Layout algorithm selection
    - Legend generation
    - Graph metrics calculation
    - Hierarchical node organization
    - Support for large graphs with pagination
    """

    def __init__(self, output_dir: str = "data/outputs"):
        """
        Initialize the graph exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.current_graph = None
        self.current_data = None

    def sanitize(self, text: Any, max_length: int = 120) -> str:
        """
        Sanitize text for Graphviz compatibility.

        Args:
            text: Input text to sanitize
            max_length: Maximum length before truncation

        Returns:
            Sanitized text string
        """
        if text is None:
            return "None"

        clean = str(text).replace('"', "'").replace("\n", " ").strip()
        clean = re.sub(r"[<>\\]", "", clean)

        if len(clean) > max_length:
            return clean[:max_length] + "..."
        return clean

    def get_node_id(self, node_name: str) -> str:
        """
        Create a consistent node ID for Graphviz.

        Args:
            node_name: The name of the node

        Returns:
            A sanitized, unique node ID
        """
        # Use a hash for node names to ensure uniqueness and valid Graphviz IDs
        name_hash = hashlib.md5(node_name.encode()).hexdigest()[:8]
        return f"node_{name_hash}"

    def create_graph(self,
                     directed: bool = True,
                     layout: str = "dot",
                     concentrate: bool = True,
                     rankdir: str = "TB",
                     ratio: str = "compress",
                     fontname: str = "Helvetica",
                     bgcolor: str = "white") -> Union[Digraph, Graph]:
        """
        Create a new Graphviz graph with specified properties.

        Args:
            directed: Whether to create a directed graph
            layout: Layout algorithm to use (dot, neato, fdp, sfdp, twopi, circo)
            concentrate: Whether to merge edges
            rankdir: Direction of graph layout (TB, LR, BT, RL)
            ratio: Aspect ratio (compress, fill, auto)
            fontname: Default font for the graph
            bgcolor: Background color

        Returns:
            A configured Graphviz graph object
        """
        graph_class = Digraph if directed else Graph
        graph = graph_class(engine=layout, format="svg")

        # Set graph attributes
        graph.attr(bgcolor=bgcolor)
        graph.attr("graph",
                   rankdir=rankdir,
                   concentrate="true" if concentrate else "false",
                   ratio=ratio,
                   ranksep="1.5",
                   nodesep="1.0",
                   fontname=fontname)

        # Set default node and edge attributes
        graph.attr("node",
                   fontname=fontname,
                   fontsize="14",
                   margin="0.3,0.1")

        graph.attr("edge",
                   fontname=fontname,
                   fontsize="12",
                   arrowsize="0.8",
                   fontcolor="#4A5568")

        return graph

    def style_node(self, graph, node_id, node_data):
        """
        Apply styling to a node based on its properties.

        Args:
            graph: The Graphviz graph object
            node_id: The ID of the node
            node_data: The data associated with this node
        """
        # Get category styling or use default
        category = node_data.get("category", "default")
        style_data = NODE_CATEGORIES.get(category, NODE_CATEGORIES["default"])

        # Calculate node size based on importance
        importance = node_data.get("importance", 1)
        width = 2 + (importance * 0.5)
        height = 1 + (importance * 0.3)

        # Apply custom color if provided
        color = node_data.get("color", style_data["color"])
        shape = node_data.get("shape", style_data["shape"])
        fontcolor = node_data.get("fontcolor", style_data["fontcolor"])

        # Add label with multiline HTML-like formatting if needed
        name = self.sanitize(node_data["name"])
        label = f"{name}"

        # For detailed nodes, add category and other info
        if node_data.get("show_details", True) and len(name) < 30:
            if "description" in node_data and node_data["description"]:
                desc = self.sanitize(node_data["description"], max_length=60)
                label = f"{name}\\n{desc}"

        # Apply styling
        graph.node(node_id,
                   label=label,
                   shape=shape,
                   style="filled,setlinewidth(2)",
                   penwidth=str(importance),
                   fillcolor=color,
                   color=self.adjust_color(color, -20),  # Darker outline
                   fontcolor=fontcolor,
                   width=str(width),
                   height=str(height))

    def style_edge(self, graph, source_id, target_id, edge_data):
        """
        Apply styling to an edge based on its properties.

        Args:
            graph: The Graphviz graph object
            source_id: The ID of the source node
            target_id: The ID of the target node
            edge_data: The data associated with this edge
        """
        # Get relationship styling or use default
        relationship = edge_data.get("relationship", "default")
        style_data = RELATIONSHIP_STYLES.get(
            relationship, RELATIONSHIP_STYLES["default"])

        # Apply custom styling if provided
        color = edge_data.get("color", style_data["color"])
        style = edge_data.get("style", style_data["style"])
        penwidth = edge_data.get("width", style_data["penwidth"])

        # Add label if relationship is specified
        label = ""
        if relationship != "default" and relationship != "relates_to":
            label = relationship.replace("_", " ")

        # Apply weight for layout priorities
        weight = str(edge_data.get("weight", 1))

        # Apply styling
        graph.edge(source_id,
                   target_id,
                   label=label,
                   color=color,
                   style=style,
                   penwidth=str(penwidth),
                   weight=weight)

    def create_clusters(self, graph, graph_data):
        """
        Create clusters for nodes with the same category.

        Args:
            graph: The Graphviz graph object
            graph_data: The complete graph data
        """
        # Group nodes by category
        categories = {}
        for node in graph_data["nodes"]:
            category = node.get("category", "default")
            if category not in categories:
                categories[category] = []
            categories[category].append(node)

        # Create clusters for categories with multiple nodes
        for category, nodes in categories.items():
            if len(nodes) < 3:  # Skip small categories
                continue

            style_data = NODE_CATEGORIES.get(
                category, NODE_CATEGORIES["default"])

            with graph.subgraph(name=f"cluster_{category}") as cluster:
                cluster.attr(label=category,
                             style="filled",
                             color=self.adjust_color(style_data["color"], -10),
                             fillcolor=self.adjust_color(
                                 style_data["color"], 30),
                             fontcolor=style_data["fontcolor"],
                             fontsize="18")

                # Add nodes to cluster
                for node in nodes:
                    node_id = self.get_node_id(node["name"])
                    cluster.node(node_id)

    def create_legend(self, graph):
        """
        Create a legend explaining node categories and edge types.

        Args:
            graph: The Graphviz graph object
        """
        with graph.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend",
                        style="filled",
                        fillcolor="#F7FAFC",
                        fontsize="16",
                        fontcolor="#2D3748",
                        color="#CBD5E0")

            # Node categories
            y_pos = 0
            for category, style in NODE_CATEGORIES.items():
                if category == "default":
                    continue

                node_id = f"legend_node_{category}"
                legend.node(node_id,
                            label=category,
                            shape=style["shape"],
                            style="filled",
                            fillcolor=style["color"],
                            fontcolor=style["fontcolor"],
                            pos=f"0,{y_pos}!")
                y_pos -= 1

            # Edge relationships
            x_pos = 3
            y_pos = 0
            prev_rel = None

            for relationship, style in RELATIONSHIP_STYLES.items():
                if relationship == "default":
                    continue

                if prev_rel:
                    source_id = f"legend_edge_src_{relationship}"
                    target_id = f"legend_edge_tgt_{relationship}"

                    legend.node(source_id,
                                label="",
                                shape="point",
                                width="0.1",
                                pos=f"{x_pos},{y_pos}!")

                    legend.node(target_id,
                                label=relationship.replace("_", " "),
                                shape="plaintext",
                                pos=f"{x_pos+1.5},{y_pos}!")

                    legend.edge(source_id,
                                target_id,
                                color=style["color"],
                                style=style["style"],
                                penwidth=style["penwidth"])

                    y_pos -= 0.8
                prev_rel = relationship

    def adjust_color(self, hex_color, amount):
        """
        Adjust a hex color by the given amount.

        Args:
            hex_color: Hex color code (with or without #)
            amount: Amount to adjust brightness (-255 to 255)

        Returns:
            Adjusted hex color
        """
        hex_color = hex_color.lstrip('#')

        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Adjust values
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))

        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"

    def calculate_metrics(self, graph_data):
        """
        Calculate graph metrics for the knowledge graph.

        Args:
            graph_data: The graph data dictionary

        Returns:
            Dictionary of graph metrics
        """
        nodes_count = len(graph_data["nodes"])
        edges_count = len(graph_data["edges"])

        # Calculate node degree information
        node_degrees = {}
        for edge in graph_data["edges"]:
            source = edge["source"]
            target = edge["target"]

            if source not in node_degrees:
                node_degrees[source] = {"in": 0, "out": 0}
            if target not in node_degrees:
                node_degrees[target] = {"in": 0, "out": 0}

            node_degrees[source]["out"] += 1
            node_degrees[target]["in"] += 1

        # Find central nodes (highest total degree)
        central_nodes = sorted(
            [(node, degrees["in"] + degrees["out"])
             for node, degrees in node_degrees.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Calculate density
        max_edges = nodes_count * (nodes_count - 1)
        density = edges_count / max_edges if max_edges > 0 else 0

        return {
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "density": density,
            "central_nodes": central_nodes,
            "avg_degree": edges_count / nodes_count if nodes_count > 0 else 0
        }

    def export_graph(self,
                     graph_data: Dict,
                     file_name: str = "knowledge_graph",
                     format: str = "svg",
                     use_clusters: bool = False,
                     add_legend: bool = True,
                     layout: str = "dot",
                     rankdir: str = "TB",
                     include_metrics: bool = True) -> str:
        """
        Export graph data to the specified format with comprehensive styling.

        Args:
            graph_data: Dictionary containing nodes and edges
            file_name: Base name for the output file
            format: Output format (svg, png, pdf, dot)
            use_clusters: Whether to group nodes by category
            add_legend: Whether to add a legend
            layout: Layout algorithm to use
            rankdir: Direction of graph layout
            include_metrics: Whether to calculate and return metrics

        Returns:
            Path to the rendered file
        """
        try:
            self.current_data = graph_data
            metrics = None

            # Validate input data
            if not graph_data or "nodes" not in graph_data or "edges" not in graph_data:
                logger.error(
                    "Invalid graph data structure: missing nodes or edges")
                return None

            # Create graph with specified layout
            graph = self.create_graph(
                directed=True, layout=layout, rankdir=rankdir)
            self.current_graph = graph

            # Process nodes
            node_ids = {}  # Map node names to their IDs
            for node in graph_data["nodes"]:
                if "name" not in node:
                    continue

                node_id = self.get_node_id(node["name"])
                node_ids[node["name"]] = node_id
                self.style_node(graph, node_id, node)

            # Process edges
            for edge in graph_data["edges"]:
                if "source" not in edge or "target" not in edge:
                    continue

                source = edge["source"]
                target = edge["target"]

                if source not in node_ids or target not in node_ids:
                    continue

                self.style_edge(
                    graph, node_ids[source], node_ids[target], edge)

            # Create clusters if requested
            if use_clusters:
                self.create_clusters(graph, graph_data)

            # Add legend if requested
            if add_legend:
                self.create_legend(graph)

            # Calculate metrics if requested
            if include_metrics:
                metrics = self.calculate_metrics(graph_data)

            # Ensure the output directory exists
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"{file_name}_{timestamp}")

            # Render the graph
            output_file = graph.render(
                output_path, format=format, cleanup=True)
            logger.info(f"Graph exported to {output_file}")

            # Return the path and metrics
            result = {
                "file_path": output_file,
                "metrics": metrics
            }

            # Save metrics as JSON if calculated
            if metrics:
                metrics_path = f"{output_path}_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                result["metrics_path"] = metrics_path

            return result

        except Exception as e:
            logger.exception(f"Error exporting graph: {str(e)}")
            return None

    def export_subgraph(self,
                        central_node: str,
                        depth: int = 2,
                        file_name: str = "subgraph",
                        **kwargs) -> str:
        """
        Export a subgraph centered around a specific node.

        Args:
            central_node: The node to center the subgraph around
            depth: How many edge traversals to include
            file_name: Base name for the output file
            **kwargs: Additional arguments for export_graph

        Returns:
            Path to the rendered file
        """
        if not self.current_data:
            logger.error("No graph data available. Call export_graph first.")
            return None

        # Collect nodes within the specified depth
        included_nodes = set([central_node])
        frontier = set([central_node])

        for _ in range(depth):
            new_frontier = set()

            # Add connected nodes
            for edge in self.current_data["edges"]:
                source = edge["source"]
                target = edge["target"]

                if source in frontier and target not in included_nodes:
                    new_frontier.add(target)
                    included_nodes.add(target)

                if target in frontier and source not in included_nodes:
                    new_frontier.add(source)
                    included_nodes.add(source)

            frontier = new_frontier
            if not frontier:
                break

        # Create subgraph data
        subgraph_data = {
            "nodes": [node for node in self.current_data["nodes"]
                      if node["name"] in included_nodes],
            "edges": [edge for edge in self.current_data["edges"]
                      if edge["source"] in included_nodes and edge["target"] in included_nodes]
        }

        # Export the subgraph
        return self.export_graph(subgraph_data, file_name=f"{file_name}_{central_node}", **kwargs)

    def batch_export(self,
                     graph_data: Dict,
                     file_name: str = "knowledge_graph",
                     max_nodes_per_graph: int = 25,
                     **kwargs) -> List[str]:
        """
        Export large graphs by breaking them into multiple smaller graphs.

        Args:
            graph_data: Dictionary containing nodes and edges
            file_name: Base name for the output files
            max_nodes_per_graph: Maximum nodes per subgraph
            **kwargs: Additional arguments for export_graph

        Returns:
            List of paths to the rendered files
        """
        if len(graph_data["nodes"]) <= max_nodes_per_graph:
            result = self.export_graph(
                graph_data, file_name=file_name, **kwargs)
            return [result["file_path"]] if result else []

        # Calculate node importance or centrality
        node_connections = {}
        for edge in graph_data["edges"]:
            source = edge["source"]
            target = edge["target"]

            if source not in node_connections:
                node_connections[source] = 0
            if target not in node_connections:
                node_connections[target] = 0

            node_connections[source] += 1
            node_connections[target] += 1

        # Sort nodes by importance (connection count)
        sorted_nodes = sorted(
            [(node["name"], node_connections.get(node["name"], 0))
             for node in graph_data["nodes"]],
            key=lambda x: x[1],
            reverse=True
        )

        # Create batches
        batches = []
        remaining_nodes = set(node["name"] for node in graph_data["nodes"])

        while remaining_nodes:
            batch = set()

            # Try to add nodes from the sorted list
            for node_name, _ in sorted_nodes:
                if node_name in remaining_nodes and len(batch) < max_nodes_per_graph:
                    batch.add(node_name)
                    remaining_nodes.remove(node_name)

            if batch:
                batches.append(batch)

        # Export each batch
        results = []
        for i, batch in enumerate(batches):
            batch_data = {
                "nodes": [node for node in graph_data["nodes"]
                          if node["name"] in batch],
                "edges": [edge for edge in graph_data["edges"]
                          if edge["source"] in batch and edge["target"] in batch]
            }

            result = self.export_graph(
                batch_data,
                file_name=f"{file_name}_part{i+1}",
                **kwargs
            )

            if result:
                results.append(result["file_path"])

        return results

    def export_all_formats(self,
                           graph_data: Dict,
                           file_name: str = "knowledge_graph",
                           formats: List[str] = ["svg", "png", "pdf", "dot"],
                           **kwargs) -> Dict[str, str]:
        """
        Export graph to multiple formats.

        Args:
            graph_data: Dictionary containing nodes and edges
            file_name: Base name for the output files
            formats: List of formats to export
            **kwargs: Additional arguments for export_graph

        Returns:
            Dictionary mapping formats to file paths
        """
        results = {}

        for format in formats:
            result = self.export_graph(
                graph_data,
                file_name=file_name,
                format=format,
                **kwargs
            )

            if result:
                results[format] = result["file_path"]

        return results


# Example usage function
def export_to_svg(graph_data, file_name="output_graph", use_enhanced=True):
    """
    Export graph data to SVG file.

    Args:
        graph_data: Dictionary containing nodes and edges
        file_name: Base name for the output file
        use_enhanced: Whether to use the enhanced exporter

    Returns:
        Path to the output SVG file
    """
    if use_enhanced:
        # Use the enhanced exporter
        exporter = GraphExporter()
        result = exporter.export_graph(
            graph_data,
            file_name=file_name,
            format="svg",
            use_clusters=True,
            add_legend=True
        )
        return result["file_path"] if result else None
    else:
        # Legacy implementation
        dot = Digraph(format="svg")
        dot.attr(bgcolor="white")
        dot.attr("graph", rankdir="TB", ranksep="2.5", nodesep="1.5")
        dot.attr("node", shape="box", style="filled,setlinewidth(3)", width="2", height="1", fontsize="24",
                 fillcolor="#FFF8DC", fontname="Helvetica-Bold", color="#FFB300", fontcolor="#000000")
        dot.attr("edge", color="#999999", arrowsize="1.4", penwidth="2.0")

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

        output_path = os.path.join("data/outputs", f"{file_name}")
        dot.render(output_path, format="svg", cleanup=True)
        return f"{output_path}.svg"


# Legacy sanitize function for backward compatibility
def sanitize(text):
    clean = str(text).replace('"', "'").replace("\n", " ").strip()
    clean = re.sub(r"[<>\\]", "", clean)
    return clean[:120] + "..." if len(clean) > 120 else clean


# For demonstration and testing
if __name__ == "__main__":
    # Sample graph data
    sample_data = {
        "nodes": [
            {"name": "AI Research", "category": "Topic", "importance": 3},
            {"name": "Data Collection", "category": "Phase", "importance": 2},
            {"name": "Analysis", "category": "Phase", "importance": 2},
            {"name": "Visualization", "category": "Phase", "importance": 2},
            {"name": "SERP API", "category": "Tool", "importance": 1},
            {"name": "EURI API", "category": "Tool", "importance": 1},
            {"name": "Graphviz", "category": "Technology", "importance": 1}
        ],
        "edges": [
            {"source": "AI Research", "target": "Data Collection",
                "relationship": "includes"},
            {"source": "Data Collection", "target": "Analysis",
                "relationship": "followed_by"},
            {"source": "Analysis", "target": "Visualization",
                "relationship": "followed_by"},
            {"source": "Data Collection", "target": "SERP API",
                "relationship": "requires"},
            {"source": "Analysis", "target": "EURI API", "relationship": "requires"},
            {"source": "Visualization", "target": "Graphviz",
                "relationship": "requires"}
        ]
    }

    # Export using the enhanced exporter
    exporter = GraphExporter()
    result = exporter.export_graph(
        sample_data,
        file_name="example_knowledge_graph",
        use_clusters=True,
        add_legend=True
    )

    print(f"Graph exported to: {result['file_path']}")
    print(f"Graph metrics: {result['metrics']}")
