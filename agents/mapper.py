import json
import re
import logging
from typing import Dict, List, Set, Tuple, Optional

def run_mapper(synthesized_text: str) -> Dict:
    """
    Transforms synthesized text into a knowledge graph structure with nodes and edges.
    
    This enhanced mapper supports:
    - Multiple relationship formats (-> and →)
    - Node categorization and metadata
    - Edge attributes and relationship types
    - Hierarchical relationships
    - Visual styling based on node importance
    
    Args:
        synthesized_text: Text output from the synthesizer agent
        
    Returns:
        Dict containing nodes, edges, and metadata for graph visualization
    """
    logging.info("Starting knowledge graph mapping process")
    
    # Initialize data structures
    nodes = []
    edges = []
    added_nodes = set()
    node_categories = {}
    importance_scores = {}
    
    # Split text into lines and clean
    lines = [line.strip() for line in synthesized_text.split("\n") if line.strip()]
    
    # Process each line for potential relationships
    for line_index, line in enumerate(lines):
        # Look for explicit relationships with arrows
        if "->" in line or "→" in line:
            # Normalize arrow formats
            normalized_line = line.replace("→", "->")
            
            # Check for relationship labels [type] in the format "A -[relates to]-> B"
            relationship_match = re.findall(r'(.*?)\s*-\[([^\]]+)\]->\s*(.*)', normalized_line)
            
            if relationship_match:
                # Process labeled relationships
                for source, rel_type, target in relationship_match:
                    source = source.strip()
                    target = target.strip()
                    rel_type = rel_type.strip()
                    
                    # Add nodes if they don't exist
                    for node, position in [(source, "source"), (target, "target")]:
                        if node and node not in added_nodes:
                            # Determine node type based on contextual clues
                            category = determine_node_category(node, line, lines)
                            
                            # Create node with metadata
                            node_data = {
                                "id": node,
                                "name": node,
                                "category": category,
                                "importance": calculate_importance(node, synthesized_text)
                            }
                            nodes.append(node_data)
                            added_nodes.add(node)
                            node_categories[node] = category
                            importance_scores[node] = node_data["importance"]
                    
                    # Add edge with relationship type
                    if source and target:
                        edges.append({
                            "source": source,
                            "target": target,
                            "relationship": rel_type,
                            "weight": 2,  # Higher weight for explicit labeled relationships
                        })
            else:
                # Process standard arrow relationships
                parts = [part.strip() for part in normalized_line.split("->")]
                
                # Add all nodes first
                for part in parts:
                    if part and part not in added_nodes:
                        # Determine node type and importance
                        category = determine_node_category(part, line, lines)
                        importance = calculate_importance(part, synthesized_text)
                        
                        # Add node with metadata
                        node_data = {
                            "id": part,
                            "name": part,
                            "category": category,
                            "importance": importance
                        }
                        nodes.append(node_data)
                        added_nodes.add(part)
                        node_categories[part] = category
                        importance_scores[part] = importance
                
                # Then add all edges
                for i in range(len(parts) - 1):
                    if parts[i] and parts[i+1]:
                        # Infer relationship type if possible
                        rel_type = infer_relationship_type(parts[i], parts[i+1], line)
                        
                        edges.append({
                            "source": parts[i],
                            "target": parts[i+1],
                            "relationship": rel_type,
                            "weight": 1
                        })
        
        # Look for implicit relationships (paragraphs that mention multiple concepts)
        elif line_index > 0 and len(line.split()) > 3:  # Only consider substantial lines
            extract_implicit_relationships(line, nodes, edges, added_nodes, node_categories, importance_scores)
    
    # Post-processing to enhance graph
    enrich_graph_metadata(nodes, edges, synthesized_text)
    
    # Return the complete graph data
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "categories": list(set(node_categories.values())),
            "created": True
        }
    }
    
    logging.info(f"Knowledge graph created with {len(nodes)} nodes and {len(edges)} edges")
    return graph_data

def determine_node_category(node: str, context_line: str, all_lines: List[str]) -> str:
    """Determine the category of a node based on its context."""
    # This is a simplified implementation - could be expanded with NLP
    node_lower = node.lower()
    
    # Check for common category patterns
    if any(tech in node_lower for tech in ["python", "java", "javascript", "sql", "react", "node"]):
        return "Technology"
    elif any(concept in node_lower for concept in ["fundamentals", "basics", "introduction", "principles"]):
        return "Concept"
    elif any(phase in node_lower for phase in ["phase", "stage", "step", "level"]):
        return "Phase"
    elif any(tool in node_lower for tool in ["tool", "library", "framework", "platform"]):
        return "Tool"
    elif any(time in node_lower for time in ["week", "month", "day", "hour"]):
        return "TimeEstimate"
    elif any(outcome in node_lower for outcome in ["skill", "certificate", "job", "position", "role"]):
        return "Outcome"
    
    # Default
    return "Topic"

def calculate_importance(node: str, full_text: str) -> int:
    """Calculate importance score based on frequency and context."""
    # Count occurrences (case-insensitive)
    occurrences = len(re.findall(r'\b' + re.escape(node) + r'\b', full_text, re.IGNORECASE))
    
    # Adjust importance based on occurrence patterns
    if occurrences > 5:
        return 3  # High importance
    elif occurrences > 2:
        return 2  # Medium importance
    else:
        return 1  # Standard importance

def infer_relationship_type(source: str, target: str, context: str) -> str:
    """Infer the type of relationship between nodes based on context."""
    context_lower = context.lower()
    
    # Look for relationship indicators in context
    if "requires" in context_lower or "prerequisite" in context_lower:
        return "requires"
    elif "leads to" in context_lower or "results in" in context_lower:
        return "leads_to"
    elif "includes" in context_lower or "contains" in context_lower:
        return "includes"
    elif "part of" in context_lower or "belongs to" in context_lower:
        return "part_of"
    elif "followed by" in context_lower or "then" in context_lower:
        return "followed_by"
    
    # Default relationship
    return "relates_to"

def extract_implicit_relationships(line: str, nodes: List[Dict], edges: List[Dict], 
                                  added_nodes: Set[str], node_categories: Dict, 
                                  importance_scores: Dict) -> None:
    """Extract implicit relationships from paragraph text."""
    # This is a simplified implementation - could use NLP for better extraction
    # Currently just looking for known nodes mentioned together in the same context
    
    existing_node_names = list(added_nodes)
    mentioned_nodes = []
    
    # Find which existing nodes are mentioned in this line
    for node_name in existing_node_names:
        if re.search(r'\b' + re.escape(node_name) + r'\b', line, re.IGNORECASE):
            mentioned_nodes.append(node_name)
    
    # If multiple nodes are mentioned together, create implicit relationships
    if len(mentioned_nodes) > 1:
        # Connect nodes that appear in the same context
        for i in range(len(mentioned_nodes) - 1):
            # Only add edge if it doesn't already exist
            edge_exists = any(
                e["source"] == mentioned_nodes[i] and e["target"] == mentioned_nodes[i+1]
                for e in edges
            )
            
            if not edge_exists:
                edges.append({
                    "source": mentioned_nodes[i],
                    "target": mentioned_nodes[i+1],
                    "relationship": "mentioned_together",
                    "weight": 0.5  # Lower weight for implicit relationships
                })

def enrich_graph_metadata(nodes: List[Dict], edges: List[Dict], full_text: str) -> None:
    """Add visual styling and additional metadata to improve graph rendering."""
    # Enrich nodes with visual properties based on importance and category
    for node in nodes:
        # Set node size based on importance
        if node["importance"] == 3:
            node["size"] = 40
            node["font_size"] = 16
        elif node["importance"] == 2:
            node["size"] = 30
            node["font_size"] = 14
        else:
            node["size"] = 20
            node["font_size"] = 12
        
        # Set node color based on category
        if node["category"] == "Technology":
            node["color"] = "#4299e1"  # Blue
        elif node["category"] == "Concept":
            node["color"] = "#48bb78"  # Green
        elif node["category"] == "Phase":
            node["color"] = "#ed8936"  # Orange
        elif node["category"] == "Tool":
            node["color"] = "#9f7aea"  # Purple
        elif node["category"] == "TimeEstimate":
            node["color"] = "#f56565"  # Red
        elif node["category"] == "Outcome":
            node["color"] = "#ecc94b"  # Yellow
        else:
            node["color"] = "#a0aec0"  # Gray
    
    # Enrich edges with styling based on relationship type and weight
    for edge in edges:
        # Set edge thickness based on weight
        edge["width"] = edge["weight"] * 2
        
        # Set edge style based on relationship type
        if edge["relationship"] == "requires":
            edge["color"] = "#f56565"  # Red
            edge["style"] = "solid"
        elif edge["relationship"] == "leads_to":
            edge["color"] = "#48bb78"  # Green
            edge["style"] = "solid"
        elif edge["relationship"] == "includes":
            edge["color"] = "#4299e1"  # Blue
            edge["style"] = "solid"
        elif edge["relationship"] == "part_of":
            edge["color"] = "#9f7aea"  # Purple
            edge["style"] = "dashed"
        elif edge["relationship"] == "followed_by":
            edge["color"] = "#ed8936"  # Orange
            edge["style"] = "solid"
        elif edge["relationship"] == "mentioned_together":
            edge["color"] = "#a0aec0"  # Gray
            edge["style"] = "dotted"
        else:
            edge["color"] = "#a0aec0"  # Gray
            edge["style"] = "solid"