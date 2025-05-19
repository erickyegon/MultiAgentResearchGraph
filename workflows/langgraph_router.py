"""
router.py - LangGraph workflow orchestration for AI Research Agent System

This module defines the workflow that coordinates the three agents:
1. Research Agent (A1): Performs web searches using SERP API
2. Summarizer Agent (A2): Summarizes search results using EURI API
3. Graph Mapper Agent (A3): Generates knowledge graphs using Graphviz

The workflow handles state management, agent transitions, and error handling.
"""

import os
import json
import logging
from typing import Dict, List, Any, TypedDict, Optional, Annotated
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.prompts import PromptTemplate
from langchain.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field

# Import agent implementations
from agents.researcher import search_google, ResearchAgent
from agents.synthesizer import synthesize, SummarizerAgent
from agents.mapper import run_mapper, GraphMapperAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the state structure using TypedDict
class WorkflowState(TypedDict):
    query: str
    research_results: Optional[List[Dict[str, Any]]]
    summarized_content: Optional[str]
    knowledge_graph: Optional[Dict[str, Any]]
    errors: Optional[List[Dict[str, Any]]]
    metadata: Dict[str, Any]

# Node implementations for each agent
def research_agent(state: WorkflowState) -> WorkflowState:
    """
    Research Agent (A1) node - Performs web searches using SERP API
    """
    logger.info(f"Research Agent processing query: {state['query']}")
    
    try:
        query = state["query"]
        
        # Create instance of research agent
        agent = ResearchAgent()
        
        # Execute search
        results = agent.search(query)
        
        # Process and structure the results
        processed_results = agent.process_results(results)
        
        # Update state with research results
        state["research_results"] = processed_results
        
        # Add metadata about this step
        state["metadata"]["research_timestamp"] = datetime.now().isoformat()
        state["metadata"]["research_query"] = query
        state["metadata"]["results_count"] = len(processed_results)
        
        logger.info(f"Research completed successfully with {len(processed_results)} results")
        
    except Exception as e:
        error_msg = f"Error in research agent: {str(e)}"
        logger.error(error_msg)
        
        if "errors" not in state or state["errors"] is None:
            state["errors"] = []
            
        state["errors"].append({
            "agent": "research",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state

def summarizer_agent(state: WorkflowState) -> WorkflowState:
    """
    Summarizer Agent (A2) node - Summarizes research results using EURI API
    """
    logger.info("Summarizer Agent processing research results")
    
    try:
        # Check if we have research results
        if "research_results" not in state or not state["research_results"]:
            raise ValueError("No research results available for summarization")
        
        # Format data for the synthesizer
        raw_data = {
            f"result_{i}": json.dumps(result)
            for i, result in enumerate(state["research_results"])
        }
        
        # Invoke summarizer function
        summary = synthesize(raw_data)
        
        # Update state with summarized content
        state["summarized_content"] = summary
        
        # Add metadata about this step
        state["metadata"]["summarizer_timestamp"] = datetime.now().isoformat()
        state["metadata"]["summary_length"] = len(summary)
        
        logger.info(f"Summarization completed successfully - {len(summary)} characters generated")
        
    except Exception as e:
        error_msg = f"Error in summarizer agent: {str(e)}"
        logger.error(error_msg)
        
        if "errors" not in state or state["errors"] is None:
            state["errors"] = []
            
        state["errors"].append({
            "agent": "summarizer",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state

def mapper_agent(state: WorkflowState) -> WorkflowState:
    """
    Graph Mapper Agent (A3) node - Generates knowledge graph using Graphviz
    """
    logger.info("Graph Mapper Agent generating knowledge graph")
    
    try:
        # Check if we have summarized content
        if "summarized_content" not in state or not state["summarized_content"]:
            raise ValueError("No summarized content available for graph mapping")
        
        # Generate knowledge graph
        graph_data = run_mapper(state["summarized_content"])
        
        # Update state with knowledge graph
        state["knowledge_graph"] = graph_data
        
        # Add metadata about this step
        state["metadata"]["mapper_timestamp"] = datetime.now().isoformat()
        state["metadata"]["nodes_count"] = len(graph_data["nodes"])
        state["metadata"]["edges_count"] = len(graph_data["edges"])
        
        logger.info(f"Graph mapping completed successfully with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
        
    except Exception as e:
        error_msg = f"Error in mapper agent: {str(e)}"
        logger.error(error_msg)
        
        if "errors" not in state or state["errors"] is None:
            state["errors"] = []
            
        state["errors"].append({
            "agent": "mapper",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    return state

def should_continue(state: WorkflowState) -> str:
    """
    Determines whether to continue the workflow or terminate due to errors
    """
    # Check for errors
    if "errors" in state and state["errors"]:
        # If the research agent failed completely (no results), we can't continue
        if any(e["agent"] == "research" for e in state["errors"]) and not state.get("research_results"):
            logger.error("Critical error in research agent - terminating workflow")
            return "terminate"
        
        # If the summarizer agent failed but we have research results, we might still be able to map them
        if any(e["agent"] == "summarizer" for e in state["errors"]) and not state.get("summarized_content"):
            # Generate simple summarized content from research results
            try:
                simple_summary = "\n".join([
                    f"{result.get('title', 'Unknown')} -> {result.get('snippet', 'No description')}"
                    for result in state.get("research_results", [])
                ])
                state["summarized_content"] = simple_summary
                logger.warning("Using simplified summary due to summarizer failure")
                return "continue_mapping"
            except:
                logger.error("Failed to create fallback summary - terminating workflow")
                return "terminate"
    
    # Normal flow
    if "research_results" in state and state["research_results"]:
        if "summarized_content" in state and state["summarized_content"]:
            return "continue_mapping"
        else:
            return "continue_summarizing"
    else:
        return "terminate"

# Create the LangGraph workflow
def create_research_workflow() -> StateGraph:
    """
    Creates and configures the LangGraph workflow for the AI Research Agent
    """
    # Initialize the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes for each agent
    workflow.add_node("research", research_agent)
    workflow.add_node("summarize", summarizer_agent)
    workflow.add_node("map", mapper_agent)
    
    # Set the entry point
    workflow.set_entry_point("research")
    
    # Define the edges based on the decision function
    workflow.add_conditional_edges(
        "research",
        should_continue,
        {
            "continue_summarizing": "summarize",
            "continue_mapping": "map",
            "terminate": END
        }
    )
    
    workflow.add_conditional_edges(
        "summarize",
        should_continue,
        {
            "continue_mapping": "map",
            "terminate": END
        }
    )
    
    # Add edge from mapper to end
    workflow.add_edge("map", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow

# Main execution function
def run_research_workflow(query: str) -> Dict[str, Any]:
    """
    Executes the entire research workflow for a given query
    
    Args:
        query: The user's research question
        
    Returns:
        The final state of the workflow containing all results
    """
    logger.info(f"Starting research workflow for query: {query}")
    
    # Create the workflow
    workflow = create_research_workflow()
    
    # Initialize the state
    initial_state: WorkflowState = {
        "query": query,
        "research_results": None,
        "summarized_content": None,
        "knowledge_graph": None,
        "errors": None,
        "metadata": {
            "workflow_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "start_timestamp": datetime.now().isoformat(),
            "query": query
        }
    }
    
    # Execute the workflow
    try:
        result = workflow.invoke(initial_state)
        logger.info("Workflow execution completed successfully")
        
        # Add completion metadata
        result["metadata"]["end_timestamp"] = datetime.now().isoformat()
        result["metadata"]["success"] = True
        
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        
        # Create error result
        error_result = initial_state.copy()
        if "errors" not in error_result or error_result["errors"] is None:
            error_result["errors"] = []
            
        error_result["errors"].append({
            "agent": "workflow",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        
        error_result["metadata"]["end_timestamp"] = datetime.now().isoformat()
        error_result["metadata"]["success"] = False
        
        return error_result

# Initialize the checkpoint saver (for restart capability)
memory_saver = MemorySaver()

# Configuration for persistence
def configure_persistence(persistence_dir: str = "./workflow_checkpoints"):
    """
    Configures workflow persistence for restart capability
    """
    os.makedirs(persistence_dir, exist_ok=True)
    # Additional persistence setup can be added here
    logger.info(f"Workflow persistence configured to: {persistence_dir}")