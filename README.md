# MultiAgentResearchGraph

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Project Status](https://img.shields.io/badge/Status-Active-green.svg) ![Python](https://img.shields.io/badge/Python-3.9+-orange.svg) ![LangGraph](https://img.shields.io/badge/LangGraph-0.0.30+-purple.svg)

## Overview

This project focuses on developing, orchestrating, and deploying a multi-agent AI system that transforms natural language queries into interactive knowledge graphs. By leveraging specialized AI agents for research, summarization, and visualization, the system creates comprehensive knowledge representations that reveal relationships between concepts, enabling users to quickly understand complex topics and discover non-obvious connections.

This repository contains the code, documentation, and resources necessary to run and extend this end-to-end AI research assistant solution.

### Key Features:

- 🤖 Multi-Agent LLM Architecture with Specialized Agents
- 🔄 Advanced LangGraph Workflow Orchestration
- 🔍 Automated Web Research and Content Aggregation
- 📝 Intelligent Content Summarization
- 📊 Interactive Knowledge Graph Visualization
- 🌐 Modern Streamlit Web Interface
- 🔄 Session Management and Persistence
- 📤 Export and Sharing Capabilities
- 📦 Modular and Extensible Architecture

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Technologies](#technologies)
- [Methodology](#methodology)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [LangGraph Workflow](#langgraph-workflow)
- [Knowledge Graph Visualization](#knowledge-graph-visualization)
- [Results and Impact](#results-and-impact)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

In today's information-rich environment, researchers, students, and professionals face significant challenges when attempting to efficiently gather and synthesize knowledge from the vast amounts of data available online. Current research tools often deliver fragmented results without showing relationships between concepts, requiring significant manual effort to construct a comprehensive understanding of a topic. This process is time-consuming, inefficient, and often results in missed connections between related concepts.

Traditional search engines return lists of links without context, and existing knowledge management tools typically require manual input to establish relationships between entities. There exists a critical gap in technology that can automatically transform natural language queries into structured, visual representations of interconnected knowledge.

---

## Project Goals

The primary objectives of this project are:

1. Develop an end-to-end AI system that transforms natural language research queries into interactive knowledge graphs showing relationships between key concepts.
2. Implement a multi-agent architecture using LangGraph to orchestrate specialized AI agents working in concert to process information.
3. Create a system that automates the entire knowledge discovery and visualization pipeline, from initial search to final presentation.
4. Build an intuitive user interface that allows non-technical users to leverage advanced AI capabilities.
5. Ensure scalability and extensibility through a modular architecture that can accommodate future AI advancements.

---

## Technologies

This project leverages several cutting-edge technologies to deliver its functionality:

| Technology         | Purpose                                                   |
|--------------------|------------------------------------------------------------|
| `Python 3.9+`      | Core programming language                                 |
| `LangGraph`        | Multi-agent workflow orchestration                        |
| `LangChain`        | LLM framework and tools                                   |
| `Streamlit`        | Web application interface                                 |
| `SERP API`         | Web search functionality                                  |
| `EURI API`         | Advanced text summarization                               |
| `Plotly`           | Interactive visualizations                                |
| `NetworkX`         | Graph data structure and algorithms                       |
| `Graphviz`         | Knowledge graph visualization                             |

---

## Methodology

This project follows a structured AI workflow:

1. **User Query Processing:** Parse and understand natural language queries to extract key concepts and requirements.
2. **Web Research:** Use specialized research agent with SERP API to gather relevant information from diverse sources.
3. **Content Summarization:** Process raw data through the summarizer agent to extract key concepts and relationships.
4. **Graph Construction:** Transform processed data into structured graph representations through the mapper agent.
5. **Interactive Visualization:** Present the knowledge graph through an intuitive, interactive visualization.
6. **Result Persistence:** Store and make results shareable for future reference.

---

## Multi-Agent Architecture

<a name="multi-agent-architecture"></a>  
The system employs three specialized AI agents working in concert:

- **Research Agent (A1):** Leverages SERP API to intelligently gather and filter web results based on relevance to the query. This agent:
  - Processes natural language queries into effective search parameters
  - Executes strategic web searches to gather comprehensive information
  - Filters and ranks results based on relevance and authority
  - Structures raw data for effective processing by subsequent agents

- **Summarizer Agent (A2):** Uses the EURI API to process raw data into structured, contextualized information. This agent:
  - Analyzes raw search results to identify key concepts and themes
  - Extracts entities, relationships, and contextual information
  - Restructures information into a format optimized for knowledge graph generation
  - Prioritizes information based on relevance to the original query

- **Graph Mapper Agent (A3):** Transforms processed text into structured graph data, extracting entities and relationships. This agent:
  - Identifies nodes (concepts) and edges (relationships) from summarized content
  - Categorizes nodes based on type and importance
  - Determines relationship types between connected concepts
  - Generates a structured representation suitable for visualization

Each agent is specialized for its specific task while maintaining seamless information flow between components.

---

## LangGraph Workflow

<a name="langgraph-workflow"></a>  
The system uses LangGraph to orchestrate the multi-agent workflow with these key components:

- **State Management System:** A sophisticated mechanism for tracking the progress of queries through the system pipeline, maintaining context, and ensuring data integrity between agent transitions.

- **Conditional Routing Logic:** Intelligent decision-making processes determine workflow paths based on results at each stage, enabling dynamic adaptation to different types of queries and information structures.

- **Error Recovery Mechanisms:** Robust handling of exceptions, timeouts, and incomplete results ensures system resilience and graceful degradation when faced with challenges.

- **Parallel Processing:** Where appropriate, the system can process multiple information streams simultaneously to improve efficiency and comprehensiveness.

- **Workflow Checkpoints:** State persistence at key points enables resumability and tracking of the information transformation process.

This orchestration system ensures that information flows smoothly between agents and maintains context throughout the knowledge discovery process.

---

## Knowledge Graph Visualization

<a name="knowledge-graph-visualization"></a>  
The system creates sophisticated knowledge graph visualizations with these features:

- **Interactive Network Visualization:** Dynamic, user-manipulable graph representation using Plotly for exploration of complex concept networks.

- **Entity Categorization:** Automatic classification of nodes into categories (Topics, Concepts, Technologies, etc.) with visual differentiation through color-coding and shape.

- **Relationship Labeling:** Explicit labeling of connections between entities (includes, requires, leads_to, etc.) to clarify the nature of relationships.

- **Importance Weighting:** Visual emphasis on key concepts through size scaling based on centrality and relevance to the query.

- **Filtering and Focus:** Tools for isolating specific parts of the knowledge graph to explore particular areas of interest.

- **Export Capabilities:** Options to save visualizations in multiple formats (SVG, PNG) for external use and sharing.

These visualization techniques transform abstract information into intuitive, interactive displays that facilitate understanding and insight discovery.

---

## Results and Impact

<a name="results-and-impact"></a>  
The AI Research Agent system delivers significant value through:

1. **Accelerated Research Efficiency:** Reduces the time required to gather and synthesize information from hours to minutes by automating the discovery and organization of knowledge.

2. **Enhanced Conceptual Understanding:** Reveals relationships between concepts that might otherwise be missed, enabling deeper insights and more comprehensive understanding of complex topics.

3. **Visual Knowledge Representation:** Transforms abstract information into intuitive, interactive visualizations that make complex topics more accessible.

4. **Seamless User Experience:** Provides an intuitive interface that makes advanced AI capabilities accessible to users of any technical background.

5. **Extensible AI Framework:** Establishes a foundation that can be enhanced with additional agents, data sources, and visualization methods.

These benefits address the core challenges of modern information discovery and synthesis, demonstrating the potential of multi-agent AI systems to transform knowledge work.

---

## Getting Started

<a name="getting-started"></a>

### Prerequisites

<a name="prerequisites"></a>
To run this project, you will need:

- Python 3.9 or higher
- API keys for:
  - SERP API (for web search functionality)
  - EURI API (for text summarization)
- Graphviz (system installation)

### Installation

<a name="installation"></a>

```bash
# Clone the repository
git clone https://github.com/yourusername/MultiAgentResearchGraph.git
cd MultiAgentResearchGraph

# Create and activate a virtual environment
python -m venv multi_agent_env
source multi_agent_env/bin/activate  # On Windows: multi_agent_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables by creating a .env file
touch .env

# Add your API keys to the .env file
echo "SERP_API_KEY=your_serp_api_key_here" >> .env
echo "EURI_API_KEY=your_euri_api_key_here" >> .env
