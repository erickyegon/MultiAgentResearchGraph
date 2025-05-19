# AI Research Agent System: Knowledge Graph Generation from Natural Language Queries

## 🔍 Problem Statement

In today's information-rich environment, researchers, students, and professionals face significant challenges in efficiently gathering and synthesizing knowledge from the vast amounts of data available online. 

Current tools:
- Deliver fragmented results without showing relationships between concepts
- Require significant manual effort to construct a comprehensive understanding
- Return link lists without meaningful context

This leads to inefficiencies, missed connections, and shallow understanding of complex topics.

## 🎯 Project Objectives

- ✅ Transform natural language research queries into **interactive knowledge graphs**
- ✅ Implement a **multi-agent architecture** using [LangGraph](https://github.com/langchain-ai/langgraph) to coordinate specialized AI agents
- ✅ Automate the full **knowledge discovery and visualization pipeline**
- ✅ Provide a **user-friendly interface** for non-technical users
- ✅ Ensure **scalability and modularity** for future AI upgrades

---

## 🧠 Technical Implementation

### 🕸 Multi-Agent LLM Architecture

- **Research Agent (A1)**: Uses SERP API to fetch and filter relevant web results
- **Summarizer Agent (A2)**: Processes raw content into structured, contextual summaries via EURI API
- **Graph Mapper Agent (A3)**: Extracts entities and relationships to form graph structure

### 🧭 LangGraph Workflow Orchestration

- Custom state management and branching logic
- Workflow orchestration with conditional agent execution
- Robust error recovery for system stability

### 📊 Knowledge Graph Visualization

- Interactive, dynamic network graphs
- Node categorization and hierarchy
- Entity relationship detection and labeling

### 🌐 Web Application (Streamlit)

- Responsive, clean UI with custom CSS
- Session management and persistence
- Export functionality for collaboration

---

## 📈 Results & Impact

- ⚡ **Accelerated Research**: Reduces time from hours to minutes
- 🧠 **Deeper Insights**: Reveals hidden conceptual relationships
- 🗺️ **Visual Understanding**: Transforms abstract queries into visual graphs
- 🎯 **User-Friendly**: Non-technical users can access powerful AI tools
- 🔧 **Extensible Framework**: Supports new agents, APIs, and graph formats

---

## 🛠️ Technical Skills Demonstrated

- ✅ Large Language Model integration (OpenAI, Hugging Face)
- ✅ Multi-agent orchestration with LangGraph
- ✅ Knowledge graph construction (networkx, pyvis, etc.)
- ✅ State-of-the-art prompt engineering
- ✅ API integration (SERP, EURI, custom tools)
- ✅ Modern web development with Streamlit
- ✅ Robust session and error handling
- ✅ Interactive data visualization

---

## 🚀 Future Work

- Integrate citation & source tracking
- Add new agent types (e.g., summarizers for PDFs)
- Use multimodal models for images/graphs
- Extend UI to support graph editing and refinement

---

## 🧑‍💻 Author

**Erick Yegon**  
Built with ❤️ using LangChain, LangGraph, Streamlit, and cutting-edge LLMs

[GitHub](https://github.com/erickyegon) • [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
