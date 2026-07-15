import streamlit as st
import websocket
import json
import graphviz
import time

# Set up page configurations
st.set_page_config(
    page_title="ThinkLM Self-Evolving Agent Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glass card container style */
    .glass-card {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.125);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Header title style */
    .gradient-text {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    /* Subtitle style */
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Status indicators */
    .status-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .status-idle {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="gradient-text">ThinkLM Agent Orchestrator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Agent Cooperative Loop • Parallel DAG Execution • Secure Subprocess Sandbox</div>', unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.markdown("### ⚙️ Engine Configurations")
    ws_url = st.text_input("WebSocket Endpoint", value="ws://localhost:8000/ws/agent")
    
    st.markdown("---")
    
    st.markdown("### 🛡️ Secure Sandbox Settings")
    st.write("Sandbox directory: `./sandbox/` (Containment Active)")
    timeout_sec = st.slider("Subprocess Timeout (seconds)", min_value=1, max_value=20, value=5)
    
    st.markdown("---")
    
    st.markdown("### ⚡ Registered MCP Servers")
    st.markdown("""
    - <span class="status-badge status-active">web_search</span> `Clustered search APIs`
    - <span class="status-badge status-active">calculator</span> `Python numpy math`
    - <span class="status-badge status-active">sandbox</span> `Secure python runner`
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 Capstone Metadata")
    st.info("BTech Capstone Exhibition — Self-Evolving Language Model Platform")

# Chat interface structure
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "dag" in msg and msg["dag"]:
            with st.expander("Planned Task Graph (DAG)", expanded=False):
                # Render graphviz if tasks exist
                tasks = msg["dag"].get("tasks", [])
                deps = msg["dag"].get("dependencies", [])
                dot = graphviz.Digraph(comment='Task DAG')
                dot.attr(bgcolor='transparent', rankdir='LR')
                dot.attr('node', shape='box', style='filled,rounded', color='#3b82f6', fontcolor='white', fillcolor='#1e293b')
                dot.attr('edge', color='#64748b')
                for task in tasks:
                    dot.node(task["id"], f"{task['id']}\n{task.get('tool', '')}")
                for edge in deps:
                    dot.edge(edge["source"], edge["target"])
                st.graphviz_chart(dot)

# Capture user query
query = st.chat_input("Enter factual, tool, or reasoning query (e.g. Compare Han-Wu age to Julius Caesar)...")

if query:
    # Append user query to chat history
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
        
    # Process execution and response via WebSocket
    with st.chat_message("assistant"):
        thought_placeholder = st.empty()
        dag_placeholder = st.empty()
        response_placeholder = st.empty()
        
        thoughts = []
        task_dag = None
        
        try:
            # Establish WebSocket connection
            ws = websocket.create_connection(ws_url)
            ws.send(json.dumps({"query": query}))
            
            while True:
                try:
                    raw_msg = ws.recv()
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type")
                    
                    if msg_type == "thought":
                        thoughts.append(msg.get("content", ""))
                        # Stream thoughts continuously
                        thought_block = "\n".join([f"* {t}" for t in thoughts])
                        thought_placeholder.markdown(f"#### 🧠 Agent Thoughts\n{thought_block}")
                        
                    elif msg_type == "classification":
                        tier = msg.get("tier", "")
                        thoughts.append(f"Query routed to tier: **{tier}**")
                        thought_block = "\n".join([f"* {t}" for t in thoughts])
                        thought_placeholder.markdown(f"#### 🧠 Agent Thoughts\n{thought_block}")
                        
                    elif msg_type == "dag":
                        task_dag = msg.get("content", {})
                        tasks = task_dag.get("tasks", [])
                        deps = task_dag.get("dependencies", [])
                        
                        if tasks:
                            # Render Graphviz DOT Graph
                            dot = graphviz.Digraph(comment='Task DAG')
                            dot.attr(bgcolor='transparent', rankdir='LR')
                            dot.attr('node', shape='box', style='filled,rounded', color='#3b82f6', fontcolor='white', fillcolor='#1e293b')
                            dot.attr('edge', color='#64748b')
                            for task in tasks:
                                dot.node(task["id"], f"{task['id']}\n{task.get('tool', '')}")
                            for edge in deps:
                                dot.edge(edge["source"], edge["target"])
                            
                            with dag_placeholder.container():
                                st.markdown("#### 🗺️ Planned Task DAG")
                                st.graphviz_chart(dot)
                                
                    elif msg_type == "final_response":
                        content = msg.get("content", "")
                        response_placeholder.markdown(content)
                        # Save final output to chat history
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": content,
                            "dag": task_dag
                        })
                        break
                        
                except websocket.WebSocketConnectionClosedException:
                    st.error("WebSocket connection closed unexpectedly.")
                    break
                    
            ws.close()
            
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend at {ws_url}. Check if the server is running. Error: {e}")
