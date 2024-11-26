import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Streamlit App Configuration
st.set_page_config(page_title="Math Solver & Knowledge Assistant", page_icon="🧮", layout="centered")

# App Title and Description
st.title("🧮 Math Solver & Knowledge Assistant")
st.markdown(
    """
    Enter a math problem or knowledge-based question, and let the assistant solve it for you!  
    Powered by **LangChain** and **Groq models**, this tool combines math problem-solving with Wikipedia search. 😊
    """
)

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Groq API Key", value="", type="password", help="Enter your Groq API Key.")
    st.markdown("---")
    model_options = ["gemma2-9b-it", "gemma-7b-it", "llama3-groq-70b-8192-tool-use-preview"]
    selected_model = st.selectbox("Select Model", model_options, help="Choose the model to process your queries.")
    st.markdown("---")
    st.markdown("**Need Help?**")
    st.markdown("[📖 Learn more about Groq API](https://groq.com/)")
    st.markdown("[❓ FAQ](https://groq.com/faq)")

# Initialize Model
if groq_api_key.strip():
    llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)
else:
    st.info("⚠️ Please provide a valid Groq API key to continue.")
    st.stop()

# Initialize Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for various topics."
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve mathematical expressions."
)

# Agent Initialization
tools = [wikipedia_tool, calculator_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Math Solver & Knowledge Assistant. How can I help you today?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input Section
user_query = st.chat_input("Enter your question here...")

if user_query:
    # Append User Query to Chat History
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    # Process Query
    with st.spinner("⏳ Processing your query..."):
        try:
            result = agent.run(user_query)
            
            # Append Assistant's Response to Chat History
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.chat_message("assistant").write(result)
        except Exception as e:
            error_message = f"❌ An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)

# Download Chat History
if st.session_state.messages:
    chat_history = "\n\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
    )
    st.download_button(
        label="📥 Download Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain",
    )

# Example Questions Section
st.markdown("---")
st.subheader("📋 Example Questions")
st.markdown(
    """
    - **Math**: "What is 3x + 5 = 11?"
    - **Knowledge**: "Who was Albert Einstein?"
    - **Science**: "Explain photosynthesis."
    """
)
