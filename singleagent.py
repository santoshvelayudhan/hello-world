%pip install langchain_core langgraph langchain-groq --quiet



from langchain_core.tools import tool

@tool
def triage_ticket(issue_text: str) -> str:
    """Categorizes the IT issue based on the text using an LLM."""
    global llm # Access the global llm object
    prompt = f"Categorize the following IT issue: '{issue_text}'. Provide only the category."
    response = llm.invoke(prompt)
    return response.content.strip()

@tool
def route_to_team(category: str) -> str:
    """Finds the right IT team to fix the issue based on category using an LLM."""
    global llm # Access the global llm object
    prompt = f"Given the IT issue category '{category}', which team should it be assigned to? Provide only the assignment. "
    response = llm.invoke(prompt)
    return response.content.strip()

    from langchain.agents import create_agent

# 2. Import the library
from langchain_groq import ChatGroq

# 3. Connect to the LLM Brain (llama3 is incredibly fast and free)
llm = ChatGroq(
    api_key="your api key",
    model="llama-3.3-70b-versatile"
)

# 2. Group our tools from yesterday into a list
my_it_tools = [triage_ticket, route_to_team]

# 3. Create the Autonomous Agent!
agent_executor = create_agent(llm, tools=my_it_tools)

# 4. Give the Agent a simulated IT ticket
ticket_complaint = "I have been charged twice for the subscription."

print(f"📩 NEW TICKET RECEIVED: '{ticket_complaint}'\n")

# 5. Tell the agent to process the ticket
response = agent_executor.invoke({"messages": [("user", f"Please process this ticket: {ticket_complaint}")]})

# 6. Read the final output from the agent
print("🤖 AGENT FINAL REPORT:")
print(response["messages"][-1].content)
