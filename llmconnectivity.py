# 1. Install the exact library we need to talk to Groq
!pip install langchain-groq --quiet

# 2. Import the library
from langchain_groq import ChatGroq

# 3. Connect to the LLM Brain (llama3 is incredibly fast and free)
llm = ChatGroq(
    api_key="YOUR API KEY HERE",
    model="llama-3.3-70b-versatile"
)

from IPython.display import Markdown

# 4. Give the Brain a command
prompt = "Explain me how AI Agents work in short and concise short 3 bullets in simple language."
response = llm.invoke(prompt)

# 5. Print the LLM's raw response
print("🤖 AI SAYS:")
display(Markdown(response.content))
