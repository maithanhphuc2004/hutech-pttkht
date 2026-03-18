# Agent-based Systems với LangChain/LangGraph

## 1. Giới thiệu

### 1.1. Giới hạn của RAG đơn giản

RAG cơ bản: Query → Retrieve → Generate  
**Vấn đề**: Chỉ xử lý được 1 câu hỏi trực tiếp, không có khả năng:
- Lập kế hoạch nhiều bước
- Sử dụng tools (search web, tính toán, gọi API)
- Tự sửa lỗi và retry
- Duy trì context qua nhiều turns

### 1.2. Agent là gì?

**Agent** = LLM + Planning + Tool Use + Memory

```
Agent workflow:
1. Nhận task từ user
2. Lập kế hoạch (plan)
3. Chọn tool phù hợp
4. Thực thi action
5. Quan sát kết quả
6. Lặp lại đến khi hoàn thành
```

**Ví dụ**:
```
User: "Tìm nghiên cứu mới nhất về AI trong y tế, tóm tắt và gửi email cho team"

Agent thinking:
1. Search web for "AI in healthcare 2024 research"
2. Retrieve top 3 papers
3. Summarize using LLM
4. Draft email
5. Send via email API
```

### 1.3. LangChain vs LangGraph

| Framework | Mô tả | Use case |
|-----------|-------|----------|
| **LangChain** | Sequential chains, pre-defined flow | Simple workflows, Q&A bots |
| **LangGraph** | Graph-based, cyclic workflows | Complex agents, multi-step reasoning |

**LangGraph** = LangChain + State Graph → Flexible hơn, kiểm soát flow tốt hơn.

---

## 2. LangChain Basics

### 2.1. Cài đặt

```bash
pip install langchain
pip install langchain-google-genai
pip install langchain-community
pip install faiss-cpu
pip install duckduckgo-search  # Web search tool
```

### 2.2. Chains đơn giản

#### a) LLM Chain

```python
# simple_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Viết 1 đoạn ngắn về {topic} bằng tiếng Việt:"
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(topic="trí tuệ nhân tạo")
print(result)
```

#### b) Sequential Chain

```python
from langchain.chains import SequentialChain

# Chain 1: Generate topic ideas
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["subject"],
        template="Gợi ý 3 ý tưởng nghiên cứu về {subject}:"
    ),
    output_key="ideas"
)

# Chain 2: Expand first idea
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["ideas"],
        template="Mở rộng ý tưởng đầu tiên trong:\n{ideas}"
    ),
    output_key="expanded"
)

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["subject"],
    output_variables=["ideas", "expanded"]
)

result = overall_chain({"subject": "machine learning"})
print(result["expanded"])
```

### 2.3. RAG với LangChain

```python
# langchain_rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("data/document.pdf")
documents = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 3. Embedding + Vector DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine"
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
answer = qa_chain.run("Các giai đoạn giáo dục cơ bản là gì?")
print(answer)
```

---

## 3. Agents với Tools

### 3.1. Định nghĩa Tools

Tool = Hàm Python + mô tả (để LLM biết khi nào dùng).

```python
# agent_tools.py
from langchain.tools import Tool
from duckduckgo_search import DDGS
import math

# Tool 1: Web Search
def search_web(query):
    results = DDGS().text(query, max_results=3)
    return "\n".join([r["body"] for r in results])

search_tool = Tool(
    name="WebSearch",
    func=search_web,
    description="Tìm kiếm thông tin trên web. Input: câu hỏi tìm kiếm."
)

# Tool 2: Calculator
def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "Lỗi tính toán"

calc_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Thực hiện phép tính. Input: biểu thức toán học (vd: '2+2', 'math.sqrt(16)')."
)

# Tool 3: RAG từ vector DB
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local("vector_index", embeddings)

def rag_search(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

rag_tool = Tool(
    name="KnowledgeBase",
    func=rag_search,
    description="Tìm kiếm trong knowledge base nội bộ. Input: câu hỏi."
)

tools = [search_tool, calc_tool, rag_tool]
```

### 3.2. Initialize Agent

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct agent
    verbose=True,  # Show thinking process
    max_iterations=5
)

# Test
response = agent.run("Tìm giá Bitcoin hiện tại và nhân với 2")
print(response)
```

**Agent reasoning**:
```
> Entering new AgentExecutor chain...
I need to find the current Bitcoin price.

Action: WebSearch
Action Input: "Bitcoin price now"

Observation: Bitcoin is trading at $45,000...

Thought: Now I need to multiply by 2.

Action: Calculator
Action Input: "45000 * 2"

Observation: 90000

Thought: I now know the final answer.

Final Answer: 90,000 USD
```

### 3.3. Custom Tool cho API

```python
import requests

def get_weather(city):
    """Get weather từ API"""
    # Giả sử dùng OpenWeatherMap
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    temp = data["main"]["temp"] - 273.15  # Kelvin to Celsius
    desc = data["weather"][0]["description"]
    
    return f"{city}: {temp:.1f}°C, {desc}"

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Lấy thông tin thời tiết của thành phố. Input: tên thành phố (tiếng Anh)."
)

tools.append(weather_tool)
```

---

## 4. LangGraph: State-based Agents

### 4.1. Tại sao cần LangGraph?

LangChain chains: Linear flow (A → B → C)  
**Vấn đề**: Không xử lý được:
- Conditional branching (nếu X thì làm Y, không thì Z)
- Loops (lặp lại action đến khi thành công)
- Parallel execution

**LangGraph** giải quyết bằng state graph.

### 4.2. Cài đặt

```bash
pip install langgraph
```

### 4.3. Khái niệm State Graph

```
State = Dict lưu trữ thông tin runtime

Nodes = Functions xử lý state

Edges = Luồng chuyển giữa các nodes
```

### 4.4. Ví dụ: Research Agent

Workflow:
1. Plan: Phân tích câu hỏi → list sub-questions
2. Research: Tìm kiếm từng sub-question
3. Synthesize: Tổng hợp câu trả lời
4. Critique: Đánh giá chất lượng → nếu chưa đủ, quay lại Research

```python
# research_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json

# Define State
class ResearchState(TypedDict):
    question: str
    sub_questions: List[str]
    research_results: List[str]
    answer: str
    quality_score: int
    iteration: int

# Node 1: Plan
def plan_node(state: ResearchState):
    question = state["question"]
    
    prompt = f"""
Phân tích câu hỏi sau thành 3 câu hỏi con để research:
{question}

Trả về JSON: {{"sub_questions": [...]}}
"""
    
    response = llm.predict(prompt)
    sub_questions = json.loads(response)["sub_questions"]
    
    return {
        **state,
        "sub_questions": sub_questions,
        "iteration": state.get("iteration", 0) + 1
    }

# Node 2: Research
def research_node(state: ResearchState):
    results = []
    
    for sub_q in state["sub_questions"]:
        # Web search
        search_result = search_web(sub_q)
        results.append(search_result)
    
    return {
        **state,
        "research_results": results
    }

# Node 3: Synthesize
def synthesize_node(state: ResearchState):
    prompt = f"""
Câu hỏi: {state['question']}

Kết quả research:
{chr(10).join(state['research_results'])}

Tổng hợp câu trả lời đầy đủ:
"""
    
    answer = llm.predict(prompt)
    
    return {
        **state,
        "answer": answer
    }

# Node 4: Critique
def critique_node(state: ResearchState):
    prompt = f"""
Đánh giá chất lượng câu trả lời (1-10):

Câu hỏi: {state['question']}
Câu trả lời: {state['answer']}

Trả về JSON: {{"score": X, "reason": "..."}}
"""
    
    response = llm.predict(prompt)
    critique = json.loads(response)
    
    return {
        **state,
        "quality_score": critique["score"]
    }

# Conditional Edge
def should_continue(state: ResearchState):
    """Nếu score < 7 và iteration < 3, research lại"""
    if state["quality_score"] < 7 and state["iteration"] < 3:
        return "research"
    else:
        return "end"

# Build Graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("plan", plan_node)
workflow.add_node("research", research_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("critique", critique_node)

# Add edges
workflow.set_entry_point("plan")
workflow.add_edge("plan", "research")
workflow.add_edge("research", "synthesize")
workflow.add_edge("synthesize", "critique")

# Conditional edge
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {
        "research": "research",  # Loop back
        "end": END
    }
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({
    "question": "AI sẽ thay đổi giáo dục như thế nào trong 5 năm tới?",
    "iteration": 0
})

print(result["answer"])
```

### 4.5. Visualize Graph

```python
from IPython.display import Image, display

# Trong Jupyter Notebook
display(Image(app.get_graph().draw_mermaid_png()))
```

Output:
```
plan → research → synthesize → critique
         ↑                         ↓
         └─────────────────────────┘ (if score < 7)
```

---

## 5. Advanced Agent Patterns

### 5.1. ReAct (Reasoning + Acting)

Agent luân phiên giữa **Thought** và **Action**.

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Load ReAct prompt từ LangChain hub
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

result = agent_executor.invoke({
    "input": "Tìm 3 nghiên cứu gần đây về RAG và tóm tắt ý chính"
})
```

### 5.2. Plan-and-Execute

Agent lập kế hoạch trước, sau đó execute từng bước.

```python
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True
)

result = agent.run("Nghiên cứu về climate change ở Việt Nam, tóm tắt và lưu vào file")
```

**Workflow**:
```
1. Planner: Tạo plan
   Step 1: Search "climate change Vietnam research"
   Step 2: Summarize findings
   Step 3: Save to file

2. Executor: Thực thi từng step
   - Execute Step 1 → output1
   - Execute Step 2 (với input = output1) → output2
   - Execute Step 3 (với input = output2) → DONE
```

### 5.3. Multi-Agent Collaboration

Nhiều agents chuyên biệt làm việc cùng nhau.

```python
# multi_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    task: str
    research_output: str
    code_output: str
    review_output: str

# Agent 1: Researcher
def researcher_agent(state):
    # Search web + summarize
    results = search_web(state["task"])
    summary = llm.predict(f"Tóm tắt:\n{results}")
    return {**state, "research_output": summary}

# Agent 2: Coder
def coder_agent(state):
    # Generate code based on research
    prompt = f"""
Dựa vào nghiên cứu:
{state['research_output']}

Viết code Python để giải quyết: {state['task']}
"""
    code = llm.predict(prompt)
    return {**state, "code_output": code}

# Agent 3: Reviewer
def reviewer_agent(state):
    # Review code
    prompt = f"""
Review code sau:
{state['code_output']}

Đưa ra nhận xét và cải tiến.
"""
    review = llm.predict(prompt)
    return {**state, "review_output": review}

# Build graph
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", END)

app = workflow.compile()

result = app.invoke({
    "task": "Tạo script Python để scrape tin tức từ VnExpress"
})

print(result["review_output"])
```

---

## 6. Memory trong Agents

### 6.1. Conversation Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Turn 1
agent.run("Tìm giá vàng hôm nay")

# Turn 2 (agent nhớ context)
agent.run("So sánh với tuần trước")  # Agent biết "tuần trước" của vàng
```

### 6.2. Summary Memory

Tự động tóm tắt lịch sử khi quá dài.

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)
```

### 6.3. Persistent Memory (Database)

```python
from langchain.memory import SQLChatMessageHistory

# Lưu vào SQLite
message_history = SQLChatMessageHistory(
    session_id="user_123",
    connection_string="sqlite:///chat_history.db"
)

memory = ConversationBufferMemory(
    chat_memory=message_history,
    memory_key="chat_history"
)
```

---

## 7. Production Deployment

### 7.1. API với FastAPI

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/agent")
async def run_agent(query: Query):
    # Load memory theo session_id
    message_history = SQLChatMessageHistory(
        session_id=query.session_id,
        connection_string="sqlite:///chat_history.db"
    )
    
    memory = ConversationBufferMemory(chat_memory=message_history)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory
    )
    
    result = agent.run(query.question)
    
    return {"answer": result}

# Run: uvicorn app:app --reload
```

### 7.2. Async Agents

```python
from langchain.callbacks import AsyncIteratorCallbackHandler
import asyncio

async def run_agent_async(question):
    callback = AsyncIteratorCallbackHandler()
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback]
    )
    
    # Run trong background
    task = asyncio.create_task(agent.arun(question))
    
    # Stream thinking process
    async for token in callback.aiter():
        print(token, end="", flush=True)
    
    result = await task
    return result
```

### 7.3. Error Handling

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,  # Tự động retry khi parsing lỗi
    max_execution_time=60,  # Timeout 60s
)

try:
    result = agent_executor.invoke({"input": "..."})
except Exception as e:
    print(f"Agent failed: {e}")
    # Fallback hoặc retry
```

---

## 8. Use Cases thực tế

### 8.1. Customer Support Agent

```python
# Support agent với tools:
# - search_kb: Tìm trong knowledge base
# - create_ticket: Tạo ticket nếu không giải quyết được
# - check_order: Tra cứu đơn hàng

def check_order(order_id):
    # Call API
    return f"Đơn hàng {order_id}: Đang giao"

support_tools = [
    Tool(name="SearchKB", func=rag_search, description="..."),
    Tool(name="CheckOrder", func=check_order, description="..."),
    # ... more tools
]

support_agent = initialize_agent(tools=support_tools, llm=llm, ...)

# Customer: "Đơn hàng 12345 của tôi đến đâu rồi?"
# Agent: CheckOrder("12345") → "Đang giao"
```

### 8.2. Data Analysis Agent

```python
import pandas as pd

def analyze_data(question):
    df = pd.read_csv("sales.csv")
    
    # Agent có thể:
    # - Tự viết pandas code
    # - Visualize
    # - Statistical tests
    
    return df.describe()

tools = [
    Tool(name="AnalyzeCSV", func=analyze_data, ...),
    Tool(name="PlotChart", func=..., ...),
]

data_agent = initialize_agent(tools=tools, llm=llm, ...)

# User: "Doanh thu Q4 tăng hay giảm so với Q3?"
# Agent: AnalyzeCSV → Compare → PlotChart → Answer
```

### 8.3. Research Assistant

```python
# Agent workflow:
# 1. Search web cho papers
# 2. Download PDFs
# 3. Extract text
# 4. Summarize each
# 5. Compare findings
# 6. Generate report

research_agent = create_react_agent(llm, tools, prompt)

result = research_agent.run(
    "So sánh các phương pháp fine-tuning LLM: LoRA vs Full Fine-tuning vs Prompt Tuning"
)
```

---

## 9. Debugging & Monitoring

### 9.1. LangSmith (Official LangChain monitoring)

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"

# Tất cả agent runs sẽ được log lên LangSmith dashboard
```

### 9.2. Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompt: {prompts[0][:100]}...")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool {serialized['name']} started with input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"Tool output: {output[:100]}...")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    callbacks=[MyCallbackHandler()],
    ...
)
```

---

## 10. Best Practices

### 10.1. Tool Design

- **Atomic**: 1 tool = 1 chức năng rõ ràng
- **Description**: Viết mô tả chi tiết để LLM hiểu khi nào dùng
- **Error handling**: Trả về error message rõ ràng
- **Idempotent**: Gọi nhiều lần không gây side effect

### 10.2. Prompt Engineering cho Agents

```python
custom_react_prompt = """
You are a helpful AI assistant.

You have access to these tools:
{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take (one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer in Vietnamese

IMPORTANT:
- Always think in Vietnamese
- Double-check your calculations
- If unsure, use WebSearch tool

Question: {input}
{agent_scratchpad}
"""
```

### 10.3. Testing

```python
import pytest

def test_agent_web_search():
    agent = initialize_agent(tools=[search_tool], llm=llm, ...)
    result = agent.run("Bitcoin price now")
    assert "USD" in result or "$" in result

def test_agent_calculator():
    agent = initialize_agent(tools=[calc_tool], llm=llm, ...)
    result = agent.run("What is 123 * 456?")
    assert "56088" in result
```

---

## 11. Bài tập thực hành

### Bài 1: Simple ReAct Agent
Tạo agent với 3 tools: WebSearch, Calculator, KnowledgeBase.  
Test với các câu hỏi phức tạp yêu cầu kết hợp nhiều tools.

### Bài 2: Research Agent với LangGraph
Implement research agent với loop:  
Plan → Research → Synthesize → Critique → (nếu không đạt) → Research lại

### Bài 3: Multi-Agent System
3 agents: Researcher, Writer, Editor.  
Input: Topic → Output: Bài báo hoàn chỉnh.

### Bài 4: Production API
Deploy agent as FastAPI endpoint với:
- Session management
- Error handling
- Async execution
- Rate limiting

---

## 12. Resources

### Documentation
- **LangChain**: https://python.langchain.com/docs
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com/

### Learning
- **LangChain Cookbook**: https://github.com/langchain-ai/langchain/tree/master/cookbook
- **LangGraph Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples

### Communities
- **Discord**: LangChain Discord
- **GitHub Discussions**: Issues & discussions

---

## Kết luận

Agent-based systems cho phép LLM tự động hóa workflows phức tạp, sử dụng tools, và reasoning qua nhiều bước. LangGraph mang lại flexibility cao hơn LangChain chains truyền thống.

**Key takeaways**:
- Agents = LLM + Tools + Planning + Memory
- LangGraph: State-based, support cycles và conditionals
- Production: FastAPI + async + monitoring

**Next step**: Evaluation & Monitoring cho RAG systems để đảm bảo chất lượng.
