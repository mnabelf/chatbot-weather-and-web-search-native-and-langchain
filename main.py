from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
import requests
import os

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="minimax/minimax-m3:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

@tool
def get_weather(city: str) -> str:
    """Ambil info cuaca untuk sebuah kota"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": os.getenv("OPENWEATHER_API_KEY"), "units": "metric", "lang": "id"}
    response = requests.get(url, params=params)
    data = response.json()
    if response.status_code != 200:
        return f"Gagal ambil cuaca: {data.get('message')}"
    return f"Cuaca di {city}: {data['weather'][0]['description']}, suhu {data['main']['temp']}°C"

search_tool = TavilySearch(max_results=3)

agent = create_agent(
    model=llm,
    tools=[get_weather, search_tool],
    system_prompt="Kamu asisten yang bisa cek cuaca dan cari info terkini dari internet."
)

app = FastAPI()

sessions: dict[str, list] = {}

class chatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: chatRequest):
    history = sessions.get(request.session_id, [])
    history.append({"role": "user", "content": request.message})

    result = agent.invoke({"messages": history})
    ai_message = result["messages"][-1]
    history.append({"role": "assistant", "content": ai_message.content})
    sessions[request.session_id] = history
    return {"reply": ai_message.content}