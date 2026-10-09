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

conversation = []

print("Ketik 'exit' untuk keluar.\n")

while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "exit":
        break

    conversation.append({"role": "user", "content": user_input})

    result = agent.invoke({"messages": conversation})

    ai_message = result["messages"][-1]
    print("Agent:", ai_message.content, "\n", flush=True)

    conversation.append({"role": "user", "content": ai_message.content})