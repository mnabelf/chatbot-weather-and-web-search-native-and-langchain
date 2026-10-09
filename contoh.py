import os

from openai import OpenAI
import json
import requests

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")
)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Ambil info cuaca untuk sebuah kota",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "Nama kota"}
        },
        "required": ["city"]
      }
    }
  },
{
    "type": "function",
    "function": {
      "name": "get_time",
      "description": "Ambil waktu saat ini di sebuah kota",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }
]

def get_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric&lang=id"
    response = requests.get(url.format(city=city, OPENWEATHER_KEY=OPENWEATHER_KEY))
    data = response.json()

    if response.status_code != 200:
        return f"Gagal ambil cuaca untuk {city}: {data.get('message', 'error tidak diketahui')}"

    suhu = data["main"]["temp"]
    deskripsi = data["weather"][0]["description"]
    kelembapan = data["main"]["humidity"]
    return f"Cuaca di {city}: {deskripsi}, suhu {suhu}°C, kelembapan {kelembapan}%"

def get_time(city):
    return f"Waktu di {city}: 11:21 WIB"

available_tools = {
    "get_weather": get_weather,
    "get_time": get_time
}

messages=[
    {"role": "user", "content": "Cuaca di Bandung hari ini gimana? dan jam berapa?"}
]
while True:
    response = client.chat.completions.create(
    model="minimax/minimax-m3:free",
    messages=messages,
    tools=tools
    )

    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        print("Jawaban Akhir:", message.content)
        break

    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"-> Model minta pakai tool: {func_name}({args})")

        result = available_tools[func_name](**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })