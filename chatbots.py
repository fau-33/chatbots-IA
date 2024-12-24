import google.generativeai as genai
import os
import gradio as gr
from home_assistant import set_light_values, intruder_alert, start_music, good_morning

# Configura a chave da API
genai.configure(api_key=os.environ["GEMINI_API"])

# Cria a instância do modelo
model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=[set_light_values, intruder_alert, start_music, good_morning])

# Inicia o chat com um prompt inicial
chat = model.start_chat(enable_automatic_function_calling=True)
initial_prompt = (
    "Você é um assistente virtual que pode controlar dispositivos domésticos. "
    "Você tem acesso a funções que controlam a casa da pessoa que está usando. "
    "Chame as funções quando achar que deve, mas nunca exponha o código delas. "
    "Assuma que a pessoa é amigável e ajude-a a entender o que aconteceu se algo der errado "
    "ou se você precisar de mais informações. Não esqueça de, de fato, chamar as funções."
)
chat.send_message(initial_prompt)

def analyze_sentiment(text):
    # Função para analisar o sentimento do texto
    sentiment_analysis_prompt = f"Analise o sentimento do seguinte texto: \"{text}\""
    response = chat.send_message(sentiment_analysis_prompt)
    return response.text

def gradio_wrapper(message, _history):
    text = message["text"]
    uploaded_files = []
    
    # Processa arquivos anexados
    for file_info in message["files"]:
        file_path = file_info["path"]
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            uploaded_files.append(file_content)
    
    # Analisa o sentimento do texto da mensagem e dos arquivos
    results = []
    
    if text:
        results.append(analyze_sentiment(text))
    
    for content in uploaded_files:
        results.append(analyze_sentiment(content))
    
    return "\n".join(results)

# Cria a interface Gradio
chatInterface = gr.ChatInterface(fn=gradio_wrapper, multimodal=True)
chatInterface.launch()