import google.generativeai as genai
import os
import gradio as gr

# Configura a chave da API
genai.configure(api_key=os.environ["GEMINI_API"])

# Cria a instância do modelo
model = genai.GenerativeModel("gemini-1.5-flash")

# Inicia o chat com um prompt inicial
chat = model.start_chat()
initial_prompt = "Você é um consultor que analisa sentimentos de textos. Por favor, envie suas mensagens ou arquivos de texto."
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