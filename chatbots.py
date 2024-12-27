import os
import gradio as gr
from google.api_core import exceptions, retry
import google.generativeai as genai
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

@retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
def analyze_sentiment_with_retry(text):
    sentiment_analysis_prompt = f"Analise o sentimento do seguinte texto: \"{text}\""
    response = chat.send_message(sentiment_analysis_prompt)
    return response.text

def gradio_wrapper(message, _history):
    text = message["text"]
    uploaded_files = []
    
    for file_info in message["files"]:
        file_path = file_info["path"]
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    file_content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    file_content = file.read()
        
        uploaded_files.append(file_content)
    
    results = []
    
    if text:
        try:
            results.append(analyze_sentiment_with_retry(text))
        except Exception as e:
            results.append(f"Erro ao analisar o texto: {str(e)}")
    
    for content in uploaded_files:
        try:
            results.append(analyze_sentiment_with_retry(content))
        except Exception as e:
            results.append(f"Erro ao analisar o arquivo: {str(e)}")
    
    return "\n".join(results)

# Cria a interface Gradio
chatInterface = gr.ChatInterface(fn=gradio_wrapper, multimodal=True)
chatInterface.launch()
