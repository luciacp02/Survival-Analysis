from ollama import Client

def generate_explanation(graph_data, model_type):
    client = Client(
        host='http://127.0.0.1:11434',  
        headers={'x-some-header': 'some-value'}
    )

    # Construir el prompt según el tipo de gráfico o modelo
    if model_type == 'kaplan-meier':
        prompt = f"Por favor, explica en español esta gráfica de Kaplan-Meier: {kaplan_img}."
    elif model_type == 'log-rank':
        prompt = f"Me puedes dar una conclusión breve de los resultados obtenidos en Log Rank: {logrank_table}."
    elif model_type == 'cox-regression':
        prompt = f"Explica los siguientes resultados de la regresión de Cox: {cox_table}."

    # Enviar el prompt al modelo
    response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    explanation = response['message']['content']
    max_length = 3000
    if len(explanation) > max_length:  
        explanation_parts = [explanation[i:i + 1000] for i in range(0, len(explanation), 1000)]
        return '\n\n'.join(explanation_parts) 
    return explanation
