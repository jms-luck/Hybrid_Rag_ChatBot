def generate_answer(docs, question,PROMPT,client, deployment_name):
    """Generate answer using Azure OpenAI."""
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = PROMPT.format(
        context=context,
        question=question
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Use your Azure deployment name
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
