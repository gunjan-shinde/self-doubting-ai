# modules/counterfactual_engine.py

from transformers import pipeline

# Load GPT-2 generator pipeline (you can later upgrade to ChatGPT via API if needed)
generator = pipeline("text-generation", model="gpt2", max_length=80, do_sample=True, top_k=50, top_p=0.95)

def generate_counterfactual(question):
    prompt = (
        "You're an AI assistant that is not sure about an answer.\n"
        "Reflect and suggest how things might change with more context.\n\n"
        f"Question: {question}\n"
        "AI (reflecting):"
    )

    response = generator(prompt, num_return_sequences=1)[0]["generated_text"]
    return response.strip().split("AI (reflecting):")[-1].strip()
