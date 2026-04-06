import ollama
from langsmith import traceable

MODEL = "qwen3:1.7b"

# --- 1. Tools (Functions with clear docstrings for the LLM) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog. 
    Use this immediately whenever a product name is mentioned."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    price = prices.get(product.lower(), 0.0)
    print(f"    [Executing Tool] get_product_price('{product}') -> {price}")
    return price

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Calculate the final price after a discount. Tiers: bronze, silver, gold."""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    pct = discount_percentages.get(discount_tier.lower(), 0)
    final = round(float(price) * (1 - pct / 100), 2)
    print(f"    [Executing Tool] apply_discount({price}, '{discount_tier}') -> {final}")
    return final

tools_map = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# --- 2. The Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # We use a system prompt that explicitly tells it to use tools first.
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a shopping assistant. "
                "RULE: You do not know prices. You MUST call get_product_price "
                "to find the price of an item before doing anything else."
            )
        },
        {"role": "user", "content": question},
    ]

    print(f"Question: {question}\n" + "="*60)

    for i in range(5):
        # Native Ollama tool call
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=[get_product_price, apply_discount],
            options={"temperature": 0} # Zero temp is key for tool reliability
        )

        # Add assistant's thought/intent to history
        messages.append(response['message'])

        # If no tool calls, it's the final answer
        if not response['message'].get('tool_calls'):
            print(f"\nFinal Answer: {response['message']['content']}")
            return response['message']['content']

        # Process tool calls (Ollama can return multiple in one go)
        for tool in response['message']['tool_calls']:
            name = tool['function']['name']
            args = tool['function']['arguments']
            
            result = tools_map[name](**args)
            
            # Feed the tool result back to the model
            messages.append({
                "role": "tool",
                "content": str(result),
                "name": name 
            })

if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")