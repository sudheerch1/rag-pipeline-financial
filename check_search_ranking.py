from query_graph import run_search

state = {
    "question": "What was Microsoft total revenue for fiscal years 2022, 2023, and 2024?",
    "hyde_text": "What was Microsoft total revenue for fiscal years 2022, 2023, and 2024?",
    "company": "microsoft"
}

result = run_search(state)
for c in result["chunks"]:
    print(round(c["score"], 3), "|", c["source"], "|", c["text"][:80].replace(chr(10), " "))
