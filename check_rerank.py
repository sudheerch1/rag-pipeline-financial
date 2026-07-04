from query_graph import run_search, run_rerank

state = {
    "question": "What was Microsoft total revenue for fiscal years 2022, 2023, and 2024?",
    "hyde_text": "What was Microsoft total revenue for fiscal years 2022, 2023, and 2024?",
    "company": "microsoft"
}

state = run_search(state)
state = run_rerank(state)

for c in state["chunks"]:
    print(round(c["score"], 3), "|", c["source"], "|", c["text"][:80].replace(chr(10), " "))
