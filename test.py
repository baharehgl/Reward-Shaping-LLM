from llm_shaping import compute_potential, llm_logs
print("=== sanity check ===")
llm_logs.clear()
φ = compute_potential((0.1, 0.2, 0.3, 0.4))
print("Returned φ =", φ)
print("Logs:", llm_logs)