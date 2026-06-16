from email_generator_updated import EmailRAG

rag = EmailRAG()
result = rag.generate("I missed last Tuesday lecture due to a sudden illness and I need the lecture recording or slides")

print("\n--- Retrieved Examples (top-3) ---")
for i, ex in enumerate(result["retrieved"], 1):
    print(f"  [{i}] [{ex['category']}] score={ex['similarity']:.4f}  {ex['input'][:80]}")

print("\n--- Generated Email ---")
print(result["email"])
