import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

EXAMPLES_PATH = "email_examples.json"
RETRIEVER_NAME = "sentence-transformers/all-mpnet-base-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-3B-Instruct"
TOP_K = 3


def load_examples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class EmailRAG:
    def __init__(self):
        examples = load_examples(EXAMPLES_PATH)
        self.examples = examples

        print(f"[1/3] Loading retriever: {RETRIEVER_NAME}")
        self.retriever = SentenceTransformer(RETRIEVER_NAME)

        print("[2/3] Encoding example pool...")
        self.embeddings = self.retriever.encode(
            [ex["input"] for ex in examples],
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(f"[3/3] Loading generator: {GENERATOR_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print("Ready.\n")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
        q_emb = self.retriever.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.examples[i], float(scores[i])) for i in top_idx]

    def build_prompt(self, query: str, retrieved: list[tuple[dict, float]]) -> str:
        examples_block = ""
        for i, (ex, _) in enumerate(retrieved, 1):
            examples_block += (
                f"### Example {i}\n"
                f"Situation: {ex['input']}\n"
                f"Email:\n{ex['output']}\n\n"
            )

        return (
            "You are an assistant that helps university students write polite and "
            "professional academic emails to their professors.\n\n"
            "Here are similar example emails for reference:\n\n"
            f"{examples_block}"
            "---\n"
            "Now write a professional academic email for the following situation. "
            "Sign off with 'ooo' as the sender's name.\n\n"
            f"Situation: {query}\n"
            "Email:"
        )

    def generate(
        self,
        query: str,
        top_k: int = TOP_K,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        retrieved = self.retrieve(query, top_k)
        prompt = self.build_prompt(query, retrieved)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that helps university students write polite "
                    "and professional academic emails to their professors."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return {
            "query": query,
            "retrieved": [
                {
                    "category": ex["category"],
                    "input": ex["input"],
                    "similarity": round(score, 4),
                }
                for ex, score in retrieved
            ],
            "email": generated,
        }


def main():
    rag = EmailRAG()

    print("=== Academic Email Generator ===")
    print("Describe your situation in English. Type 'quit' to exit.\n")

    while True:
        query = input("Situation: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        result = rag.generate(query)

        print("\n--- Retrieved Examples (top-3) ---")
        for i, ex in enumerate(result["retrieved"], 1):
            print(
                f"  [{i}] [{ex['category']}] score={ex['similarity']:.4f}  "
                f"{ex['input'][:80]}{'...' if len(ex['input']) > 80 else ''}"
            )

        print("\n--- Generated Email ---")
        print(result["email"])
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
