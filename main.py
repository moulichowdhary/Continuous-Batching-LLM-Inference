

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ============================================================
#  PART 1: Model Wrapper with KV Cache
# ============================================================

class GPT2Runner:
    """
    Wraps HuggingFace GPT-2 with explicit prefill/decode phases
    and KV cache management — just like a real inference server.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        print(f"[Model] Loading {model_name}...")
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Loaded: {n_params:,} parameters on {device}")

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        PREFILL: Process entire prompt, return logits + KV cache.

        This runs all prompt tokens through the model in one pass.
        The KV cache stores the keys/values so we don't recompute them.
        """
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,         # Build KV cache
        )
        # outputs.logits:          [batch, seq_len, vocab_size]
        # outputs.past_key_values: tuple of (key, value) per layer
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Tuple,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        DECODE: Process ONE new token, reusing KV cache.

        This is the fast path — only 1 token goes through the model,
        while all previous context is retrieved from the cache.
        """
        outputs = self.model(
            input_ids=input_ids,          # Just 1 token: [batch, 1]
            past_key_values=past_key_values,  # Reuse cached K, V
            use_cache=True,               # Continue building cache
        )
        return outputs.logits, outputs.past_key_values


# ============================================================
#  PART 2: Request & Sampling
# ============================================================

@dataclass
class Request:
    """A generation request with all its state."""
    id: int
    prompt: str
    prompt_ids: List[int]
    max_tokens: int = 60
    temperature: float = 0.7
    top_k: int = 50

    # State
    generated_ids: List[int] = field(default_factory=list)
    kv_cache: Optional[Tuple] = None
    is_finished: bool = False
    prefill_done: bool = False
    created_at: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None


def sample_token(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 50) -> int:
    """Sample from logits with temperature and top-k."""
    if temperature < 1e-6:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits[logits < min_val] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1).item()
    return token


# ============================================================
#  PART 3: Continuous Batching Server
# ============================================================

class ContinuousBatchingServer:
    """
    Inference server with continuous batching.

    The core loop:
    1. Remove finished requests from the running batch
    2. Prefill waiting requests if there's room
    3. Decode one token for every running request
    4. Repeat until all done

    Requests enter and exit the batch dynamically —
    no request waits for the whole batch to finish.
    """

    def __init__(self, model_name: str = "gpt2", max_batch_size: int = 4, device: str = "cpu"):
        self.runner = GPT2Runner(model_name, device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = device
        self.max_batch_size = max_batch_size

        self.waiting_queue: deque = deque()
        self.running: List[Request] = []
        self.completed: List[Request] = []
        self.request_counter = 0

        # EOS token
        self.eos_id = self.tokenizer.eos_token_id

    def add_request(self, prompt: str, max_tokens: int = 60, temperature: float = 0.7):
        """Submit a generation request."""
        ids = self.tokenizer.encode(prompt)
        req = Request(
            id=self.request_counter,
            prompt=prompt,
            prompt_ids=ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.request_counter += 1
        self.waiting_queue.append(req)
        return req

    def prefill_request(self, req: Request):
        """Prefill: process full prompt, cache KV, get first token."""
        input_ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)

        logits, kv_cache = self.runner.prefill(input_ids)

        # Sample first token from the last position
        next_token = sample_token(logits[0, -1, :], req.temperature, req.top_k)

        req.kv_cache = kv_cache
        req.generated_ids.append(next_token)
        req.prefill_done = True
        req.first_token_time = time.time()

        return next_token

    def decode_one_token(self, req: Request) -> int:
        """Decode: generate exactly one token using KV cache."""
        last_token = torch.tensor([[req.generated_ids[-1]]], dtype=torch.long, device=self.device)

        logits, new_kv = self.runner.decode_step(last_token, req.kv_cache)

        next_token = sample_token(logits[0, -1, :], req.temperature, req.top_k)

        req.kv_cache = new_kv
        req.generated_ids.append(next_token)

        # Check stop conditions
        if next_token == self.eos_id or len(req.generated_ids) >= req.max_tokens:
            req.is_finished = True

        return next_token

    def run_step(self) -> bool:
        """
        One step of the continuous batching loop.

        This is where the magic happens:
        - Finished requests leave
        - New requests get prefilled
        - All running requests decode one token
        """
        # ── Remove finished requests ──
        newly_done = [r for r in self.running if r.is_finished]
        for r in newly_done:
            self.completed.append(r)
            text = self.tokenizer.decode(r.generated_ids, skip_special_tokens=True)
            latency = time.time() - r.created_at
            ttft = r.first_token_time - r.created_at
            print(f"\n  ✓ Request {r.id} DONE ({len(r.generated_ids)} tokens, "
                  f"TTFT={ttft*1000:.0f}ms, total={latency:.2f}s)")
            print(f"    Prompt: \"{r.prompt}\"")
            print(f"    Output: \"{text}\"\n")

        self.running = [r for r in self.running if not r.is_finished]

        # ── Prefill new requests if room ──
        while self.waiting_queue and len(self.running) < self.max_batch_size:
            req = self.waiting_queue.popleft()
            token = self.prefill_request(req)
            tok_str = self.tokenizer.decode([token])
            print(f"  [Prefill] Req {req.id}: \"{req.prompt}\" "
                  f"→ first token: \"{tok_str}\"")
            self.running.append(req)

        # ── Decode one token for each running request ──
        if self.running:
            for req in self.running:
                token = self.decode_one_token(req)
            # Show streaming progress
            for req in self.running:
                partial = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
                n = len(req.generated_ids)
                print(f"    Req {req.id} [{n:2d} tokens]: {partial[:80]}", end="")
                if len(partial) > 80:
                    print("...", end="")
                print()
            return True

        return bool(newly_done)

    def has_work(self) -> bool:
        return bool(self.waiting_queue or self.running)

    def run_until_complete(self):
        """Run the continuous batching loop until all requests finish."""
        step = 0
        while self.has_work():
            step += 1
            print(f"  ── Step {step} ({len(self.running)} running, "
                  f"{len(self.waiting_queue)} waiting) ──")
            self.run_step()
        return step


# ============================================================
#  PART 4: Demo
# ============================================================

def main():
    print("=" * 70)
    print("  GPT-2 INFERENCE SERVER — PRETRAINED WEIGHTS")
    print("  Continuous Batching • KV Cache • Token-by-Token Generation")
    print("=" * 70)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize server
    server = ContinuousBatchingServer(
        model_name="gpt2",        # 124M parameter model
        max_batch_size=4,
        device=device,
    )

    # Submit requests — diverse prompts to see GPT-2 in action
    prompts = [
        # Knowledge / factual
        "The capital of Japan is",
        "Water boils at a temperature of",
        "Albert Einstein was famous for",
        "The largest ocean on Earth is",

        # Creative / storytelling
        "Once upon a time in a small village",
        "In a distant galaxy, the last human",
        "The dragon opened its eyes and saw",
        "Dear diary, today I discovered that",

        # Opinion / reasoning
        "The best programming language is",
        "The meaning of life is",
        "The secret to happiness is",
        "Artificial intelligence will",

        # Technical / science
        "Machine learning works by",
        "The speed of light is",
        "Quantum computers are different because",
        "Neural networks learn by",
    ]

    print(f"\nSubmitting {len(prompts)} requests (max_batch_size=4)...\n")
    for prompt in prompts:
        server.add_request(prompt, max_tokens=40, temperature=0.7)

    # Run!
    print("\n" + "─" * 70)
    print("  CONTINUOUS BATCHING IN ACTION")
    print("─" * 70 + "\n")

    start = time.time()
    steps = server.run_until_complete()
    elapsed = time.time() - start

    # ── Final Summary ──
    print("\n" + "─" * 70)
    print("  FINAL RESULTS")
    print("─" * 70 + "\n")

    total_tokens = 0
    for req in server.completed:
        text = server.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        total_tokens += len(req.generated_ids)
        ttft = req.first_token_time - req.created_at
        latency = time.time() - req.created_at
        print(f"  Request {req.id}:")
        print(f"    Prompt: \"{req.prompt}\"")
        print(f"    Output: \"{text}\"")
        print(f"    Tokens: {len(req.generated_ids)}, "
              f"TTFT: {ttft*1000:.0f}ms, Total: {latency:.2f}s\n")

    print("─" * 70)
    print(f"  Requests:    {len(server.completed)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Steps:       {steps}")
    print(f"  Wall time:   {elapsed:.2f}s")
    print(f"  Throughput:  {total_tokens / elapsed:.1f} tokens/sec")
    print(f"  Device:      {device}")
    print("─" * 70)

    print("""
HOW IT WORKS:
─────────────────────────────────────────────────────────────────
1. PRETRAINED GPT-2: Real 124M parameter model with learned weights.
   It actually "knows" language — outputs are coherent English.

2. PREFILL: Full prompt processed in one forward pass.
   KV cache built so we never recompute prompt attention.

3. DECODE: One token at a time. Each step only runs the NEW token
   through the model — all previous context via KV cache.

4. CONTINUOUS BATCHING: 4 requests run together. When one finishes
   (hits EOS or max tokens), a new request from the queue takes
   its slot immediately — no wasted compute.

5. STREAMING: You can see tokens appear one by one, just like
   ChatGPT. Each decode step adds exactly one token per request.
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
