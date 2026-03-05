# tensor-ffn

A focused toolkit for training and testing FFN experts — part of the Tensor framework.

tensor-ffn does exactly two things: train an expert, and test whether it works.
No router. No gating. No multi-expert orchestration. That complexity lives elsewhere.

---

## What is an FFN Expert?

In a transformer, the Feed-Forward Network (FFN) is the part of each layer
responsible for storing and recalling factual knowledge. It is the memory
of the model — not the attention, which handles relationships between tokens.

In this project, an expert is a standalone FFN module trained to store
knowledge that the shared expert (the base model) does not have. It sits
alongside the base FFN, outputs a correction delta (an additive offset, not
a replacement), and is saved as an independent file on disk.

```
Base FFN (shared expert):   frozen, untouched — general knowledge
Expert FFN:                 your trained module — domain-specific knowledge
Combined at inference:      base + expert delta → better output
```

The base model is referred to as the shared expert throughout this codebase.
The expert you train here is always a delta on top of it, never a replacement.

---

## What This Repo Is Not

tensor-ffn has no router. Routing — deciding which expert fires on which
token — is a separate problem solved after you have proven your experts work.

Building a router before you know your experts are good is wasted effort.
tensor-ffn lets you validate expert quality first, cheaply, before committing
to the full MoE training pipeline.

---

## The Goal

Train a domain expert. Run it alongside the shared expert. Compare against
the shared expert alone. If the combined output is better, the expert is good.

```
Test 1 — shared expert only:     what does the base model know?
Test 2 — shared + your expert:   does the expert add correct knowledge?
```

If Test 2 consistently beats Test 1, you have a real expert worth keeping.
Only then does it make sense to invest in router training and multi-expert
orchestration. tensor-ffn saves you that time and compute cost upfront.

---

## Usage

tensor-ffn has three subcommands, one per stage of the pipeline:

```
tensor-ffn tokenize    ← corpus preparation
tensor-ffn train       ← expert training
tensor-ffn inference   ← baseline vs expert comparison
```

### Happy Path

```bash
# 1. tokenize your corpus once
tensor-ffn tokenize --model ./qwen2.5-0.5b/ --in ./docs/ --out ./corpus.bin

# 2. train the expert (~2.5 min on A100 for 50M tokens)
tensor-ffn train --model ./qwen2.5-0.5b/ --data ./corpus.bin --out ./golang-expert.bin

# 3. test it — two runs, compare outputs
tensor-ffn inference --model ./qwen2.5-0.5b/ --prompt "func main()"
tensor-ffn inference --model ./qwen2.5-0.5b/ --expert ./golang-expert.bin --prompt "func main()"
```

---

### tokenize

Reads raw text and produces a pre-tokenized corpus file for training.

```bash
tensor-ffn tokenize \
  --model  ./qwen2.5-0.5b/     \   ← HF repo dir, reads tokenizer.json
  --in     ./raw-text/          \   ← dir of .txt files or a single file
  --out    ./corpus.bin             ← flat binary of packed uint32 token ids
```

The output format is a flat binary of packed `uint32` token IDs with a small header:

```
[4 bytes magic]  [4 bytes version]  [8 bytes token count]  [token_count × uint32]
```

No sequences, no padding, no structure. The training loader slices it into
fixed-length windows at load time with proper EOS boundary handling. The
format stays dumb; the smarts live in the loader.

---

### train

Trains an expert FFN on a pre-tokenized corpus. The base model is fully frozen
throughout — only the expert FFN receives gradients.

```bash
tensor-ffn train \
  --model  ./qwen2.5-0.5b/     \   ← HF repo dir, reads config.json + .safetensors
  --data   ./corpus.bin         \   ← pre-tokenized corpus from tokenize step
  --out    ./golang-expert.bin      ← where the trained expert lands
```

Those are the only required flags. Everything else is derived or auto-tuned.

#### What is auto-derived from the model directory

HuggingFace repos ship `config.json`, `tokenizer.json`, and
`generation_config.json` alongside the weights. tensor-ffn reads everything
it needs from there — no extra flags required:

```
config.json            → hidden_dim, intermediate_dim, num_layers, seq_len, vocab_size
tokenizer.json         → vocab + BPE merges
generation_config.json → EOS token id, pad token id
```

The expert FFN shape is derived directly from the model config, so it is
always correct by construction. You cannot accidentally build a mismatched expert.

#### What is auto-tuned behind the scenes

```
batch size       auto from available VRAM — fills the GPU
seq length       pulled from config.json, capped at 2048 for training efficiency
lr schedule      cosine decay with 100-step linear warmup
learning rate    3e-4
weight decay     0.1
grad clip        1.0
dtype            bf16
steps            one full pass through --data
```

#### Optional overrides

```bash
tensor-ffn train \
  --model  ./qwen2.5-0.5b/     \
  --data   ./corpus.bin         \
  --out    ./golang-expert.bin  \
  --steps  5000                 \   ← override auto epoch calculation
  --lr     1e-4                 \   ← override if you have a reason
  --seq    1024                     ← override if your domain is short sequences
```

Nothing else is exposed. No dropout flag, no scheduler flag, no warmup flag,
no beta1/beta2. Those have correct values and exposing them creates rope to
hang yourself with.

---

### inference

Runs the model in one of two modes. Without `--expert`, the base model runs
alone as the baseline. With `--expert`, the trained expert delta is added at
every layer. Toggling between the two is the entire test harness.

```bash
# baseline — shared expert only
tensor-ffn inference \
  --model   ./qwen2.5-0.5b/    \
  --prompt  "func main()"

# combined — shared + your expert
tensor-ffn inference \
  --model   ./qwen2.5-0.5b/    \
  --expert  ./golang-expert.bin \
  --prompt  "func main()"
```

If the combined run consistently produces better domain-specific output than
the baseline, the expert has learned real knowledge and is worth keeping.

---

## Project Structure

```
tensor-ffn/
  trainer/
    src/          ← training loop, data loading, optimizer, checkpointing
    include/      ← trainer headers

  inference/
    src/          ← model loader, expert loader, forward pass, test harness
    include/      ← inference headers
```

### trainer

Trains the expert FFN on a raw text corpus using next-token loss across
every token position — identical to continued pretraining, but only the
expert FFN receives gradients. The shared expert (base model) is fully
frozen throughout. On completion the expert weights are saved as a
standalone .bin file, independent of the base.

#### Trainer File Structure — Qwen 2.5 0.5B

The trainer targets Qwen 2.5 0.5B only. The natural seams are what the data
is (io), what the model is (model), what training does (training), and what
hardware it runs on (backend).

```
tensor-ffn/
  trainer/
    src/

      main.cpp                  ← arg parsing, wires everything together

      io/
        safetensors.cpp         ← parse .safetensors header, mmap tensor data
        tokenizer.cpp           ← qwen tiktoken BPE — load vocab + merges, encode
        corpus.cpp              ← read raw text, tokenize, pack into fixed-len sequences
        checkpoint.cpp          ← save / load expert weights as .bin

      model/
        qwen.cpp                ← full qwen 2.5 frozen forward pass
        ffn_expert.cpp          ← trainable expert FFN — same SwiGLU shape, outputs delta
        rmsnorm.cpp             ← shared, used by both
        rope.cpp                ← rotary embeddings, shared

      training/
        trainer.cpp             ← main loop — forward, loss, backward, step
        loss.cpp                ← cross-entropy over logits, every token position
        optimizer.cpp           ← AdamW — param update + moment buffers for expert only

      backend/
        cuda_ops.cu             ← kernels: matmul (or cublas wrap), SiLU, softmax
        memory.cpp              ← device alloc/free, host↔device transfers

    include/
      io/
        safetensors.h
        tokenizer.h
        corpus.h
        checkpoint.h
      model/
        qwen.h
        ffn_expert.h
        rmsnorm.h
        rope.h
      training/
        trainer.h
        loss.h
        optimizer.h
      backend/
        cuda_ops.h
        memory.h
```

**`safetensors.cpp` is its own unit** — the format has a JSON header + raw binary
blob. Parsing that cleanly before anything else touches weights is worth the
separation. Used for both the base model load and reading any pretrained expert
checkpoints.

**`qwen.cpp` is frozen, `ffn_expert.cpp` is not** — they are intentionally
separate files. `qwen.cpp` never accumulates gradients or holds optimizer state.
`ffn_expert.cpp` is the only file that does. This boundary is the core invariant
of the whole project and the file split makes it impossible to accidentally mix
them up.

**`ffn_expert.cpp` mirrors the SwiGLU shape** — Qwen 2.5 FFN is `gate_proj`,
`up_proj`, `down_proj` with SiLU gating. The expert has the same three weight
matrices. The output of `ffn_expert` is added to the output of the base FFN,
not substituted.

**`corpus.cpp` owns sequence packing** — packing multiple documents into
fixed-length sequences with proper EOS boundaries so loss does not bleed across
documents is non-trivial enough to deserve its own file rather than being stuffed
into the training loop.

**`cuda_ops.cu` can start thin** — for 0.5B you can lean on cuBLAS for matmuls
and only write custom kernels for things cuBLAS does not cover (fused SiLU,
in-place RMSNorm). Keep it one file until you have a reason to split.

#### Speed Table — Single A100 80GB, Qwen 0.5B Base

Corpus size used for time estimates: **50M tokens**
(reasonable for a focused domain expert like a golang package set)

| Method                   | Trainable Params | Tokens / sec | 50M Token Corpus |
|--------------------------|------------------|--------------|------------------|
| Full continued pretrain  | ~500M            | ~90k         | ~9 min           |
| Your expert FFN only     | ~50M             | ~350k        | ~2.5 min         |

Training only the expert FFN is ~4× faster and uses ~10× fewer trainable
parameters than full continued pretraining — with no changes to the base model.

The speedup comes from three compounding effects: no gradients flow through
the frozen base layers, activations do not need to be stored for the base
during the forward pass, and the resulting memory reduction allows larger
batch sizes on the same GPU. The forward pass still runs the full model, so
the gain is on the backward pass and memory side — ~4× is the realistic
number once that is accounted for.

> **Note:** Actual throughput depends on batch size tuning, gradient
> checkpointing settings, and sequence packing efficiency. The ~350k figure
> assumes the freed memory is recovered via larger batches. Running with
> default settings carried over from a full-training config may not realise
> the full gain.

### inference

Loads the shared expert (base model) and optionally loads a trained expert.
Runs in one of two modes toggled by a single flag — shared expert only as
the baseline, or shared plus expert as the combined test. No router, no
routing weights, no conditions. When the expert is loaded it fires on every
token, always.

---

## Two Modes, No Router

```
Mode 1 — shared only:     inference runs the base model alone
Mode 2 — shared + expert: inference adds the expert delta at every layer
```

Toggling between modes is the entire test harness. If Mode 2 produces
domain-specific outputs that Mode 1 gets wrong, the expert has learned
real knowledge.

---

## What Comes After tensor-ffn

Once your experts pass testing here, they move into the broader Tensor
pipeline where routing and multi-expert orchestration are handled separately.

```
tensor-ffn     ← you are here — train experts, validate quality
tensor-router  ← gating logic, decide which expert fires when
tensor-moe     ← full assembly, multiple experts, routing weights
```
