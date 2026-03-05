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
alongside the base FFN, outputs a correction delta, and is saved as an
independent file on disk.

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

### inference

Loads the shared expert (base model) and optionally loads a trained expert.
Runs in one of two modes toggled by a single flag — shared expert only as
the baseline, or shared plus expert as the combined test. No router, no
routing weights, no conditions. When the expert is loaded it fires on every
token, always.

---

## Two Modes, No Router

```
Mode 1 — shared only:    inference runs the base model alone
Mode 2 — shared + expert: inference adds the expert delta at every layer
```

Toggling between modes is the entire test harness. If Mode 2 produces
correct signatures, accurate import paths, and package-specific patterns
that Mode 1 gets wrong, the expert has learned real knowledge.

---

## What Comes After tensor-ffn

Once your experts pass testing here, they move into the broader Tensor
pipeline where routing and multi-expert orchestration are handled separately.

```
tensor-ffn     ← you are here — train experts, validate quality
tensor-router  ← gating logic, decide which expert fires when
tensor-moe     ← full assembly, multiple experts, routing weights
```
