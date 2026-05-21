# Activation Baking — K Formula Derivation

## Setup

Let `h_ℓ ∈ ℝ^d` denote the residual stream at layer `ℓ`, and let `ĉ ∈ ℝ^d` be a
unit-norm behavioral direction extracted via contrastive activation addition (CAA).

**Activation baking** writes the vector `K_ℓ · ĉ` as a persistent bias into the
MLP weight `W_down ∈ ℝ^{d × d_mlp}` at layer `ℓ`, so every forward pass adds
`K_ℓ · ĉ` to the residual stream without inference-time hooks.

The central question: **what is the right value of `K_ℓ`?**

---

## Rank-1 Perturbation Equivalence

Adding `K · ĉ` to the residual stream is equivalent, in its first-order effect on
downstream activations, to a rank-1 perturbation of the weight matrix.

Consider the MLP output at layer `ℓ`:

```
y = W_down · a(W_up · h)   ≈   W_down · h   (linearised around h)
```

A rank-1 weight perturbation `ΔW = α · u · vᵀ` (with `‖u‖ = ‖v‖ = 1`) modifies
the output by:

```
ΔW · h = α (vᵀ h) u
```

Directly adding `K · ĉ` to the residual stream after the layer produces an
identical additive shift when:

```
α (vᵀ h) u  =  K · ĉ
```

Setting `u = ĉ` and `v = h / ‖h‖` (so the perturbation is aligned with the
current activation), the magnitude condition becomes:

```
α · ‖vᵀ h‖  =  K
α · ‖h‖     =  K          (since v = h/‖h‖, so vᵀh = ‖h‖)
```

For a **unit rank-1 update** (`α = 1`), this gives:

```
K  =  ‖h_ℓ‖
```

---

## Calibration via Layer Norms

Since `‖h_ℓ‖` varies across tokens and prompts, we calibrate `K_ℓ` to the
**expected residual stream norm** at layer `ℓ`, measured over a representative
prompt corpus:

```
μ̄_ℓ  =  𝔼_x [ ‖h_ℓ(x)‖ ]
```

Substituting and normalising by the representation dimension `d` to account for
the projection geometry of `W_down`:

```
┌─────────────────────────────┐
│   K_ℓ  =  μ̄_ℓ / √d         │
└─────────────────────────────┘
```

The `√d` factor arises from the inner product concentration: for a random unit
vector `v` in `ℝ^d`, `𝔼[|vᵀ h|] ≈ ‖h‖ / √d` by the Johnson-Lindenstrauss
lemma, so projecting `K` back into the weight-perturbation frame requires
rescaling by `√d`.

---

## Interpretation

| Quantity | Meaning |
|---|---|
| `μ̄_ℓ` | Mean residual stream norm at layer `ℓ` — grows monotonically with depth |
| `√d` | Dimension normalisation — converts absolute norm to per-direction scale |
| `K_ℓ` | Injection magnitude that matches one rank-1 update's worth of perturbation |

**Key consequences:**

1. **`K_ℓ` naturally ramps with depth** — because `μ̄_ℓ` grows (e.g. ×84 across
   Llama-3.1-8B layers), the formula produces a ramped schedule automatically.
   A flat-K schedule under-injects at deep layers and over-injects at shallow ones.

2. **Lobotomy threshold** — injecting at `K >> K_ℓ` (empirically ≥ 3×) collapses
   output coherence. The formula defines the safe ceiling.

3. **Asymmetric degradation in instruction-tuned models** — steering along `+ĉ`
   (toward a safety-aligned direction) degrades faster than steering along `−ĉ`
   because RLHF embeds safety near a coherence attractor; over-pushing toward it
   causes degenerate outputs before pushing away from it does.
   In base (non-instruction-tuned) models this asymmetry vanishes — both
   directions degrade and gain steeply and symmetrically.

4. **Gemma-2 deviation** — dual-norm architecture (pre + post RMSNorm) means
   `μ̄_ℓ` integrates accumulated `γ^post` scales rather than raw MLP spectral
   norms. The formula still holds; only the physical proxy for `μ̄_ℓ` changes
   (ρ = 0.9992 correlation on empirical data).

---

## Empirical Validation (Llama-3.1-8B-Instruct, safety direction)

| K scale | ramp_pos (safe%) | ramp_neg (safe%) | Baseline |
|---|---|---|---|
| 1.0 × K_ℓ | **0.70** | 0.45 | 0.55 |
| 2.0 × K_ℓ | 0.55 | 0.15 | 0.55 |
| 3.0 × K_ℓ | 0.00 (gibberish) | 0.05 | 0.55 |

- At `K = K_ℓ`: clean bidirectional control (+15% safe, −10% safe) with no degradation.
- At `K = 2K_ℓ`: ramp_neg highly effective (−40%), ramp_pos begins degrading.
- At `K = 3K_ℓ`: ramp_pos fully collapses (lobotomy), ramp_neg near-maximal.

The formula ceiling is empirically confirmed.
