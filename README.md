# LLaDA on Tiny Shakespeare

This project is a compact, from-scratch implementation of a masked diffusion language model based
on **LLaDA: Large Language Diffusion Models** (`2502.09992v3.pdf`). It implements the complete
experimental pipeline on Tiny Shakespeare:

1. diffusion-language-model pretraining;
2. speaker-conditioned supervised fine-tuning (SFT);
3. group-relative reinforcement learning with Sandwiched Policy Gradient (SPG);
4. checkpoint-based text generation;
5. TensorBoard monitoring and automated tests.

The model uses characters as tokens so that every stage can run on a laptop. This is an educational
small-scale adaptation, not a reproduction of the paper's 8B-parameter results.

## Contents

- [Implemented features](#implemented-features)
- [Project structure](#project-structure)
- [Requirements and installation](#requirements-and-installation)
- [Recommended end-to-end workflow](#recommended-end-to-end-workflow)
- [Stage 1: pretraining](#stage-1-pretraining)
- [Stage 2: dialogue SFT](#stage-2-dialogue-sft)
- [Stage 3: SPG reinforcement learning](#stage-3-spg-reinforcement-learning)
- [Inference](#inference)
- [TensorBoard](#tensorboard)
- [Checkpoints and outputs](#checkpoints-and-outputs)
- [Testing](#testing)
- [Current verified artifacts](#current-verified-artifacts)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Implemented features

### Model

- LLaMA-style Transformer blocks.
- Full bidirectional attention with no causal mask.
- RMSNorm, SwiGLU, and rotary positional embeddings (RoPE).
- Tied token-embedding and output-projection weights.
- PyTorch scaled dot-product attention.
- Automatic device selection in the order CUDA, Apple MPS, CPU.

The default model has six layers, eight attention heads, a width of 256, an FFN width of 768,
a 256-character context window, and approximately 5.13 million parameters.

### Pretraining

- Continuous diffusion time sampled independently for every sequence.
- Independent token masking according to the sampled time.
- Paper-faithful masked-token objective with `1/t` weighting.
- One-percent variable-length sequence augmentation.
- AdamW and a warmup-stable-decay learning-rate schedule.
- Validation, gradient clipping, checkpoints, resume support, and TensorBoard.

### Supervised fine-tuning

- Parses speaker-labelled Shakespeare dialogue into prompt/response pairs.
- Uses preceding turns and the next speaker label as the prompt.
- Leaves the prompt clean and masks response tokens only.
- Adds EOS while preserving all pretrained vocabulary rows.
- Uses length-bucketed batches to reduce excessive EOS padding.
- Treats EOS padding as response data, following the LLaDA SFT procedure.

### Reinforcement learning

- Group-relative rollouts and mean-centered advantages.
- Semi-autoregressive block-wise diffusion generation.
- SPG lower-bound optimization for positive advantages.
- A 50/50 EUBO-ELBO mixture for negative advantages.
- Block-wise Monte Carlo perturbations.
- Frozen SFT reference model and exact categorical KL anchoring.
- Composite Shakespeare reward with independently logged components.
- Resume support, evaluation, checkpoints, parameter histograms, and generated-text logging.

### Inference

- Greedy or temperature-based token prediction.
- Top-k filtering.
- Low-confidence or random remasking.
- Optional fixed prompt prefix.
- Multiple samples and deterministic seeds.
- EOS-aware decoding for SFT and SPG checkpoints.

## Project structure

```text
.
├── 2502.09992v3.pdf       # LLaDA paper
├── train.py               # masked-diffusion pretraining
├── sft.py                 # Shakespeare dialogue SFT
├── spg_train.py           # SPG reinforcement learning
├── generate.py            # checkpoint-based inference
├── requirements.txt
├── llada/
│   ├── data.py            # download, character tokenizer, pretraining data
│   ├── model.py           # Transformer and vocabulary resizing
│   ├── objective.py       # pretraining and SFT objectives
│   ├── reward.py          # composite Shakespeare reward
│   ├── sampling.py        # diffusion samplers
│   ├── sft_data.py        # dialogue parsing and SFT batching
│   └── spg.py             # SPG perturbations and evidence bounds
├── tests/
│   └── test_llada.py
├── data/                  # downloaded data; ignored by Git
└── out/                   # checkpoints and TensorBoard events; ignored by Git
```

## Requirements and installation

Python 3.10 or newer is recommended. The project requires PyTorch 2.1+ and TensorBoard.

```bash
python3 -m pip install -r requirements.txt
```

Confirm that PyTorch can see the expected accelerator:

```bash
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
PY
```

Every training command accepts `--device auto`, `--device cuda`, `--device mps`, or
`--device cpu`. The default is `auto`.

The first pretraining or SFT run downloads Tiny Shakespeare to
`data/shakespeare/input.txt` from Karpathy's `char-rnn` repository.

## Recommended end-to-end workflow

Run commands from the repository root.

### 1. Run the tests

```bash
python3 -m unittest discover -s tests -v
```

### 2. Pretrain LLaDA

```bash
python3 train.py
```

### 3. Fine-tune for three dialogue epochs

There are 6,387 training pairs. At batch size eight, three epochs require
`ceil(6387 / 8) * 3 = 2397` iterations.

```bash
python3 sft.py \
  --pretrained-checkpoint out/shakespeare/best.pt \
  --out-dir out/shakespeare-sft-3epoch \
  --max-iters 2397 \
  --batch-size 8 \
  --warmup-iters 50 \
  --eval-interval 100 \
  --eval-iters 20 \
  --checkpoint-interval 200 \
  --log-interval 10
```

### 4. Run SPG reinforcement learning

For a fresh 200-update run:

```bash
python3 spg_train.py \
  --checkpoint out/shakespeare-sft-3epoch/best.pt \
  --out-dir out/shakespeare-spg-new \
  --max-updates 200
```

The existing `out/shakespeare-spg/` directory contains a short validation trial. To continue that
trial instead of starting over:

```bash
python3 spg_train.py \
  --checkpoint out/shakespeare-sft-3epoch/best.pt \
  --out-dir out/shakespeare-spg \
  --resume out/shakespeare-spg/latest.pt \
  --max-updates 200
```

### 5. Generate dialogue

```bash
python3 generate.py \
  --checkpoint out/shakespeare-spg-new/best.pt \
  --prompt $'ROMEO:\n' \
  --length 96 \
  --steps 32 \
  --temperature 0.9 \
  --top-k 20
```

## Stage 1: pretraining

For clean tokens `x0`, pretraining samples `t ~ U(0, 1]` and independently replaces every token
with `[MASK]` with probability `t`. The implemented Monte Carlo loss is

```text
loss = mean_batch(
    sum_i(1[x_t_i = MASK] * CE(model(x_t)_i, x0_i)) / (t * L)
)
```

The loss divides by the complete sequence length `L`, not by the realized number of masks. Together
with `1/t`, this matches Equation 3 and Algorithm 1 of the LLaDA paper.

### Default pretraining configuration

| Setting | Default |
|---|---:|
| Iterations | 5,000 |
| Batch size | 32 |
| Sequence length | 256 |
| Layers / heads / width | 6 / 8 / 256 |
| FFN width | 768 |
| Maximum learning rate | `4e-4` |
| Minimum learning rate | `4e-5` |
| Warmup | 200 iterations |
| Decay begins | iteration 4,000 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Variable-length probability | 0.01 |

### Quick pretraining smoke test

```bash
python3 train.py \
  --out-dir out/pretrain-smoke \
  --max-iters 20 \
  --eval-interval 10 \
  --eval-iters 2 \
  --checkpoint-interval 20 \
  --batch-size 8 \
  --block-size 64 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --ffn-dim 256
```

### Resume pretraining

The architecture arguments must match the checkpoint. For a default-architecture run:

```bash
python3 train.py \
  --resume out/shakespeare/latest.pt \
  --max-iters 7000
```

For a custom architecture, repeat its `--block-size`, `--n-layer`, `--n-head`, `--n-embd`, and
`--ffn-dim` values when resuming.

## Stage 2: dialogue SFT

Tiny Shakespeare is not an instruction dataset. The SFT stage converts its dialogue turns into
conditional examples such as:

```text
Prompt:
ROMEO:
But, soft! what light through yonder window breaks?

JULIET:

Response:
O Romeo, Romeo! wherefore art thou Romeo?
```

For prompt `p0` and response `r0`, SFT keeps the prompt unchanged, samples diffusion time `t`,
and masks response tokens only:

```text
input = concat(p0, r_t)
loss  = sum_i(1[r_t_i = MASK] * CE(model(input)_i, r0_i)) / (t * L_response)
```

The base tokenizer contains 65 Shakespeare characters plus `[MASK]`. SFT adds EOS, expanding the
vocabulary from 66 to 67 entries. All original embedding rows are copied exactly; only EOS starts
from a new initialization.

### Short SFT run

```bash
python3 sft.py \
  --pretrained-checkpoint out/shakespeare/best.pt \
  --out-dir out/shakespeare-sft \
  --max-iters 500
```

### Three-epoch SFT run

Use the 2,397-iteration command in the end-to-end workflow. The learning-rate schedule warms to
`2.5e-5`, remains stable, and decays to `2.5e-6` over the final ten percent of iterations.

Unlike pretraining and SPG, `sft.py` currently does not resume optimizer state. Start SFT runs in a
new output directory if their settings differ.

## Stage 3: SPG reinforcement learning

Ordinary GRPO cannot directly use an exact LLaDA sequence log-likelihood because that likelihood is
intractable. `spg_train.py` uses the Sandwiched Policy Gradient construction:

1. Sample a prompt and a group of responses.
2. Compute reward `R_j` for each response.
3. Center rewards within the group:

   ```text
   A_j = R_j - mean_group(R)
   ```

4. For `A_j >= 0`, maximize the diffusion ELBO.
5. For `A_j < 0`, minimize a mixture of the evidence upper bound and ELBO.
6. Add KL regularization against a frozen SFT reference policy.

The optimized score is

```text
score_j = ELBO_j                              if A_j >= 0
score_j = omega * EUBO_j + (1-omega)*ELBO_j  if A_j < 0

loss = -mean_j(A_j * score_j) + kl_coefficient * KL(policy || reference)
```

### Composite Shakespeare reward

| Component | Weight | Purpose |
|---|---:|---|
| Reference character n-gram F1 | 0.35 | Rewards similarity to the real next utterance |
| Held-out character 4-gram fluency | 0.30 | Rewards Shakespeare-like local character statistics |
| EOS and response length | 0.15 | Rewards valid termination and discourages extreme lengths |
| Dialogue formatting | 0.10 | Rewards capitalization, punctuation, and clean response form |
| Anti-repetition | 0.10 | Penalizes repeated trigrams and long character runs |

This is a heuristic research reward, not human-preference feedback. Inspect each component in
TensorBoard because the policy can exploit imperfect rewards—for example, by producing very short
responses or repetitive high-scoring phrases.

### Default SPG configuration

| Setting | Default |
|---|---:|
| Policy checkpoint | `out/shakespeare-sft-3epoch/best.pt` |
| Updates | 200 |
| Group size | 4 |
| Generation length | 96 |
| Diffusion steps | 32 |
| Block length | 16 |
| Monte Carlo samples | 2 |
| Inner updates per rollout | 2 |
| EUBO beta | 1.5 |
| EUBO mixture weight | 0.5 |
| Learning rate | `1e-6` |
| Gradient clipping | 0.2 |
| Rollout temperature | 0.9 |
| Top-k | 20 |
| Prompt/context masking | 0.15 |
| Reference KL coefficient | 0.02 |

`--generation-length` must be divisible by `--block-length`. The prompt plus generation length must
also fit within the model's 256-character context window.

### Quick SPG smoke test

```bash
python3 spg_train.py \
  --checkpoint out/shakespeare-sft-3epoch/best.pt \
  --out-dir out/spg-smoke \
  --max-updates 2 \
  --eval-interval 1 \
  --eval-prompts 1 \
  --checkpoint-interval 1 \
  --histogram-interval 1 \
  --text-interval 1
```

### Resume SPG

`--max-updates` is the desired total, not the number of additional updates:

```bash
python3 spg_train.py \
  --checkpoint out/shakespeare-sft-3epoch/best.pt \
  --out-dir out/shakespeare-spg \
  --resume out/shakespeare-spg/latest.pt \
  --max-updates 500
```

Keep `--checkpoint` pointed at the original SFT policy when resuming. It defines the frozen reference
model used for KL regularization.

## Inference

### Base pretrained model

```bash
python3 generate.py \
  --checkpoint out/shakespeare/best.pt \
  --length 256 \
  --steps 128
```

### Speaker-conditioned SFT model

```bash
python3 generate.py \
  --checkpoint out/shakespeare-sft-3epoch/best.pt \
  --prompt $'ROMEO:\n' \
  --length 128 \
  --steps 64 \
  --temperature 0.8 \
  --top-k 20
```

### SPG model

```bash
python3 generate.py \
  --checkpoint out/shakespeare-spg/best.pt \
  --prompt $'KING RICHARD:\n' \
  --length 96 \
  --steps 32 \
  --temperature 0.9 \
  --top-k 20 \
  --num-samples 3
```

Use zsh's `$'...'` quoting when a prompt contains `\n`; ordinary double quotes pass the backslash
literally.

### Inference options

| Option | Meaning |
|---|---|
| `--checkpoint` | Checkpoint to load |
| `--prompt` | Fixed prefix; must use only training-vocabulary characters |
| `--length` | Number of characters to generate |
| `--steps` | Reverse diffusion steps |
| `--remasking low_confidence` | Remask the lowest-confidence predictions |
| `--remasking random` | Follow the random reverse-process remasking rule |
| `--temperature 0` | Paper-style greedy prediction |
| `--temperature > 0` | Stochastic token sampling |
| `--top-k` | Restrict sampling to the highest-scoring tokens; zero disables it |
| `--num-samples` | Number of responses generated in parallel |
| `--seed` | Sampling seed |
| `--show-progress` | Print masks remaining at every diffusion step |

The prompt length plus `--length` cannot exceed the checkpoint's `max_seq_len`, which is 256 for
the default model.

## TensorBoard

TensorBoard is enabled by default for all training stages. Launch it with the Python module form so
the command works even when the standalone `tensorboard` executable is not on `PATH`.

### Pretraining

```bash
python3 -m tensorboard.main --logdir out/shakespeare/tensorboard
```

### Three-epoch SFT

```bash
python3 -m tensorboard.main --logdir out/shakespeare-sft-3epoch/tensorboard
```

### SPG

```bash
python3 -m tensorboard.main --logdir out/shakespeare-spg/tensorboard
```

Then open [http://localhost:6006](http://localhost:6006).

The message `TensorFlow installation not found - running with reduced feature set` is harmless.
TensorBoard can read PyTorch event files without TensorFlow.

### Logged information

Pretraining logs loss, masked accuracy, mask ratio, diffusion time, learning rate, gradient norm,
sequence length, throughput, validation metrics, configuration text, and parameter histograms.

SFT logs response-only loss, masked accuracy, masking statistics, validation metrics, learning rate,
gradient norm, and configuration text.

SPG logs:

- mean, minimum, maximum, and standard deviation of group rewards;
- every reward component separately;
- positive/negative advantage fractions and advantage variance;
- ELBO, EUBO, mixed negative score, policy objective, and reference KL;
- EOS rate, response length, and within-group uniqueness;
- learning rate, gradient norm, update time, and rollout throughput;
- validation rewards and reward components;
- generated prompt/reference/sample text;
- periodic parameter histograms.

Use a different `--out-dir` for independent experiments so their TensorBoard events are not mixed.

## Checkpoints and outputs

Each stage writes to its configured output directory:

```text
out/<run>/
├── best.pt             # best validation loss or evaluation reward
├── latest.pt           # most recent periodic/final checkpoint
├── config.json         # exact CLI configuration
├── tokenizer.json      # character vocabulary and EOS capability
└── tensorboard/        # TensorBoard event files
```

Pretraining and SFT define “best” using the lowest validation loss. SPG defines it using the highest
mean evaluation reward.

Checkpoints contain model weights, model configuration, tokenizer metadata, training progress, and
optimizer state. SPG checkpoints also record the frozen reference-checkpoint path and best evaluation
reward.

The `data/` and `out/` directories are ignored by Git because they contain downloaded data and
potentially large binary artifacts.

## Testing

Run the complete test suite:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- model forward and backward passes;
- bidirectional rather than causal attention;
- pretraining and SFT objectives;
- prompt preservation and mask resolution;
- random, low-confidence, and block-wise generation;
- pretrained-row preservation during vocabulary expansion;
- dialogue parsing and EOS behavior;
- differentiable and finite SPG bounds;
- bounded reward components and reference similarity.

Inspect every command-line option with:

```bash
python3 train.py --help
python3 sft.py --help
python3 spg_train.py --help
python3 generate.py --help
```

## Current verified artifacts

The current workspace contains the following completed or validated runs:

| Stage | Checkpoint | Progress | Best metric |
|---|---|---:|---:|
| Pretraining | `out/shakespeare/best.pt` | 5,000 iterations | validation loss `1.6989` |
| Three-epoch SFT | `out/shakespeare-sft-3epoch/best.pt` | best at iteration 2,300 | validation loss `1.2923` |
| SPG run | `out/shakespeare-spg/best.pt` | best at update 150 | evaluation reward `0.4183` |
| SPG latest | `out/shakespeare-spg/latest.pt` | update 199 | best reward remains `0.4183` |

The SPG result is a small 200-update experiment. It proves that rollouts, evidence-bound updates,
checkpoints, resume behavior, evaluation, and TensorBoard work together; its noisy heuristic reward
alone is not evidence that RL improves subjective output quality. Compare fixed prompts and inspect
reward components before drawing conclusions.

## Limitations

- Character tokenization makes training accessible but is much less capable than a modern subword
  tokenizer.
- Tiny Shakespeare is small and was already seen during pretraining. The dialogue split is useful for
  engineering validation, not a rigorous generalization benchmark.
- Shakespeare dialogue SFT does not create a general-purpose instruction-following assistant.
- The SPG reward is heuristic and can be exploited. Reference similarity also favors the recorded next
  line even though many alternative continuations could be valid.
- SPG evaluation uses sampled responses and is noisy, particularly with few evaluation prompts.
- Full-parameter optimization is used; LoRA and distributed training are not implemented.
- SFT does not currently support resuming optimizer state.
- The small model can produce malformed words, repetitive phrases, or premature EOS.

## Troubleshooting

### `zsh: command not found: tensorboard`

Use:

```bash
python3 -m tensorboard.main --logdir PATH/TO/tensorboard
```

### Checkpoint not found

Run the preceding stage or pass the correct path with `--checkpoint`,
`--pretrained-checkpoint`, or `--resume`.

### Prompt contains unknown characters

The character tokenizer rejects characters absent from Tiny Shakespeare. Use ASCII punctuation and
characters present in the corpus. In particular, avoid typographic quotation marks unless they occur
in the tokenizer vocabulary.

### Sequence exceeds the context length

Reduce `--length` or shorten the prompt. Their combined length must be at most 256 with the default
checkpoints.

### SPG generation/block assertion fails

Choose a generation length divisible by the block length, for example:

```bash
--generation-length 96 --block-length 16
```

### Out of memory or very slow training

- Pretraining: reduce `--batch-size`, `--block-size`, or model dimensions.
- SFT: reduce `--batch-size`.
- SPG: reduce `--group-size`, `--generation-length`, `--diffusion-steps`, `--mc-samples`, or
  `--inner-updates`.
- Use `--histogram-interval 0` to disable expensive parameter histograms.

### SPG reward rises while text gets worse

This is reward hacking. Inspect individual reward components and generated samples in TensorBoard.
Common mitigations are increasing the KL coefficient, strengthening anti-repetition and length
penalties, using more evaluation prompts, or replacing the heuristic reward with a separately trained
reward model.

### Reproducibility

All scripts accept `--seed`. Exact results can still vary between CPU, CUDA, and MPS because some
device kernels and sampling operations are not bitwise deterministic.
