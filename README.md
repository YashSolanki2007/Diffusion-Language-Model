# LLaDA pretraining on Tiny Shakespeare

This repository implements the pretraining stage of **LLaDA (Large Language Diffusion Models)**
from the included paper (`2502.09992v3.pdf`) on the Tiny Shakespeare corpus.

The implementation keeps the parts of the paper that define LLaDA:

- a LLaMA-style Transformer with RMSNorm, SwiGLU, and RoPE;
- full bidirectional attention (there is no causal attention mask);
- a dedicated mask token;
- one independently sampled diffusion time `t ~ U(0, 1]` per sequence;
- independent token masking with probability `t`;
- cross-entropy only at masked positions, with the paper's `1/t` weighting;
- AdamW and a warmup-stable-decay learning-rate schedule;
- the paper's 1% random-sequence-length augmentation.

Characters are tokens in this small-data adaptation. That keeps tokenization transparent and makes
the whole experiment trainable on a laptop while leaving the diffusion objective unchanged.

## Quick start

Python 3.10+, PyTorch 2.1+, and TensorBoard are required.

```bash
python3 -m pip install -r requirements.txt
python3 train.py
```

The first run downloads Tiny Shakespeare from Karpathy's `char-rnn` repository. Checkpoints,
the exact run configuration, and the character vocabulary are written to `out/shakespeare/`.
TensorBoard event files are written to `out/shakespeare/tensorboard/` by default.

In another terminal, launch TensorBoard with:

```bash
python3 -m tensorboard.main --logdir out/shakespeare/tensorboard
```

Using `python3 -m` ensures TensorBoard is launched from the same Python environment as the
training script, even when its standalone executable is not available on your shell's `PATH`.

The dashboard records per-step training loss, masked-token accuracy, realized mask fraction,
sampled diffusion time, learning rate, gradient norm, sequence length, throughput, validation
metrics, run configurations, and periodic parameter histograms. Use
`--histogram-interval 0` to disable histograms, `--tensorboard-log-dir PATH` to change the event
directory, or `--no-tensorboard` to disable TensorBoard entirely.

For a quick CPU/MPS smoke run:

```bash
python3 train.py \
  --max-iters 20 --eval-interval 10 --eval-iters 2 \
  --checkpoint-interval 20 --batch-size 8 --block-size 64 \
  --n-layer 2 --n-head 4 --n-embd 128 --ffn-dim 256
```

Resume a run with the same architecture arguments:

```bash
python3 train.py --resume out/shakespeare/latest.pt
```

## Inference

Generate 256 characters from the best checkpoint with the paper's low-confidence remasking:

```bash
python3 generate.py --checkpoint out/shakespeare/best.pt --length 256 --steps 128
```

Condition generation on a fixed prefix:

```bash
python3 generate.py --prompt $'ROMEO:\n' --length 200 --steps 100
```

Greedy token prediction (`--temperature 0`) matches the paper. To obtain diverse samples, use a
positive temperature and optionally top-k filtering:

```bash
python3 generate.py --temperature 0.8 --top-k 20 --num-samples 3
```

Both `--remasking low_confidence` (Algorithm 5) and `--remasking random` (Algorithm 4) are
available. The output length plus prompt length cannot exceed the checkpoint's training block size.

## Shakespeare dialogue SFT

Tiny Shakespeare has no instruction annotations, but its speaker-labelled turns can be converted
into conditional dialogue pairs. `sft.py` uses preceding turns plus the next speaker label as the
prompt and that speaker's utterance as the response. It follows Equation (5) and Algorithm 2:
the prompt remains clean, only response tokens are masked, and loss is normalized by `t * L'`.

The SFT tokenizer adds EOS to the pretrained vocabulary. Shorter examples in a length-bucketed
batch are padded with EOS, and those positions remain part of the response objective as prescribed
by the paper. All pretrained token embeddings are retained; only the new EOS row is initialized.

Run SFT from the pretrained checkpoint:

```bash
python3 sft.py --pretrained-checkpoint out/shakespeare/best.pt
```

Monitor it with:

```bash
python3 -m tensorboard.main --logdir out/shakespeare-sft/tensorboard
```

Then generate speaker-conditioned dialogue:

```bash
python3 generate.py \
  --checkpoint out/shakespeare-sft/best.pt \
  --prompt $'ROMEO:\n' --length 128 --steps 64 \
  --temperature 0.8 --top-k 20
```

Run tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Objective

For clean tokens `x0`, the training code samples a time `t` and creates `xt` by replacing every
token independently with `[MASK]` with probability `t`. The implemented Monte Carlo loss is:

```text
loss = mean_batch(sum_i(1[xt_i = MASK] * CE(model(xt)_i, x0_i)) / (t * L))
```

Dividing by the full sequence length `L`, rather than only by the realized number of masks, is
important: together with `1/t`, this is the estimator in Equation (3) and Algorithm 1 of the paper.
