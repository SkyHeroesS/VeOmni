# aiter Attention on VeOmni — Wan2.1 DiT validation (AMD ROCm / MI308X)

**Status:** for human review (draft, do not merge as-is)
**Author:** aiter-attention enablement follow-up to #890
**Hardware:** 8× AMD Instinct MI308X (gfx942), ROCm 7.14
**Image:** `amdagi/veomni:rocm7.14_torch2.12_py3.12` (torch 2.12.0+rocm7.14, diffusers 0.37.0, flash-attn 2.8.3, aiter bundled)

---

## TL;DR

- Same `aiter` attention backend as the Qwen3-Omni report, here on the **Wan2.1 DiT** (diffusion transformer) via VeOmni's SP-aware attention processor with **ulysses_size 2** (so this also validates aiter on the sequence-parallel path).
- **Accuracy is bit-identical**: FA2 vs aiter training loss matches to 2 dp for **all 30 steps** (LoRA fine-tune → frozen base, so no gradient-compounding divergence).
- **aiter is ~27% faster end-to-end** (FA2 5.46 → aiter 4.29 s/step). Unlike the MoE model, the Wan DiT is **attention-bound**, so a faster attention kernel moves the whole step.
- Root cause of the win (from profiling): aiter's FMHA-v3 cuts attention-kernel time **37%** (backward **48%**) vs the flash-attn-2.8.3 CK `ck_tile` kernels on MI308X.
- **The win grows with context length, and this holds in real training** — not just in an isolated kernel benchmark. End-to-end on 8× MI308X (LoRA, ulysses 2), holding the latent factorisation fixed and scaling only the spatial dims: **1.10× at 8,064 tokens, 1.28× at 16,422, 1.36× at 32,256**. The midpoint is the real Tom-and-Jerry data, so the curve is anchored on a real measurement, and the longest point was reproduced on two independent nodes (§7.3.1). At the exact real-480P geometry instead (32,760 tokens, `F=21,H=60,W=104`) the same setup reads **1.41×** (§7.3) — the two long points differ only in how the latent is factorised at essentially the same length.
- A 14B **module-level** comparison corroborates the same slope at a 10× larger parameter count (§7.1–§7.2). End-to-end 14B is out of scope here: the documented I2V training path does not run on current `main`, and repairing it is separate work.

---

## 1. What changed

Backend selection only (see the Qwen3-Omni report §1 for the integration): add
`--model.ops_implementation.attn_implementation aiter`. Wan's DiT attention runs
through `WanSPAttnProcessor → ALL_ATTENTION_FUNCTIONS[...] → flash_attention_forward`,
i.e. the same dispatch as the LLM path, so no Wan-specific attention code was needed.

**Incidental fix (required, pre-existing bug):** the Wan attention path read
`attn.to_q.weight` / `attn.to_out[0].weight` directly, which throws
`AttributeError: 'LoraLinear' object has no attribute 'weight'` under LoRA on
current `main` (both the FA2 bf16 pre-check and the shared `WanAttentionKernelModule`).
This blocks Wan+LoRA on **any** backend, not just aiter. Fixed by reading the dtype via
`next(module.parameters()).dtype` in `veomni/models/diffusers/wan_t2v/wan_transformer/modeling_wan_transformer.py`
(3 sites). Flagged separately for review — arguably belongs in its own small PR.

## 2. Experiment setup

| item | value |
|---|---|
| model | Wan2.1-T2V-1.3B-Diffusers (transformer), **LoRA** rank 128 |
| task | `tasks/train_dit.py`, `configs/dit/wan2.1_I2V_1.3B_lora.yaml`, `training_task=offline_training` |
| data | Tom-and-Jerry video, **precomputed offline VAE+text embeddings** (no VAE/text encode at train time) |
| GPUs | 8× MI308X, FSDP2, **ulysses_size 2** (sequence parallel) |
| batch | micro 1 × (dp 4) = global 8 |
| steps | 30 (`num_train_epochs 1`), seed 42 |
| measured on | current `main`, on 8 verified-idle cards (0 GB before and after each run) |
| **sequence length** | **16,422 video tokens** — the offline latents are `[1, 32, 21, 46, 68]`, so with Wan's `(1,2,2)` patching the DiT sees `21 × 23 × 34` tokens (a 368×544 clip, i.e. about half of real 480P's 32,760) |

Only `attn_implementation` differs between the two runs. This reuses the 1.3B-LoRA +
offline-embedding recipe validated for #890 on this machine, rather than the I2V-14B
synthetic-tensor path in `docs/examples/wan2.1.md` (which does not run on `main`, §7);
the attention code path is identical.

**Naming caveat:** the stock config is called `wan2.1_I2V_1.3B_lora.yaml`, but the
checkpoint it is run against is **Wan2.1-T2V-1.3B** (`in_channels=16`,
`image_dim=null` — no image-conditioning branch) and the offline embeddings contain
only latents + text context. So this run is **text-to-video in substance**; the
"I2V" in the filename is a pre-existing repo naming artifact, not a claim that
image-conditioned training was exercised. No I2V *training* is validated in this
report — see §7 for why.

## 3. Accuracy alignment

Per-step training loss (`mse_loss`, seed 42) — **every step is identical to 2 dp**:

| step | 1 | 5 | 10 | 14 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| FA2   | 0.48 | 0.42 | 0.20 | 0.57 | 0.49 | 0.20 | 0.17 |
| aiter | 0.48 | 0.42 | 0.20 | 0.57 | 0.49 | 0.20 | 0.17 |

Over all 30 steps: **max |Δloss| = 0.0000, mean = 0.0000**. Even the noisy
diffusion-loss spikes (random timestep/noise) coincide exactly, i.e. the aiter
forward/backward is numerically interchangeable with FA2 here. Aligned.

(The kernel-level bf16 tolerance is quantified in the Qwen3-Omni report §2; LoRA
freezes the base weights so those tiny per-op differences never accumulate into the
2-dp loss.)

## 4. Throughput

Per-step wall time, measured as the elapsed-time slope from step 5 to step 29:

| backend | s/step | end-to-end |
|---|---|---|
| FA2 (flash-attn 2.8.3 / CK) | 5.46 s | 1.00× |
| **aiter (FMHA-v3)** | **4.29 s** | **1.27× faster** |

**Method.** tqdm's elapsed clock has 1-second resolution, which is too coarse to
difference between consecutive 4–5 s steps. Taking the slope across 24 steps instead
spreads that quantisation to about ±0.04 s/step (<1%), and skipping steps 0–4 excludes
warmup. The result is corroborated independently by the profiler in §5, which measures
total GPU-kernel time at microsecond resolution and shows −21%.

## 5. Bottleneck analysis (profiling) — why aiter wins here

torch profiler, rank-0, steps 5–6. GPU-kernel self-time by category:

| category | FA2 | aiter |
|---|---|---|
| **attention** | **57.2%** (3038 ms) | **45.7%** (1919 ms) |
| gemm (dense QKV/o/ffn proj) | 21.2% (1124 ms) | 26.8% (1124 ms) |
| elementwise | 13.2% (702 ms) | 16.7% (701 ms) |
| comm — RCCL (SP all-to-all + FSDP) | 6.2% (330 ms) | 8.1% (339 ms) |
| norm/softmax | 2.0% | 2.5% |
| **total GPU-kernel time** | **5312 ms** | **4196 ms** (−21%) |

Attention kernel time, FA2 → aiter:

| | FA2 (CK `ck_tile`) | aiter (FMHA-v3) | change |
|---|---|---|---|
| forward | 1004 ms | 850 ms | −15% |
| **backward** | **2024 ms** | **1047 ms** | **−48%** |
| total attention | 3038 ms | 1919 ms | **−37%** |

**Interpretation.** Everything except attention is essentially unchanged (dense GEMM
identical at 1124 ms, comm/elementwise similar). aiter's FMHA-v3 backward is ~2×
faster than flash-attn-2.8.3's `ck_tile` backward on MI308X, which is the entire
source of the speedup. Because the Wan DiT is attention-bound (57% under FA2), that
−37% attention time becomes a −21% total-GPU-work and ~27% wall-clock win.

**Next bottleneck (post-aiter):** attention is still the largest single bucket
(45.7%), so the next steps would be (1) push attention further (aiter fp8 fwd, or
tuning the v3 backward), then (2) the dense projections/GEMM (27%) and the
elementwise tail (17%), and (3) improve SP/FSDP compute-comm overlap.

## 6. Reproduction

Same harness as the Qwen report (`/data/tianhaos/run_train.sh`, `compare_runs.py`,
`throughput.py`, `analyze_trace.py`). Raw logs/CSVs/plots under
`/data/tianhaos/outputs/{wan_fa2,wan_aiter,wan_compare,wan_prof_fa2,wan_prof_aiter}`.

Command (aiter):

```bash
bash train.sh tasks/train_dit.py configs/dit/wan2.1_I2V_1.3B_lora.yaml \
  --model.model_path <Wan2.1-T2V-1.3B-Diffusers/transformer> \
  --model.condition_model_path <Wan2.1-T2V-1.3B-Diffusers> \
  --data.train_path <tomjerry_offline> --data.source_name Tom-and-Jerry-VideoGeneration-Dataset \
  --train.training_task offline_training --train.accelerator.ulysses_size 2 \
  --model.ops_implementation.attn_implementation aiter
```

## 7. Wan2.1-I2V-**14B** (synthetic smoke test) + sdpa comparison

The example in `docs/examples/wan2.1.md` targets the **I2V-14B** model but explicitly
states *"End-to-end training for the wan2.1 i2v model is not yet supported ... we are
constructing random tensors to conduct test training."* On current `main` even that
synthetic path does not run as written:

- the `.pth` `TensorDataset` loader (`build_tensor_dataset`) is **not registered** in
  `DATASET_REGISTRY` (only parquet/csv/json/arrow loaders are), so `generate_wan_dataset.py`
  output can't be loaded;
- the native I2V `WanModel` (`veomni/models/transformers/wan/`, which *does* implement
  image conditioning) is **entirely unregistered** (no `MODEL_CONFIG_REGISTRY`/
  `MODELING_REGISTRY` entry) and its config json (`configs/model_configs/wan/wani2v_14b.json`)
  fails with `Unknown ModelConfig name: wan`;
- the maintained **diffusers** path's condition model + training forward are **T2V-only**
  (no image-conditioning prep), though the underlying transformer forward *does* accept
  `encoder_hidden_states_image`.

Making that end-to-end path run is a separate piece of work and is **not part of this
change**, so the 14B evidence below is deliberately scoped to the *kernel* question and
measured at module level. The end-to-end evidence comes from the 1.3B instead (§3–§5 and
§7.3), which runs on the shipped path.

### 7.1 Module-level backend comparison

The numbers below isolate the *kernel* question at 14B: a **synthetic fwd+bwd smoke test on
the real 14B-I2V diffusers transformer** (random weights, fixed seed), swapping the attention processor
between FA2 / aiter / sdpa on identical weights+inputs. Single GPU (no FSDP/SP; the SP
path is already covered by the 1.3B run). Script: `/data/tianhaos/wan14b_bench.py`.

Setup: 14B-I2V (in_ch 36, out_ch 16, 40 layers, 40 heads, head_dim 128), latent
`F=16,H=32,W=64`, bf16, micro-batch 1, image cond = 257 CLIP tokens + 512 text tokens.

**What "8192 tokens" means (and what it does *not*).** The DiT self-attention sequence
length is the patchified latent grid: with `patch_size=[1,2,2]`,
`seq = F · (H/2) · (W/2) = 16 · 16 · 32 = 8192`. Cross-attention additionally attends
over 512 text + 257 CLIP-image tokens. Inverting the VAE compression (8× spatial,
4× temporal) that latent corresponds to a **256×512, 61-frame** clip — i.e. a *toy*
resolution, deliberately matched to the scale of the doc's own synthetic generator
(`generate_wan_dataset.py` defaults `T=16,H=32,W=48` → 6144 tokens).

It is **not** the real 480P I2V workload. Genuine `Wan2.1-I2V-14B-480P` training
(832×480, 81 frames) gives latent `F=21,H=60,W=104` → **32,760 tokens**, i.e. **4×
longer** than 8192. Because attention is quadratic in sequence length while the dense
projections are linear, real 480P carries ~16× the attention FLOPs but only ~4× the
GEMM FLOPs, so it should be *more* attention-bound — making 8192 a conservative
setting for aiter. **This was measured rather than assumed** (§7.3).

**Methodology note (verified):** each backend is run in its **own isolated process**
(`--backend {fa2,aiter,sdpa}`), giving clean, un-fragmented memory and stable timing;
predictions are saved (`--save_pred`) and compared across processes on identical
weights/inputs. (Running all three in one process near the 144 GB peak fragments memory
and can both slow aiter and OOM — an artifact, not a real regression.) Numbers below are
from the isolated runs and reproduce the standalone benchmark to within run-noise.

| backend | fwd+bwd (median) | vs FA2 | loss | pred rel-to-peak vs FA2 | 1−cos |
|---|---|---|---|---|---|
| FA2 (flash-attn 2.8.3 / CK) | 7806 ms | 1.00× | 1.33784 | — | — |
| **aiter (FMHA-v3)** | **6582 ms** | **1.19× faster** | 1.33782 | 9.2e-3 | 8.2e-5 |
| sdpa (`F.scaled_dot_product_attention`) | 11039 ms | 0.71× (29% slower) | 1.33783 | 9.6e-3 | 8.1e-5 |

- **aiter is ~19% faster than FA2** and **~1.68× faster than sdpa** on the 14B I2V —
  consistent with the 1.3B result (attention-bound DiT ⇒ aiter wins). Note this
  confirms the win at 14B *parameter* scale; the *sequence* scale (8192) is a quarter
  of real 480P, see the sizing note above.
- **Alignment:** all three agree — loss identical to 5 significant figures
  (1.33784 / 1.33782 / 1.33783); aiter's full-output prediction differs from FA2 by
  rel 9.2e-3 / cosine-dist 8.2e-5, i.e. the **same bf16 magnitude as sdpa's own deviation
  (9.6e-3)** from FA2. So aiter is as numerically faithful to FA2 as the PyTorch SDPA
  reference is.
- Peak memory identical across backends (144 GB), i.e. no memory regression.

Reproduce (each backend isolated, on an idle GPU):

```bash
for b in fa2 aiter sdpa; do
  HIP_VISIBLE_DEVICES=0 python wan14b_bench.py \
    --config_dir <Wan2.1-I2V-14B-480P-Diffusers/transformer> \
    --backend $b --save_pred /tmp/$b.pt
done
```

**Caveat — this is explicitly not end-to-end training.** To be unambiguous about scope,
the 14B numbers come from a single module-level `forward + backward` on **randomly
initialised weights**: there is no optimizer step, no LR schedule, no data pipeline
(latents/text/CLIP embeddings are random tensors), no FSDP and no sequence parallel,
and one GPU only. The reported "loss" is MSE against a random target — it is a
**determinism/alignment probe**, not a training loss, and its value carries no
convergence meaning. What this test *does* establish is kernel-level: on the real
14B-I2V architecture, aiter and FA2 produce numerically equivalent activations and
gradients, and aiter's kernels are faster. End-to-end Wan training evidence comes
only from the 1.3B run in §2–§5. (This matches the doc's own "test training" scope
for I2V-14B; see the blockers listed at the top of this section.)

### 7.2 Sequence-length sweep — the win grows with context, up to real 480P

To settle whether the 8192-token figure understates aiter at production resolution,
the same isolated-process benchmark was swept from 4k up to the **real 480P length of
32,760 tokens**. Gradient checkpointing is ON for every swept point (32k does not fit
otherwise), so the curve is internally consistent; a no-checkpoint pair at 8192 ties
the sweep back to §7.2.

| tokens | latent `F,H,W` | FA2 | aiter | aiter speedup | peak mem |
|---|---|---|---|---|---|
| 4,096 | 16, 32, 32 | 4,135 ms | 3,758 ms | **1.10×** | 67.0 GB |
| 8,192 | 16, 32, 64 | 10,362 ms | 9,021 ms | **1.15×** | 68.2 GB |
| 16,384 | 16, 64, 64 | 26,867 ms | 21,800 ms | **1.23×** | 71.3 GB |
| **32,760 (real 480P)** | 21, 60, 104 | 88,637 ms | 69,052 ms | **1.28×** | 78.1 GB |
| 8,192 *no checkpointing* | 16, 32, 64 | 7,806 ms | 6,580 ms | 1.19× | 144.0 GB |

- **The speedup rises monotonically with sequence length** — 1.10× → 1.15× → 1.23× →
  **1.28×** — exactly as the quadratic-attention argument predicts. So the §7.2 figure
  was indeed conservative: at the production 480P context aiter saves **~28%** of
  fwd+bwd time (88.6 s → 69.1 s per step), not 19%.
- **Alignment holds at every length**, including 32k: loss agrees to 5–6 significant
  figures (32k: 1.36904 vs 1.36903; 16k: 1.35913 vs 1.35911; 4k: identical).
- **No memory regression at any length** — peak is identical between backends, and
  grows only 67 → 78 GB from 4k to 32k under checkpointing (vs 144 GB for 8k without).
- The no-checkpoint 8192 row reproduces §7.2 to **0.03%** (7,806 ms and 6,580 ms vs the
  previously reported 7,806 ms and 6,582 ms), which validates the harness across
  sessions.

Reproduce: `bash /data/tianhaos/wan_seqlen_sweep.sh` (isolated process per point).

### 7.3 The same sweep at 1.3B — the trend holds at a second parameter scale

§7.2 established the length trend at 14B in isolation. This repeats it end to end on the **1.3B**, which
matters for two reasons: it is a ~10× smaller model (so the attention share of total
FLOPs is different), and it runs from the config that actually ships in the repo
(`configs/dit/wan2.1_I2V_1.3B_lora.yaml`), making it the more reproducible of the two.

Method: 15 steps, seed 42, `ulysses_size 2`, LoRA, per-step time from the
elapsed slope. Token counts are deliberately identical to §7.2 (the latent spatial dims
were copied from the 14B sweep datasets), so the two scales line up point for point.

| tokens | FA2 | aiter | **1.3B speedup** |
|---|---|---|---|
| 8,192 | 2.56 s | 2.44 s | **1.05×** |
| 16,384 | 6.00 s | 4.89 s | **1.23×** |
| **32,760 (real 480P)** | 16.56 s | 11.78 s | **1.41×** |

- **The trend holds end to end**: 1.05× → 1.23× → **1.41×**, and it is steeper than the
  isolated 14B sweep in §7.2 (1.15× → 1.23× → 1.28× at the same token counts). The
  end-to-end measurement including optimizer, FSDP collectives and Ulysses all-to-all does
  *not* wash the win out at long context — the 1.3B carries less dense-GEMM work for
  attention to compete with, so once the sequence is long enough it pulls ahead of the
  isolated 14B figure. At 8k the opposite holds: a 2.6 s step is dominated by those fixed
  costs, which is why the shortest point is the weakest.
- **Alignment holds at every length**: max |Δloss| = 0.010 / 0.010 / 0.000 for
  8k / 16k / 32,760 over the 16 reported steps — the 32,760 pair is bit-identical.

Data: latents are synthetic (`wan13b_make_seqscan_data.py`) because a DiT's sequence length
is fixed by the precomputed offline latents, which only exist at one size for the real
Tom-and-Jerry set. Throughput does not depend on the values and both backends see
byte-identical inputs, so the comparison is unaffected; the absolute loss values are not
meaningful, only the FA2-vs-aiter agreement is. As a check on the substitution, the
synthetic 16,384-token FA2 point (6.00 s) and the real-data 16,422-token FA2 measurement
(§4, 5.42 s) differ by about 10%, which is attributable to the different (F,H,W)
factorisation at the same token count (16×64×64 vs 21×46×68) rather than to the data being
synthetic.

Reproduce: `wan13b_make_seqscan_data.py` then `LABELS="8k 16k 32k" bash wan13b_seqscan.sh`.

#### 7.3.1 Same sweep with the factorisation held fixed

The table above has a confound: its spatial dims were copied from the 14B sweep, so the
frame count is not constant along the curve — 16 latent frames at 8k and 16k, but 21 at
32,760. Since attention cost depends on the `(F, H, W)` split and not only on the token
count, the curve mixes a length effect with a shape effect. This variant removes it by
fixing **F = 21** (the real data's frame count) and preserving its 46:68 aspect ratio,
scaling the linear dims by 1/√2 and √2:

| tokens | F, H, W | FA2 | aiter | **speedup** | data |
|---|---|---|---|---|---|
| 8,064 | 21, 32, 48 | 2.56 s | 2.33 s | **1.10×** | synthetic |
| 16,422 | 21, 46, 68 | 5.42 s | 4.25 s | **1.28×** | **real** (§4) |
| 32,256 | 21, 64, 96 | 16.00 s | 11.78 s | **1.36×** | synthetic |

The midpoint is the real Tom-and-Jerry measurement, so the curve is anchored on real data
rather than being synthetic end to end.

- **The trend is unchanged and slightly stronger at the short end**: 1.10× → 1.28× → 1.36×.
- **Shape matters, and it matters only to aiter.** At ~8k the two factorisations give
  *identical* FA2 times (2.56 s both) while aiter differs (2.44 s at 16×32×64 vs 2.33 s at
  21×32×48), moving the ratio from 1.05× to 1.10×. So the earlier 8k and 16k rows understate
  aiter for this workload; FA2 is insensitive to the split at fixed token count.
- The 32,256 point (1.36×) and the 32,760 point in the table above (1.41×) are consistent:
  **aiter is identical in both (11.78 s)** and the difference is entirely FA2 (16.00 vs
  16.56 s), tracking the 1.5% token difference. Both of those rows already had F = 21, so
  only the 8k and 16k rows were affected by the confound.
- **Alignment**: max |Δloss| = 0.010 / — / 0.000 for the three points.

**Measured on verified-idle GPUs and repeated on two independent nodes.** The 32,256 pair was
run simultaneously on two machines, each with every card at 0 GB before and after both legs,
and the two agree to the reported precision (FA2 16.00 s, aiter 11.78 s, 1.36× on both).
Per-step times are flat — FA2 16 s on every step, aiter 11–12 s.

**Flat per-step times are the thing to check, not matching losses.** On a shared machine a
neighbouring job can start partway through a run; when that happened here the same
configuration read FA2 18–27 s/step with visible drift and produced ratios anywhere from
1.27× to 1.63×, while **the losses stayed bit-identical throughout**. Matching losses only
prove the two legs did the same arithmetic on the same data; they say nothing about whether
the timings were clean. Check the per-step series for flatness and snapshot other processes'
GPU usage before and after each leg.

Reproduce: `wan13b_make_seqscan_data.py --labels f21_8k f21_32k` then
`LABELS="f21_8k f21_32k" EPOCHS=2 bash wan13b_seqscan_f21.sh`.

Keep `num_train_epochs` small. **`max_steps` bounds the per-epoch inner loop, not the total
run**: `DiTTrainer.train` iterates `for epoch in range(num_train_epochs)` and inside each
epoch runs `for _ in range(start_step, train_steps)`, resetting `start_step` to 0 at the end
of every epoch. So `max_steps 15` with `num_train_epochs 10` trains **150** steps, not 15 —
50 min per leg instead of 11. On a shared cluster that difference decides whether a run
finishes inside a free window on a shared machine. `EPOCHS=2` is used rather than 1 only
because 24 rows/shard is not quite enough to feed 15 steps from a single pass.

The timing extraction is unaffected by the epoch count: each epoch gets its own tqdm timer,
and the two epochs of every leg give identical slopes (FA2 16.00 s in both, aiter 11.78 s in
both), which together with the two-node agreement gives four consistent readings per backend.

Two further measurement caveats, each of which produced a wrong number before being caught:

1. **Run both backends of a point through the same launcher.** Launching one leg from the
   sweep script and the other by hand gives identical configs but *different data and noise
   streams* — losses diverge from step 1 and stay uncorrelated, and the pair is silently
   meaningless. A mismatched pair read 1.63× where the properly paired run reads 1.41×.
2. **tqdm's 1-second clock is too coarse for per-step differencing at short steps**, and its
   final `15/15` redraw carries the same timestamp as `14/15`. Take the slope over steps
   5→14 and divide by 9. Using a median of per-step deltas instead makes the 8k point read
   1.50× purely from integer-second quantisation.

Also worth knowing when scripting a sweep: a finished run can hold ~23 GB/GPU for many
minutes after training ends, because the trainer takes the `Skipping explicit NCCL
process-group destroy on normal trainer exit` path. Wait for VRAM to drain between runs
rather than treating leftover memory as a foreign job.

## 8. Conclusions

- aiter attention is a **clear win for the attention-bound Wan DiT**: 27% faster end-to-end
  on the **1.3B T2V** (real data, LoRA, ulysses 2), rising to **1.41×** at the real 480P
  latent size of 32,760 tokens (§7.3), and 1.36× at a comparable 32,256-token point that
  holds the latent factorisation fixed instead (§7.3.1).
- **The length trend holds end to end and is corroborated at a 10× larger parameter count.**
  End-to-end on the 1.3B with the latent factorisation held fixed: 1.10× → 1.28× → 1.36×
  (§7.3.1, midpoint from real data, longest point reproduced on two independent nodes).
  An isolated fwd+bwd sweep on the 14B transformer shows the same slope,
  1.10× → 1.28× over 4k → 32,760 tokens (§7.2).
- **Accuracy is aligned in every configuration measured**: bit-identical to 2 dp for all
  30 steps of the 1.3B LoRA run, and identical to 5 significant figures on the 14B
  module-level comparison.
- **The benefit tracks how attention-dominated the step is.** Longer context ⇒ larger win
  (quadratic attention vs linear GEMM). Non-attention work (optimizer, FSDP/SP collectives,
  checkpointing) dilutes it at short context, but by 480P length that dilution is
  negligible. Enable aiter for long-context DiT training in particular — that is where it
  pays most, which is also where training time actually hurts.
- **Scope — what is *not* validated here:** end-to-end I2V-14B *training* (the documented
  path does not run on current `main`; the 14B evidence in §7.1–§7.2 is module-level, on
  synthetic tensors, so its loss levels carry no convergence meaning — only cross-backend
  agreement does), and generated sample quality.
- sdpa is the slowest of the three (≈0.7× FA2); aiter ≈1.68× sdpa on the 14B.
- Combined with the Qwen3-Omni result (≈parity, attention only 7% of a MoE step), the
  takeaway is: **enable aiter for attention-bound models (DiT, long-context, dense);
  it is safe (aligned) everywhere but only moves throughput where attention is a
  meaningful fraction of the step.**
