# Stage 2 QLoRA Fine-tuning — Debugging Summary

## Setup
- Kaggle notebook, GPU accelerator = Tesla T4 x2, but only 1 GPU is actually used (`device_map={"": 0}`).
- Model: `google/gemma-4-E2B`, loaded in 4-bit (bitsandbytes NF4) via `BitsAndBytesConfig`, `bnb_4bit_compute_dtype=torch.float16`.
- Continuing a Stage 1 LoRA adapter (`PeftModelForCausalLM.from_pretrained(model, stage1_adapter_dir, is_trainable=True)`) on a new bidirectional (Darija<->English) translation dataset (~96.5k train examples).
- Training via `trl.SFTTrainer` / `SFTConfig`.

## Problem 1: Extremely slow training
Original config (`bf16=True`, `packing=False`) projected **~49 hours** for 3016 steps (~0.02 it/s) on a single T4. This is many orders of magnitude slower than expected for a small model + LoRA + 4-bit quantization on a T4.

## Problem 2: Crash when trying fp16
Switching to `fp16=True, bf16=False` (to work around suspected bf16-on-T4 slowness) crashed with:
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```
This is the fp16 `GradScaler`'s `unscale_` step encountering a bf16 tensor somewhere in the trainable parameters (most likely the Stage 1 adapter checkpoint's LoRA weights, which appear to have been saved as bf16 and don't get force-cast just because the surrounding config says fp16).

## Things tried, and what actually happened

| # | Change | Result |
|---|--------|--------|
| 1 | Cast all `requires_grad=True` (LoRA) params to `torch.float32` right after `PeftModelForCausalLM.from_pretrained(..., is_trainable=True)`, keep `fp16=True, bf16=False` | **Crash still occurred.** Either the cast wasn't reaching every tensor GradScaler touches (PEFT's adapter-loading path may reassign parameter dtype in a way that isn't caught by a single post-load `.to(fp32)` sweep), or some other trainable tensor besides the obvious LoRA `named_parameters()` set was still bf16. Root cause of *why* the cast didn't fully take was not conclusively identified. |
| 2 | Disable Trainer-level AMP entirely: `fp16=False, bf16=False` | **Crash gone**, but **training got even slower** (0.04 it/s, ~7.3h for 1134 steps after also enabling packing — worse than expected given packing should have helped). Hypothesis: with no `autocast` context, normal PyTorch dtype-promotion rules apply. The frozen 4-bit base layers still compute in fp16 (via `bnb_4bit_compute_dtype`), but the fp32 LoRA branch output gets added to that fp16 output, and the sum is silently promoted to fp32 (`fp16 + fp32 → fp32`). That fp32 tensor then becomes the input to the *next* transformer layer, cascading fp32 through the entire rest of the network layer by layer — effectively the whole model ends up computing in fp32 despite the 4-bit quantization and compute_dtype settings. **This hypothesis was never directly confirmed with profiling/dtype-tracing — it's a plausible explanation, not a verified root cause.** |
| 3 | Re-enable autocast via `bf16=True` (not fp16, so no `GradScaler` is ever created — bf16 doesn't need loss scaling) + switch `optim="paged_adamw_8bit"` → `optim="adamw_8bit"` (paged optimizers can silently page to CPU RAM under VRAM pressure, causing large slowdowns with no visible error) + added a diagnostic cell (checks parameter `.device`, trainable/frozen param dtypes, and `nvidia-smi` utilization) to be run before committing to a full training run | **Not yet tested by the user** — conversation ended here to bring in another model. |

## Other changes made along the way (status unclear / not isolated)
- `packing=True` was enabled in `SFTConfig` (previously `False`) to reduce padding waste, cutting step count from 3016 → 1134. **Not verified in isolation** — it's possible packing increased per-step sequence length enough (every packed sequence now fills the full `max_length=256`, vs. shorter dynamic-padding batches before) that it partially offset its own benefit. Worth re-testing with `packing=False` once the dtype/precision issue is actually resolved, to isolate its real effect.
- A cell already existed that selectively upcasts small non-4bit base-model params to fp32 (`SKIP_UPCAST` set) *before* the adapter is loaded — this logic predates this debugging session and was not changed, but is a candidate suspect if fp32 tensors are still leaking into the forward pass somewhere.
- Only 1 of the 2 available T4 GPUs is used (`device_map={"": 0}`). This was **not addressed at all** — real multi-GPU (DDP via `accelerate launch` / `notebook_launcher`) was suggested as a next step but never implemented, and is a separate, structurally bigger change from the precision/speed issue above.

## What was NOT tried / not verified
- No direct profiling was done (no `torch.profiler`, no step-by-step timing breakdown of forward/backward/optimizer) to actually pinpoint where the time is going. All fixes so far have been based on reasoning about known failure modes (AMP/dtype interactions, paged optimizer CPU offload), not measurement.
- Never confirmed whether the GPU is even being utilized during the slow steps (e.g., `nvidia-smi` polled *during* a training step, not just before). The newly-added diagnostic cell checks this before training starts, not during.
- Never checked whether `gradient_checkpointing=True` (with `use_reentrant=False`) is contributing meaningfully to the slowdown — it trades ~20-30% extra compute for memory savings, which shouldn't cause 40-100x slowness on its own, but wasn't isolated/tested with it disabled.
- Never checked bitsandbytes/CUDA/driver version compatibility on the Kaggle image, or whether the 4-bit kernels are actually dispatching to the fast path vs. some slow fallback (e.g., due to `transformers==5.16.1` being a very new/unusual pinned version that may have compatibility quirks with `peft`/`bitsandbytes`/`trl`).
- Never isolated the effect of `packing=True` from the precision changes — both were changed in the same round.
- Multi-GPU (using both T4s) was identified as a possible major win but never implemented or tested.

## Suggested next steps for the more powerful model
1. Get a real profile (even just per-step wall-clock breakdown, or `torch.cuda.synchronize()` + `time.time()` around forward/backward/optimizer.step()) instead of continuing to guess from symptoms.
2. Check `nvidia-smi` **during** an actual slow step (e.g., `watch -n1 nvidia-smi` in a parallel terminal, or log GPU utilization periodically during `trainer.train()`) to confirm the GPU is actually busy and not idle/CPU-bound.
3. Verify bitsandbytes is dispatching 4-bit matmuls to its fast CUDA kernels on this specific driver/CUDA version combo on Kaggle's T4 image — this pinned `transformers==5.16.1` is unusual and worth checking for known issues with `bitsandbytes`/`peft`/`trl` compatibility.
4. Re-isolate `packing=True` vs `False` once precision is settled, rather than changing both at once.
5. If GPU-bound and dtype is confirmed clean (all frozen params fp16, no stray fp32/bf16), consider real DDP across both T4s as the next lever — but only after single-GPU speed is understood, since DDP won't fix an underlying per-step inefficiency, it'll just run the same slow thing twice in parallel.

---

## RESOLUTION (applied to `stage2.ipynb`)

Root causes identified and fixed. Chosen strategy: **single T4, done right** (DDP deferred).

### 1. Crash (`..._unscale_cuda not implemented for 'BFloat16'`) — FIXED
Confirmed by inspecting the checkpoint: **all 410 tensors in the Stage-1 adapter are BF16.** Fix: cast every `requires_grad` param to fp32 right after `PeftModelForCausalLM.from_pretrained(..., is_trainable=True)`, followed by a hard `assert`.

**...but that was not the whole story.** The full traceback showed the crash is in
`_clip_grad_norm` → `accelerator.clip_grad_norm_` → `unscale_gradients()` → `scaler.unscale_(opt)`.
`unscale_` walks the **optimizer's** params, which the assert had already proven were fp32. So an fp32 *parameter* was producing a **bf16 gradient** — which only happens when the autocast context is bf16 while the GradScaler is fp16.

**Leading root cause: `AcceleratorState` is a process-global singleton.** Building a Trainer with `bf16=True` earlier in the *same kernel* pins `mixed_precision="bf16"`; a later Trainer with `fp16=True` does **not** reset it, so you get bf16 autocast + fp16 GradScaler. This explains why casting the LoRA weights to fp32 never helped, and why the failure survived every config change — it was kernel state, not notebook state.

Handled in the trainer cell:
- `os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"` before the Trainer is built — an env var overrides `TrainingArguments`, so this has to be set first.
- `AcceleratorState._reset_state(reset_partial_state=True)` to clear the singleton.
- A **precision smoke test cell** that prints every input deciding the autocast dtype (args, env, accelerator state, full param-dtype histogram), then runs one real forward/backward and reports actual gradient dtypes. It *reports* rather than crashes, so one run yields the full picture.

**CONFIRMED ROOT CAUSE (measured).** The smoke test's param-dtype histogram settled it:

```
frozen    torch.float16      2751.5M     <- base model correct
frozen    torch.uint8        1168.3M     <- 4-bit quantized, correct
trainable torch.bfloat16       24.2M     <- THE BUG
accelerator.mixed_precision : fp16       <- autocast correct
training_args.fp16: True / bf16: False   <- config correct
```

The **trainable LoRA params are bf16 at training time** — 24.2M, matching the 410 bf16 tensors in the Stage-1 checkpoint. Under fp16 autocast a bf16 *weight* produces a bf16 *gradient*, and the fp16 GradScaler then raises the error in `unscale_()`. Everything else (autocast dtype, env, accelerator state, base-model dtype) was already correct — three intermediate hypotheses (adapter-weights-on-disk, `AcceleratorState` singleton, `ACCELERATE_MIXED_PRECISION` env) were all **wrong** and are recorded here so they are not re-tried.

The prep cell's cast to fp32 *does* run and *does* pass its assert, but something in the PEFT/TRL init path restores the adapters to the checkpoint dtype afterwards. Fix: cast at the **last possible moments**, in the trainer cell —
1. `force_fp32_trainable(trainer.model)` immediately after `SFTTrainer.__init__` (where the revert happens), and
2. a `ForceFp32Adapters` callback re-casting in `on_train_begin`, which runs *after* the optimizer is built.

`.data` is mutated in place, so the optimizer keeps referencing the same `Parameter` objects either way. Both sites assert afterwards.

**Dead end — do not retry:** `p.register_hook(lambda g: g.to(torch.float32))` to coerce grads. PyTorch rejects it outright: `RuntimeError: hook has changed the type of value (was CUDABFloat16Type got torch.cuda.FloatTensor)`. Tensor hooks may not change gradient dtype. (Its one virtue: the error message is what proved the grads were bf16.)

### 2. Extreme slowness — FIXED (root cause: never hit fp16 tensor cores)
Neither prior config used the T4's fast path:
- `bf16=True` → T4 (Turing, sm_75) has **no bf16 tensor cores** → emulated math (~50-100x slower). **The summary's "next step #3" to re-enable `bf16=True` was a trap and would reproduce the 49h estimate.**
- `fp16=False, bf16=False` → no autocast → fp32 LoRA branch cascades fp32 through the network → fp32 matmuls (no tensor cores).
- **Fix:** `fp16=True, bf16=False`. Autocast runs matmuls in fp16 (tensor cores = fast); GradScaler unscales the now-fp32 LoRA grads (no crash). This is the canonical QLoRA-on-T4 recipe.

### 3. Silent quality bug the prior session missed — FIXED
The trainer log warned: `packing=True` + `sdpa` → **cross-contamination between samples** (attention bleeds across unrelated translation pairs). Safe packing needs FlashAttention, which is unavailable on T4. Fix: `packing=False` + `group_by_length=True` — kills padding waste without contamination.

### 4. Other changes
- **Length filter, not truncation:** examples > `MAX_SEQ_LEN` (512 tok, ~95% kept) are dropped, so the model is never taught to emit half-finished translations.
- **Cheaper eval:** eval set capped at 1000 examples, `eval_steps`/`save_steps` → 200 (full 5% made each eval nearly as costly as ~100 train steps).
- `optim` `paged_adamw_8bit` → `adamw_8bit` (plenty of free VRAM; avoid silent CPU paging).
- `dataset_num_proc` 1 → 4; added `warmup_ratio=0.03`.
- Added a **dtype/device diagnostic cell** (before training) and a **`SpeedCallback`** that prints s/step for the first 8 steps — so speed is verifiable, not guessed. Expect a few s/step, not ~50.
- Inference cell: re-enable `use_cache`, `.eval()`, `torch.inference_mode()`.

### Not done (by design)
Multi-GPU DDP across both T4s — deferred. The single-GPU precision fix addresses the orders-of-magnitude problem; DDP is an optional ~2x on top and needs the loop extracted to an `accelerate launch`/`torchrun` script (notebook DDP is fragile once CUDA is initialized).