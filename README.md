# Diffusion-Based Next-Frame Prediction for Visual Stories

## Overview

This project studies next-frame prediction for visual stories, comparing a supervised CNN-GRU baseline with a conditional diffusion model. The goal is to generate sharper, context-aware frames by conditioning on the previous K frames.

## Method

- Dataset: StoryReasoning (image sequences).
- Baseline: CNN frame encoder + GRU over context + CNN decoder.
- Diffusion: conditional UNet-lite denoiser with a CNN+GRU context encoder and sinusoidal time embeddings.
- Metrics: L1, MSE, PSNR, and temporal consistency (TC2) against the last context frame.

## Quickstart

```bash
cd "Deepika"

# Baseline training
python src/train_diffusion.py --mode baseline --epochs 2

# Diffusion training
python src/train_diffusion.py --mode diffusion --epochs 3 --t_steps 100
```

## Notes

- Images are normalized to [-1, 1] at 128x128 by default.
- Diffusion sampling is slower; use small subsets for quick evaluation.

## Results

Table 1: Next-frame prediction metrics (subset evaluation)

| Model | L1 | MSE | PSNR |
| --- | --- | --- | --- |
| Baseline GRU encoder-decoder | 0.3248 | 0.2163 | 12.67 |
| Diffusion from noise (DDPM sampling) | 0.5510 | 0.5432 | 8.67 |
| Diffusion refiner (t_start=30) | 0.3502 | - | 12.14 |

Table 2: Temporal consistency (refinement mode)

| Metric | Value |
| --- | --- |
| delta_pred | 0.0679 |
| delta_true | 0.3557 |
| delta_gap | 0.2878 |

## Findings

The diffusion model trained successfully under the noise-prediction objective (validation noise MSE decreased to approximately 0.0287 at epoch 2). However, generating the next frame from pure noise using standard DDPM reverse diffusion performed poorly (L1 around 0.55 and PSNR around 8.67 on a validation subset). This shows that under limited training budget and a lightweight UNet, full generative sampling was not strong enough for this dataset.

Using the diffusion model as a refinement module (starting from the last observed frame with limited noise and running partial denoising) produced much better outputs than pure sampling. The best refinement setting (t_start=30) achieved L1 around 0.350 and PSNR around 12.14. This indicates the model can denoise and sharpen frames when anchored to a realistic starting point.

Despite stable refinement, the baseline encoder-GRU-decoder remained more accurate on next-frame prediction (baseline L1 around 0.325 and PSNR around 12.67). The temporal-motion metrics explain why: the diffusion refiner predicted a very small change relative to the last frame (delta_pred about 0.068), while the true next-frame change magnitude was much larger (delta_true about 0.356). Therefore the model produced temporally smooth outputs but under-modeled real motion, which increases error when the scene changes.

Overall conclusion: diffusion objectives learned useful denoising behavior, but with the current architecture and training budget the method behaved as a conservative "stability prior" (copy-last-frame tendency). This improves visual smoothness qualitatively but does not outperform a direct supervised baseline for accuracy. A stronger diffusion model would likely require more epochs, EMA, and/or an auxiliary reconstruction loss to model motion magnitude.

## Repository Layout

- `src/dataloader.py`: indexing and temporal image dataset with preprocessing.
- `src/model_baseline_cnn.py`: GRU baseline + metrics helpers.
- `src/context_encoder.py`: CNN+GRU context encoder.
- `src/diffusion_model_wrapper.py`: diffusion schedule, UNet-lite, and conditional model.
- `src/model_diffusion.py`: DDPM/DDIM/refine sampling helpers.
- `src/train_diffusion.py`: training for baseline or diffusion.
- `src/eval_diffusion.py`: evaluation utilities for baseline and diffusion sampling.
