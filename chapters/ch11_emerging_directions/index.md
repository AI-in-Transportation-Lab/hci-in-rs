---
title: "11. Emerging Directions in HCI for Roadway Safety"
parent: Chapters
nav_order: 11
---

# Chapter 11: Emerging Directions in HCI for Roadway Safety
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 11` from the repository root; outputs go to `outputs/ch11/`.

## Case Study 1: Adaptive AI Copilots for Personalized Roadway Safety Interfaces

Compare one drowsiness threshold with personal baselines, decide when to ask and when to act from expected utility, and learn each driver's best alert with Thompson sampling.

**Libraries:** NumPy, pandas, SciPy, scikit-learn, matplotlib

[Download cs1_adaptive_copilot.py](cs1_adaptive_copilot.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch11_emerging_directions/cs1_adaptive_copilot.py){: .btn .fs-3 }

![Adaptive AI Copilots for Personalized Roadway Safety Interfaces]({{ '/docs/figures/chap11_fig01_copilot.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_adaptive_copilot.py %}
```

## Case Study 2: AR/VR Training Environments for Safety Evaluation

Estimate intention-to-treat and per-protocol effects of VR hazard training with sickness dropout, transfer to road practice, and skill decay after training.

**Libraries:** NumPy, pandas, SciPy, statsmodels, matplotlib

[Download cs2_vr_training.py](cs2_vr_training.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch11_emerging_directions/cs2_vr_training.py){: .btn .fs-3 }

![AR/VR Training Environments for Safety Evaluation]({{ '/docs/figures/chap11_fig02_vr_training.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_vr_training.py %}
```
