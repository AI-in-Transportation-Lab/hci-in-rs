---
title: "10. Evaluation Methods for Safety-Critical HCI"
parent: Chapters
nav_order: 10
---

# Chapter 10: Evaluation Methods for Safety-Critical HCI
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 10` from the repository root; outputs go to `outputs/ch10/`.

## Case Study 1: Simulator and Field Evaluation of Distraction, Workload, and Trust

Run NHTSA-style glance tests on touchscreen and voice entry, compute how often such a test passes, relate simulator and track effects, and size a track study.

**Libraries:** NumPy, pandas, SciPy, statsmodels, matplotlib

[Download cs1_simulator_field.py](cs1_simulator_field.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch10_evaluation_methods/cs1_simulator_field.py){: .btn .fs-3 }

![Simulator and Field Evaluation of Distraction, Workload, and Trust]({{ '/docs/figures/chap10_fig01_sim_field.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_simulator_field.py %}
```

## Case Study 2: Translating Naturalistic Driving Evidence into Safety Claims

Estimate the odds ratio for visual-manual phone use seven ways, correct AI-coded labels for misclassification, and bootstrap the share of events prevented by halving phone use.

**Libraries:** NumPy, pandas, SciPy, scikit-learn, statsmodels, matplotlib

[Download cs2_nds_claims.py](cs2_nds_claims.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch10_evaluation_methods/cs2_nds_claims.py){: .btn .fs-3 }

![Translating Naturalistic Driving Evidence into Safety Claims]({{ '/docs/figures/chap10_fig02_nds_claims.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_nds_claims.py %}
```
