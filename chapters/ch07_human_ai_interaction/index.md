---
title: "7. Human-AI Interaction in Safety-Critical Transportation Systems"
parent: Chapters
nav_order: 7
---

# Chapter 7: Human-AI Interaction in Safety-Critical Transportation Systems
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 7` from the repository root; outputs go to `outputs/ch07/`.

## Case Study 1: Explainable AI Dashboards for Transportation Safety Decision Support

Rank 1,200 rural segments with a crash-rate model, show uncertainty, global and local attributions and what-if options, and compare each view with the true process.

**Libraries:** NumPy, pandas, SciPy, scikit-learn, matplotlib

[Download cs1_xai_dashboard.py](cs1_xai_dashboard.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch07_human_ai_interaction/cs1_xai_dashboard.py){: .btn .fs-3 }

![Explainable AI Dashboards for Transportation Safety Decision Support]({{ '/docs/figures/chap07_fig01_xai_dashboard.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_xai_dashboard.py %}
```

## Case Study 2: Bias, Accountability, and Transparency in AI-Enabled Road Safety Systems

Audit a crash-risk model trained on under-reported police data for calibration and equal opportunity across two communities, and simulate the feedback loop of allocating by detections.

**Libraries:** NumPy, pandas, SciPy, scikit-learn, matplotlib

[Download cs2_bias_audit.py](cs2_bias_audit.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch07_human_ai_interaction/cs2_bias_audit.py){: .btn .fs-3 }

![Bias, Accountability, and Transparency in AI-Enabled Road Safety Systems]({{ '/docs/figures/chap07_fig02_bias_audit.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_bias_audit.py %}
```
