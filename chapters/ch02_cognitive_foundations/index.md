---
title: "2. Cognitive Foundations for Safe Driving Interaction"
parent: Chapters
nav_order: 2
---

# Chapter 2: Cognitive Foundations for Safe Driving Interaction
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 2` from the repository root; outputs go to `outputs/ch02/`.

## Case Study 1: Measuring Driver Workload Using NASA-TLX and Eye Tracking

Simulate a within-subject work-zone simulator study, compare a revised HMI with the baseline segment by segment, and test the longest off-road glance at the taper against the 2.0 s criterion.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_workload_tlx_eye_tracking.py](cs1_workload_tlx_eye_tracking.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch02_cognitive_foundations/cs1_workload_tlx_eye_tracking.py){: .btn .fs-3 }

![Measuring Driver Workload Using NASA-TLX and Eye Tracking]({{ '/docs/figures/fig02_06_cs1_workload.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_workload_tlx_eye_tracking.py %}
```

## Case Study 2: Trust Calibration in Automated Driving Decision Support

Model reliance on a pedestrian-conflict alert that stays silent behind a stopped bus, and show which outcome measure reveals what an interface change actually does.

**Libraries:** NumPy, pandas, matplotlib

[Download cs2_trust_calibration.py](cs2_trust_calibration.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch02_cognitive_foundations/cs2_trust_calibration.py){: .btn .fs-3 }

![Trust Calibration in Automated Driving Decision Support]({{ '/docs/figures/fig02_07_cs2_trust.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_trust_calibration.py %}
```
