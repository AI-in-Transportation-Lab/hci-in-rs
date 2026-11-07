---
title: "5. Driver Assistance and Partial Automation (SAE Levels 1-3)"
parent: Chapters
nav_order: 5
---

# Chapter 5: Driver Assistance and Partial Automation (SAE Levels 1-3)
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 5` from the repository root; outputs go to `outputs/ch05/`.

## Case Study 1: Driver Monitoring and Engagement in SAE Level 2 Automation

Compare no monitoring, hands-on-wheel checks and camera monitoring with escalation by long glances away, tune the first alert, and show how correlated errors defeat a frame filter.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_driver_monitoring.py](cs1_driver_monitoring.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch05_driver_assistance/cs1_driver_monitoring.py){: .btn .fs-3 }

![Driver Monitoring and Engagement in SAE Level 2 Automation]({{ '/docs/figures/chap05_fig01_monitoring.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_driver_monitoring.py %}
```

## Case Study 2: Takeover Request Timing and Transition Design for SAE Level 3 Systems

Model takeover times by secondary task, find the lead time a takeover request needs, the collision risk at speed, and the sample size of a 95/95 validation test.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs2_takeover_timing.py](cs2_takeover_timing.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch05_driver_assistance/cs2_takeover_timing.py){: .btn .fs-3 }

![Takeover Request Timing and Transition Design for SAE Level 3 Systems]({{ '/docs/figures/chap05_fig02_takeover.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_takeover_timing.py %}
```
