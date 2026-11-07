---
title: "6. Full Autonomy and Human-Machine Interaction (SAE Levels 4-5)"
parent: Chapters
nav_order: 6
---

# Chapter 6: Full Autonomy and Human-Machine Interaction (SAE Levels 4-5)
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 6` from the repository root; outputs go to `outputs/ch06/`.

## Case Study 1: eHMI Intent Signaling for Pedestrian Interaction with Autonomous Vehicles

Simulate a VR crossing experiment with kinematics only, a reliable eHMI and an eHMI with two faults, and track crossing times, trust, false starts and discrimination of yielding intent.

**Libraries:** NumPy, pandas, SciPy, statsmodels, matplotlib

[Download cs1_ehmi_crossing.py](cs1_ehmi_crossing.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch06_full_autonomy/cs1_ehmi_crossing.py){: .btn .fs-3 }

![eHMI Intent Signaling for Pedestrian Interaction with Autonomous Vehicles]({{ '/docs/figures/chap06_fig01_ehmi_crossing.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_ehmi_crossing.py %}
```

## Case Study 2: Safety Assurance and Standardization for SAE Level 4-5 Deployment

Compare a robotaxi fleet's injury-crash rate with human benchmarks four ways, monitor it with a Bayesian posterior in an expansion area, and check scenario coverage against tolerable failure rates.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs2_safety_assurance.py](cs2_safety_assurance.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch06_full_autonomy/cs2_safety_assurance.py){: .btn .fs-3 }

![Safety Assurance and Standardization for SAE Level 4-5 Deployment]({{ '/docs/figures/chap06_fig02_safety_assurance.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_safety_assurance.py %}
```
