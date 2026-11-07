---
title: "3. Distraction and Secondary Task Interaction"
parent: Chapters
nav_order: 3
---

# Chapter 3: Distraction and Secondary Task Interaction
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 3` from the repository root; outputs go to `outputs/ch03/`.

## Case Study 1: Smartphone and Infotainment Distraction in Urban Driving

Compare four destination-entry designs on glances, detection-response cost, exposure-weighted delay and driver preference, and show that the winner depends on the measure.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_infotainment_distraction.py](cs1_infotainment_distraction.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch03_distraction/cs1_infotainment_distraction.py){: .btn .fs-3 }

![Smartphone and Infotainment Distraction in Urban Driving]({{ '/docs/figures/fig03_06_cs1_interfaces.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_infotainment_distraction.py %}
```

## Case Study 2: Warning Overload and Attention Capture in Safety-Critical Alerts

Compare an as-built warning suite with retuned thresholds and with arbitration by warnings per hour, predictive value, compliance and conflicts braked for in time.

**Libraries:** NumPy, pandas, matplotlib

[Download cs2_warning_overload.py](cs2_warning_overload.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch03_distraction/cs2_warning_overload.py){: .btn .fs-3 }

![Warning Overload and Attention Capture in Safety-Critical Alerts]({{ '/docs/figures/fig03_07_cs2_alerts.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_warning_overload.py %}
```
