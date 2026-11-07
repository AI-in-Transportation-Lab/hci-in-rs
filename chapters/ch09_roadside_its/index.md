---
title: "9. Roadside Interfaces, Connected Infrastructure, and ITS"
parent: Chapters
nav_order: 9
---

# Chapter 9: Roadside Interfaces, Connected Infrastructure, and ITS
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 9` from the repository root; outputs go to `outputs/ch09/`.

## Case Study 1: Dynamic Message Signs and Work Zone Messaging for Driver Comprehension

Check which of four message-sign texts drivers can read at speed, and estimate how a queue warning changes stopping distance and the share of drivers who cannot stop.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_dms_work_zone.py](cs1_dms_work_zone.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch09_roadside_its/cs1_dms_work_zone.py){: .btn .fs-3 }

![Dynamic Message Signs and Work Zone Messaging for Driver Comprehension]({{ '/docs/figures/chap09_fig01_dms_queue.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_dms_work_zone.py %}
```

## Case Study 2: AI-Assisted Traffic Management Center Visualization for Safety Decisions

Simulate AI video alerts verified by two TMC operators, compare alert thresholds and triage orders, and find the setting that minimizes secondary crashes.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs2_tmc_triage.py](cs2_tmc_triage.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch09_roadside_its/cs2_tmc_triage.py){: .btn .fs-3 }

![AI-Assisted Traffic Management Center Visualization for Safety Decisions]({{ '/docs/figures/chap09_fig02_tmc_triage.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_tmc_triage.py %}
```
