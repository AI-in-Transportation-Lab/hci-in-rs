---
title: "8. Vulnerable Road Users and Inclusive Interface Design"
parent: Chapters
nav_order: 8
---

# Chapter 8: Vulnerable Road Users and Inclusive Interface Design
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 8` from the repository root; outputs go to `outputs/ch08/`.

## Case Study 1: V2P Safety Alerts for Pedestrians and Cyclists

Choose a V2P alert threshold, show how protection depends on positioning error, latency and false alerts, and compare visual, auditory and haptic alerts by user group.

**Libraries:** NumPy, pandas, matplotlib

[Download cs1_v2p_alerts.py](cs1_v2p_alerts.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch08_vulnerable_road_users/cs1_v2p_alerts.py){: .btn .fs-3 }

![V2P Safety Alerts for Pedestrians and Cyclists]({{ '/docs/figures/chap08_fig01_v2p_alerts.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_v2p_alerts.py %}
```

## Case Study 2: Inclusive Interface Design for Accessibility, Language, and Equity

Test pictogram comprehension with Wilson intervals, compute the reach of each message bundle for every user group, and find the bundles that lift the worst-off group.

**Libraries:** NumPy, pandas, matplotlib

[Download cs2_inclusive_design.py](cs2_inclusive_design.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch08_vulnerable_road_users/cs2_inclusive_design.py){: .btn .fs-3 }

![Inclusive Interface Design for Accessibility, Language, and Equity]({{ '/docs/figures/chap08_fig02_inclusive_design.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_inclusive_design.py %}
```
