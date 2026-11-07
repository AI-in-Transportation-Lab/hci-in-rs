---
title: "4. In-Vehicle Interface Design for Safety-Critical Use"
parent: Chapters
nav_order: 4
---

# Chapter 4: In-Vehicle Interface Design for Safety-Critical Use
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 4` from the repository root; outputs go to `outputs/ch04/`.

## Case Study 1: Glance Management and Usability Testing for In-Vehicle HMIs

Simulate glances for three destination-entry designs, check them against the NHTSA criteria and an occlusion test, and estimate how many participants a usability test needs.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_glance_usability.py](cs1_glance_usability.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch04_in_vehicle_interfaces/cs1_glance_usability.py){: .btn .fs-3 }

![Glance Management and Usability Testing for In-Vehicle HMIs]({{ '/docs/figures/chap04_fig01_glance_usability.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_glance_usability.py %}
```

## Case Study 2: Comparing Touch, Voice, Gesture, and Multimodal Interfaces for Driver Safety

Compare five selection modalities by task time and eyes-off-road time, size touch targets for road vibration, model redundant signals, and find the voice error rate at which touch wins.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs2_interaction_modalities.py](cs2_interaction_modalities.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch04_in_vehicle_interfaces/cs2_interaction_modalities.py){: .btn .fs-3 }

![Comparing Touch, Voice, Gesture, and Multimodal Interfaces for Driver Safety]({{ '/docs/figures/chap04_fig02_modalities.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_interaction_modalities.py %}
```
