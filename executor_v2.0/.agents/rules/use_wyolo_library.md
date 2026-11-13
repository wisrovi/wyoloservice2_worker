---
trigger: always_on
description: Mandatory rule to strictly use the official wyolo library without local code vendoring.
---

# Strict Rule: Official `wyolo` Library Usage

1. **Mandatory Use of Official `wyolo` Library:**
   - All YOLO and RT-DETR training, hyperparameter optimization, and post-training analysis MUST strictly import and use the official corporate `wyolo` library (`from wyolo import ...`, `from wyolo.trainer import ...`).
   - The worker container installs `wyolo` directly as a dependency from PyPI or corporate wheels.

2. **Absolute Prohibition of Local Code Vendoring:**
   - It is strictly forbidden to vendor or maintain local copies of `wyolo` source code (such as `lib/`, `_trainer/`, or local duplicate trainer scripts) inside the worker or executor codebase.
   - All shared logic, post-training states, XAI metrics, and MLflow integrations belong to and must be developed within the official library repository: `/home/william.rodriguez/Documents/w_libraries/w_libraries/wyolo`.

3. **Modifications and Enhancements:**
   - Any enhancements, bug fixes, or new states must be implemented and tested in the `wyolo` repository first, released/bumped, and consumed by the worker executor via its declared requirements.
