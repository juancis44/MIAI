# Vision

## Mission

MIAI aims to become an open-source ecosystem that simplifies the development
of reproducible, modular, and clinically-oriented medical imaging workflows.

Rather than replacing existing libraries such as MONAI, PyTorch, or
SimpleITK, MIAI integrates them into a coherent software architecture focused
on research reproducibility, software engineering best practices, and
clinical applicability.

## The problem

Modern medical imaging research requires combining many disconnected tools:
DICOM management, NIfTI conversion, image preprocessing, dataset
organization, deep learning frameworks, evaluation, visualization, and
experiment management. Excellent libraries exist individually, but there is
no unified framework that connects the complete research workflow. MIAI fills
this gap.

## Philosophy

MIAI is not another deep learning library. MIAI is a **workflow framework**.
Its goal is to standardize everything surrounding deep learning — deep
learning becomes only one component of the pipeline.

## Target users

- Medical imaging researchers
- Clinical AI researchers
- Research engineers
- Radiotherapy researchers
- PhD students
- Hospitals
- Medical physics teams
- Healthcare AI startups

## Design principles

1. **Reproducibility first.** Every experiment should be reproducible.
   Configurations define experiments instead of modifying Python code.
2. **Modular architecture.** Every component is independent. Users can
   replace modules without affecting the rest of the pipeline.
3. **Clinical-oriented.** Designed around real hospital workflows, not toy
   datasets — real imaging pipelines.
4. **Open source.** MIT license, community contributions, transparent
   development.
5. **Simplicity.** Readable code, good documentation, few dependencies,
   consistent APIs.

## Scope

MIAI focuses on: MRI, CT, PET, ultrasound, radiotherapy imaging, histology,
and microscopy — across segmentation, registration, reconstruction, image
enhancement, artifact reduction, image synthesis, generative AI, dataset
management, clinical imaging pipelines, evaluation, and visualization.

## Long-term goal

Become a community-driven ecosystem for reproducible medical imaging AI
research, combining software engineering best practices with real clinical
workflows.
