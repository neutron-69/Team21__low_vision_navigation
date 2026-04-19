# Prior Work Basis

This section summarizes how five key research papers informed the design, evaluation, and accessibility focus of our low-vision navigation assistant.

## 1. Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals [Link to Paper](https://aclanthology.org/2025.mmloso-1.8.pdf)

This paper is one of the strongest direct influences on our project because it focuses on blind and low-vision accessibility with lightweight VLMs.

### Influence on our work

- Demonstrated that smaller VLMs can still produce useful scene descriptions while remaining practical for resource-constrained deployment.
- Motivated prompt design strategies (prompt-only, prompt + context, and prompt + AD-style guidance) that are relevant to improving narration quality in our pipeline.
- Introduced accessibility-relevant evaluation dimensions such as descriptiveness, clarity, objectivity, spatial orientation, and action understanding.
- Supported our keyframe-based video processing trade-off: more frame coverage can improve temporal understanding, but increases latency and compute.

### Project relevance

This paper directly shaped our evaluation philosophy: we prioritize actionable and spatially grounded guidance quality over generic caption-only NLP scores.

## 2. End-to-End Navigation with Vision-Language Models: Transforming Spatial Reasoning into Question-Answering (VLMnav) [Link to Paper](https://arxiv.org/abs/2411.05755)

This paper is closely aligned with our objective of converting scene understanding into navigation decisions.

### Influence on our work

- Reinforced the use of depth-informed navigation to estimate safe movement directions.
- Encouraged framing navigation into discrete actionable choices (for example: left, right, forward).
- Inspired the use of visual direction cues to improve reasoning quality for navigation decisions.
- Supported obstacle-aware action proposal logic for selecting safer paths.
- Showed that RGB plus lightweight depth can already provide useful navigation behavior without a heavy robotics stack.

### Project relevance

Although our implementation remains modular (detection + depth + planning), the decision framing is similar: convert spatial perception into practical movement guidance.

## 3. VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding [Link to Paper](https://arxiv.org/abs/2410.13860)

This paper strongly influenced our multi-frame and grounding-oriented reasoning design.

### Influence on our work

- Motivated multi-frame aggregation to improve scene coverage beyond single-frame reasoning.
- Highlighted dynamic stitching/selection strategies that preserve coverage while reducing latency.
- Inspired robustness ideas from feedback/retry style grounding when predictions are uncertain.
- Emphasized benefits of multi-view cues for better localization of important objects.
- Showed that combining depth and multi-frame context can improve grounding quality without full 3D reconstruction.

### Project relevance

It supports our temporal reasoning choices and our focus on video-specific stability and hazard-priority behavior.

## 4. VizWiz: Nearly Real-Time Answers to Visual Questions [Link to Paper](https://dl.acm.org/doi/10.1145/1866029.1866080)

VizWiz is foundational for assistive vision systems targeting blind users in real-world conditions.

### Influence on our work

- Emphasized that assistive feedback must be immediate, contextual, and useful in practice.
- Reinforced a question-driven utility perspective (for example: what is ahead, where is risk, what should I do now).
- Highlighted real deployment challenges such as blur, low light, and poor framing.
- Strengthened our low-latency design priority and spoken feedback orientation.

### Project relevance

This work reinforced our focus on real-time, user-actionable guidance rather than raw detection output.

## Overall Impact on Our System

Collectively, these papers shaped our project in four ways:

- Accessibility-first objective: prioritize blind-user utility and actionability.
- Navigation-centric reasoning: convert perception outputs into concrete movement guidance.
- Temporal grounding: use multi-frame context for stability and better hazard prioritization.
- Evaluation design: include human-centered and navigation-specific metrics beyond generic NLP scores.
