# PROJECT.md

# Clinical PHI Protection for LLM Applications

## 1. Project Context

This project aims to develop an open-source, healthcare-focused PHI protection integration for **NVIDIA NeMo Guardrails**.

Rather than developing a new clinical de-identification model or competing with existing enterprise solutions, we aim to leverage existing PII detection tools and clinical NLP models to improve PHI protection in LLM applications.

The initial focus is protecting user inputs before they reach the LLM. Future extensions may cover retrieval, output and tool interactions.

## 2. Project Goals

The project will initially focus on:

- Benchmarking existing solutions: Presidio, GLiNER-PII and clinical transformer models.
- Developing a common entity schema and standardised character-span representation.
- Implementing entity reconciliation, including duplicate and overlapping entity handling.
- Evaluating a hybrid detection approach combining general-purpose PII detection and clinical NLP models.
- Integrating the selected approach into NeMo Guardrails through its custom actions and rail mechanisms.
- Measuring detection quality, masking effectiveness, false negatives and inference latency.

**Benchmarking comes first.** Implementation decisions must be supported by experimental results rather than assumptions.

## 3. Development Principles

- **Keep it simple:** Prefer straightforward, readable Python code over unnecessarily complex abstractions, class hierarchies or design patterns.
- **Modular development:** Separate detectors, entity processing, reconciliation, masking, evaluation and NeMo integration into logical components.
- **Avoid overengineering:** Implement only what is necessary for the current project scope.
- **Reuse existing solutions:** Prefer established libraries, models and NeMo capabilities over reinventing existing functionality.
- **Maintain consistency:** Follow the existing project structure, coding conventions and dependencies.
- **Test incrementally:** Validate individual components before integrating them into the complete pipeline.

## 4. Project Constraints

- Do not introduce major architectural changes, new dependencies or additional frameworks without approval.
- Do not create unnecessary files, classes, abstractions or configuration layers.
- Do not modify existing components or interfaces unnecessarily.
- Do not implement speculative features outside the agreed scope.
- Do not claim enterprise readiness, regulatory compliance or guaranteed PHI protection based solely on benchmark results.
- Use authorised datasets and respect their licensing, access and redistribution restrictions.

## 5. Agent Behaviour and Decision-Making

The following rules are mandatory:

1. **Follow this document:** Always consider PROJECT.md and the existing project structure before implementing changes.
2. **No autonomous architectural decisions:** Do not make major structural, architectural or dependency changes without explicit approval.
3. **Pause when blocked:** If an issue cannot be resolved through a straightforward implementation, stop and explain the problem.
4. **Propose before changing:** When multiple technical approaches are possible, briefly explain the alternatives, trade-offs and your recommendation. Wait for approval before proceeding if the decision affects architecture or project direction.
5. **No unnecessary complexity:** Do not attempt complicated workarounds merely to force an implementation to succeed.
6. **Explain important decisions:** Briefly communicate significant implementation choices and their technical reasoning.
7. **Preserve working functionality:** Avoid unrelated refactoring and changes to previously validated components.

## 6. Working Philosophy

Develop incrementally, prioritising correctness, reproducibility, readability and maintainability.

The goal is not to build the largest or most sophisticated system possible. It is to build a technically sound, useful and demonstrable open-source contribution through controlled, evidence-driven development.