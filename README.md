# Python Reproducibility Package

## Study

Bilateral Stable Matching for Low-Value Recyclable Collection under Multi-Type Fuzzy Information and Partially Known Criterion Weights

## Overview

This package provides a complete, executable implementation of the computational workflow of the paper, covering data definitions, multi-type fuzzy distance computation, satisfaction normalization, criterion-matrix construction, weight optimization, comprehensive-satisfaction aggregation, partial one-to-one matching enumeration and stability checks, and policy-weight robustness analysis.

## Repository and Code Availability

- Source code repository: https://github.com/qiao268/recyclable-matching
- The repository is publicly accessible and contains the source code, input template, tests and requirements described in the Package Contents section below.
- The Code availability section of the manuscript cites the same persistent link.

## Package Contents

- `reproduce_paper.py`: Table 2 data; binary, linguistic, interval, intuitionistic fuzzy and Pythagorean fuzzy distance computation; domain-normalized satisfactions; criterion matrices; maximum-deviation and neutral feasible weight optimizers; comprehensive-satisfaction aggregation; partial one-to-one matching enumeration; weak-stability and blocking-pair checks; policy-weight sensitivity; an alternative Hamming distance; and input-direction perturbation utilities.
- `weight_constraints_template.json`: input template for the incomplete weight restrictions of each participant.
- `outputs/`: generated criterion matrices, together with the comprehensive matrices and matching diagnostics reported in the manuscript.
- `tests/`: automated verification tests.
- `requirements.txt`: Python dependencies.
- `CODE_AVAILABILITY_STATEMENT.txt`: manuscript statement template.

## Requirements and Running

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python reproduce_paper.py
python -m unittest discover -s tests -v
```

## Scope of Reproduction and Validated Results

- All evaluation data reported in Table 2 of the manuscript are encoded in the source, and criterion-level scores are recomputed from them.
- Matching verification uses the comprehensive satisfaction matrices Y, B and P reported in the manuscript; the package checks 1,045 feasible partial one-to-one assignments.
- The unique stable matching is verified as `(A1,R5), (A2,R6), (A3,R4), (A4,R1)`.
- The unconstrained aggregate-score matching is verified as `(A1,R5), (A2,R2), (A3,R4), (A4,R1)`, with blocking pair `(A2,R6)`.
- All of the above are covered by the automated tests under `tests/`.

## Input Requirements

- To reproduce the weight-optimization results and the weight-dependent sensitivity results, populate `weight_constraints_template.json` before running with the feasible weight sets Omega_i and Lambda_j of the participants, the unrounded optimized weight vectors, and the numerical settings of each robustness scenario.
- These values form part of the authors' original computational record and are not distributed with this package; `reproduce_paper.py` implements the general maximum-deviation and neutral feasible optimization workflow and consumes the restrictions entered through the template.
