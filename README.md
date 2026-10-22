# Molecular Modelling OJ (course submission)

A classical molecular dynamics simulator for mixtures of small molecules
(water, ethanol, O₂, H₂), written as the final project of the course
*Introduction to Molecular Modeling and Simulation* (2MMN40) at TU Eindhoven,
supervised by Björn Baumeier (submitted February 2025).

## Authors

- **Olivier Meiler**: wrote the simulation code.
- **Justin Hen**: wrote most of the project report,
  *Simulating Water and Ethanol* ([report.pdf](report.pdf)).

## About this branch

This branch keeps the code as it was submitted for the course, and the report
describes this version. It is not maintained. The
[`main`](../../tree/main) branch has a reworked version by Olivier Meiler:
bug fixes, tests, an installable package, a much faster force calculation
and the missing features.

Run with `python main.py`, which asks for the mixing weights; the other
settings are in `main.py` and `config.py`.
