# Kinetic Monte Carlo

Notebook for CHME 5137, Lecture 10 (2026): rabbits and foxes as individual animals,
simulated by kinetic Monte Carlo.

| Notebook | What it is |
|---|---|
| `rabbits-and-foxes-kmc.ipynb` | The exercise. Started in class, finished for homework |

The notebook has a paired `.py` file holding the same cells as plain text (see
**Jupytext** below).

## Getting started

1. **Fork** this repository to your own GitHub account (button at top right).
2. **Clone your fork**, not this one:
   ```bash
   git clone git@github.com:YOUR-USERNAME/kinetic-monte-carlo.git   # SSH, e.g. on Explorer
   # or
   git clone https://github.com/YOUR-USERNAME/kinetic-monte-carlo.git
   ```
   On Explorer, clone it into `/courses/CHME5137.202710/students/$USER/`.
3. Open `rabbits-and-foxes-kmc.ipynb` and work through it, using your own kernel
   (`chme5137` on Explorer). Commit and push as you go:
   ```bash
   git add rabbits-and-foxes-kmc.ipynb rabbits-and-foxes-kmc.py
   git commit -m "Task 1: one KMC trajectory, plotted against the ODE"
   git push
   ```
   This is your own fork, so committing on `main` is fine. Don't open a pull request
   back to `CHME5137/kinetic-monte-carlo`: that would publish your answers to everyone.

## Jupytext

`jupytext.toml` pairs the notebook with a `.py` script in the *percent* format, as in
the `differential-equations` repository. Saving the notebook updates the `.py` as well,
and you can run the `.py` as a script, or as a batch job on Explorer (Lecture 9).
If the two get out of step, `jupytext --sync rabbits-and-foxes-kmc.ipynb` brings them
back in line.

---

*Written with Claude Code (Claude Opus 5.5) and checked by Prof. West.*
