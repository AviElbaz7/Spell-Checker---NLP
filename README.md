# Spell Checker - NLP Assignment 1

This project implements a **context-sensitive spell checker** using a **noisy channel model** combined with a **Markov language model**. The system corrects both **non-word** and **real-word** spelling errors within the context of a sentence.

---

## 📚 Assignment Summary

- **Goal:** Implement a probabilistic spell checker using a language model and confusion matrices.
- **Techniques:**
  - N-gram Language Model (Markov-based)
  - Noisy Channel Model with edit types
  - Context-aware word correction
  - Laplace smoothing
- **Corpus Used:** [`big.txt`](https://norvig.com/big.txt)

---

## 🛠 Project Structure

- `ex1.py` — Main implementation file, including:
  - `Spell_Checker` class
  - `Language_Model` inner class
  - Utility functions: `normalize_text()`, `who_am_i()`

- `spelling_confusion_matrices.py` — Provided file with confusion matrices (errors).

- `test_spell_checker.py` — Example usage script for testing the spell checker.

---

## ✅ Features Implemented

- Context-sensitive correction using n-gram model
- Support for character-level edit operations:
  - Insertion
  - Deletion
  - Substitution
  - Transposition
- Evaluation of candidate corrections based on:
  - Edit probability
  - Sentence likelihood
  - Unigram frequency bonus
- Efficient candidate generation via edit-distance
- Compliant with project constraints (e.g. allowed imports only)

---

## 🚀 How to Run

1. Make sure you have `nltk` installed:
   ```bash
   pip install nltk

2. Place big.txt and spelling_confusion_matrices.py in the project directory.

3. Run the test script:
   python3 test_spell_checker.py

## 🧪 Example
Input:
I enoy learnig new things

Output:
i enjoy learning new things
