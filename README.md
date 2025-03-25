# Neural Transition-Based Dependency Parser

## Overview

This project implements a neural network-based dependency parser that incrementally builds dependency trees using transition-based parsing. The parser employs three primary transitions—**SHIFT**, **LEFT-ARC**, and **RIGHT-ARC**—to predict grammatical relationships between words. The final performance is measured by the Unlabeled Attachment Score (UAS).

## Environment Setup

A conda environment file `local_env.yml` is provided to simplify dependency installation. This environment is named **cs224n_a2** and includes all necessary packages.

### Steps to Create the Environment

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create the Conda Environment:**

   ```bash
   conda env create -f local_env.yml
   ```

3. **Activate the Environment:**

   ```bash
   conda activate cs224n_a2
   ```

---

## Project Structure

- **`parser_transitions.py`**  
  Implements the parsing mechanics including:
  - **Transition Functions:** `init` and `parse_step` for managing the state (stack, buffer, and dependency list).
  - **Minibatch Parsing:** An algorithm to parse multiple sentences simultaneously.

- **`parser_model.py`**  
  Contains the neural network model for predicting the next transition:
  - Custom implementations of linear and embedding layers (avoid using `torch.nn.Linear` or `torch.nn.Embedding`).
  - A forward pass that processes the feature vector derived from the current parser state.

- **`run.py`**  
  The main script for training and evaluating the parser:
  - Runs the training loop.
  - Computes performance using the UAS metric.
  - Optionally uses a debug mode (`python run.py -d`) to quickly iterate on a smaller dataset.

- **`utils/parser_utils.py`**  
  Utility functions for feature extraction (e.g., extracting the last word of the stack or first word of the buffer).

---

## Assignment Details

### Transition-Based Parsing

- **Transitions:**
  - **SHIFT:** Moves the first word from the buffer onto the stack.
  - **LEFT-ARC:** Attaches the second element of the stack as a dependent of the top element and removes it.
  - **RIGHT-ARC:** Attaches the top element of the stack as a dependent of the second element and removes the top element.

- **Goals:**
  - Correctly update the parser’s state after each transition.
  - Build a valid dependency tree for each sentence.

### Minibatch Parsing

- **Algorithm:**
  - Processes multiple sentences in parallel.
  - At every step, uses the neural network to predict the next transition for each sentence in a batch.
  - Continues until all sentences have been fully parsed (empty buffer with only ROOT in the stack).

### Neural Network Model

- **Architecture:**
  - **Input:** A concatenation of word embeddings for selected features from the parser state.
  - **Hidden Layer:** A fully connected layer with ReLU activation.
  - **Output Layer:** Generates logits for the three transitions, followed by a softmax to produce probabilities.

- **Training:**
  - The model is trained to minimize the cross-entropy loss between predicted and true transitions.
  - Uses the Adam optimizer with momentum and adaptive learning rates.
  - Performance is evaluated using the Unlabeled Attachment Score (UAS).

---

## Training and Evaluation

- **Training Procedure:**
  - Use the provided training data (e.g., from the Penn Treebank with Universal Dependencies).
  - Adjust hyperparameters using the development set.
  - The best UAS should be reported for both development and test sets in your submission.

- **Debugging Tips:**
  - Use the debug mode (`python run.py -d`) for faster iterations on a subset of data.
  - Ensure efficient embedding lookup and forward pass implementations to avoid performance bottlenecks.

---

## Deliverables

- **Transition Mechanics:** Fully implemented functions in `parser_transitions.py`.
- **Minibatch Parser:** A working minibatch parsing function.
- **Neural Dependency Parser:** A complete neural network model in `parser_model.py` and training routines in `run.py`.
- **Performance Report:** A report (PDF) detailing the best UAS on the dev and test sets along with any analysis.

---

## Resources

- **PyTorch Documentation:** [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- **Neural Dependency Parsing Paper:** Chen & Manning (2014) – [Link](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf)
- **Universal Dependencies:** [Universal Dependencies](http://universaldependencies.org)
- **TQDM Documentation:** [TQDM GitHub](https://github.com/tqdm/tqdm)

---

## Conclusion

This assignment focuses on implementing a neural transition-based dependency parser from scratch. Through this project, you will gain a deep understanding of transition-based parsing, feature extraction, and neural network training using PyTorch. With the provided environment and code structure, you can efficiently develop, test, and improve your parser. Happy parsing!
