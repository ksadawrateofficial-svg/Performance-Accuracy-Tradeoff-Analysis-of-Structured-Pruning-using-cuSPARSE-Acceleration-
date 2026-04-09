# Performance-Accuracy Tradeoff Analysis of Structured Pruning using cuSPARSE Acceleration

## About this Project

This project explores how we can **make deep learning models faster without losing too much accuracy**.

The idea is simple:

> Remove unnecessary weights from a trained model (pruning), and then use GPU acceleration to make computations faster.

We use a **VGG-16 model trained on CIFAR-10**, apply different structured pruning techniques, and analyze how performance and accuracy change.

---

## What I Implemented

* **Vector Pruning** → removes small weights within segments (keeps accuracy)
* **Block Pruning** → removes entire blocks (improves speed)
* **Mixed Pruning** → combines both for best tradeoff
* Conversion to **CSR format** for efficient sparse computation
* GPU acceleration using **NVIDIA cuSPARSE**
* Performance vs Accuracy analysis with graphs

---

## How it Works

1. Train a **VGG-16 model** on CIFAR-10
2. Extract weight matrices from the model
3. Apply pruning:

   * Vector-wise
   * Block-wise
   * Mixed (paper-based approach)
4. Convert pruned matrix into **sparse format (CSR)**
5. Run computation on:

   * CPU (dense)
   * GPU (cuSPARSE sparse)
6. Compare results (speed vs accuracy)

---

## Final Results

| Metric   | Value                |
| -------- | -------------------- |
| Sparsity | ~71%                 |
| CPU Time | 0.354 ms             |
| GPU Time | 0.030 ms             |
| Speedup  | **11.82× faster** |

This shows that we can **significantly speed up computation** while maintaining reasonable accuracy.

---

## Tradeoff Visualization

This graph shows how **speed improves as accuracy decreases**, and helps find the best balance.

![Tradeoff](results/tradeoff.png)

---

## Tech Stack

* Python (PyTorch)
* CUDA C
* cuSPARSE
* NumPy
* Matplotlib

---

## How to Run

### 1. Train the model

```bash
python train_vgg.py
```

### 2. Apply pruning

```bash
python pruning.py
```

### 3. Run CUDA acceleration

```bash
nvcc main.cu -lcusparse -o run
./run
```

---

## Project Structure

```
├── main.cu              # CUDA + cuSPARSE implementation
├── pruning.py          # Pruning algorithms
├── train_vgg.py        # Model training
├── results/
│   └── tradeoff.png    # Graphs
├── paper/
│   └── ieee_paper.pdf  # Report
```

---

## Key Takeaway

* More sparsity → faster computation
* Less sparsity → better accuracy
* **Best balance is achieved using mixed pruning (p ≈ 0.7–0.9)**

---

## Authors

- **Kedar Sadawrate**  
  M.Tech Embedded Systems, VIT Vellore  

- **Tejankur Tatkare**  
  M.Tech Embedded Systems, VIT Vellore  
---

##  Note

This project is built for **learning + research purposes**, and follows ideas from structured pruning papers.

---

If you found this useful, feel free to star the repo!
