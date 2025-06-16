# Hierarchical Variational Memory (HVM) Implementation

Implementation based on the paper "Hierarchical Variational Memory for Few-shot Learning Across Domains" (ICLR 2022).



## File Structure

```
├── features.py                     # Feature extractors (including hierarchical versions)
├── hierarchical_memory.py          # Core HVM implementation
├── run_hvm_classifier.py          # HVM classifier main program
├── test_hvm.py                     # HVM functionality test script
├── inference.py                    # Inference networks
├── utilities.py                    # Utility functions
├── data.py                         # Data loading
└── README_HVM.md                   # This document
```



## Usage

### 1. Basic Testing

First run the test script to verify the implementation:

```bash
python test_hvm.py
```

### 2. Training HVM Model

Use HVM for few-shot classification training:

```bash
python run_hvm_classifier.py \
    --dataset miniImageNet \
    --mode train_test \
    --shot 1 \
    --way 5 \
    --num_levels 4 \
    --kl_weight 0.1 \
    --iterations 1000 \
    --learning_rate 0.0001
```

### 3. Parameter Description

#### HVM-specific Parameters:

- `--num_levels`: Number of hierarchical levels (default: 4)
- `--kl_weight`: Weight for KL divergence loss (default: 0.1)
- `--hierarchical`: Whether to use hierarchical memory (default: True)

#### General Parameters:

- `--dataset`: Dataset selection (Omniglot, miniImageNet, tieredImageNet, cifarfs)
- `--shot`: Number of support samples per class
- `--way`: Number of classes
- `--memory_samples`: Number of memory samples (default: 50)
- `--top_k`: Top-k value for attention mechanism (default: 10)


## Reference

```bibtex
@inproceedings{du2022hierarchical,
  title={Hierarchical Variational Memory for Few-shot Learning Across Domains},
  author={Du, Yingjun and Zhen, Xiantong and Shao, Ling and Snoek, Cees GM},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
``` 