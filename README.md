# PeekGPT
From-scratch implementation of a GPT2/3-style transformer model allowing to peek inside during inference/training.
- Runs entirely on CPU
- No network/API calls nor ML frameworks
- Pure Go, but OpenBLAS can be optionally linked in for faster matrix products

## Example

### Training
```
$ go run . -mode train -model models/names -data ./data/names -text -v 200 \
    -dmodel 32 -ctx 8 -blocks 2 -attn 2 -mlp 2 \
    -iters 1000 -lr 0.01 -ub 64
```
This trains a model generating random names with:
- 32-dimensional embedding vectors
- context size of 8 tokens
- 2 blocks
- 2 attention heads per-block
- ~19k parameters

Training configuration:
- location of training data `data/names`
- validation set size 200
- 1000 iterations @ learning rate 0.01 (Adam)
- batch size 64

Training above takes 2 seconds on my Zen5 CPU.

### Text generation
```
$ go run . -mode prompt -model ./models/names -text -prompt 'adam' -n 50
```
Will generate random names like
```
adam
allaunex
bandero
briestyn
nelun
kad
feren
dondlyn
```

### Vector flow
```
$ go run . -mode peek -model ./models/names -prompt 'adam'
```

### Attention matrices
```
$ go run . -mode peek -attention -model ./models/names -prompt 'briestyn'
```

## Overview
<img width="400" height="800" alt="image" src="https://github.com/user-attachments/assets/3d442d70-7092-49be-88e2-2db41df1e5cc" />
