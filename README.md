# PeekGPT
From-scratch implementation of a GPT-style transformer allowing to peek inside during inference/training.
- Runs entirely on CPU
- No network/API calls nor ML frameworks
- Pure Go, OpenBLAS can be optionally linked in for faster matrix products

## Example

### Training
```
$ go run . -mode train -model models/names -data ./data/names -text -v 200 \
    -dmodel 32 -ctx 8 -blocks 2 -attn 2 -mlp 2 \
    -iters 1000 -lr 0.01 -ub 64
```
This trains a character-level transformer to generate names:

Model:
- 32-dimensional embedding vectors
- context size of 8 tokens
- 2 blocks
- 2 attention heads per-block
- ~19k parameters

Training:
- location of training data `data/names`
- validation set size 200
- 1000 iterations (Adam, learning rate 0.01)
- batch size 64

Training above takes 2 seconds on my Zen5 CPU.

### Text generation
```
$ go run . -mode prompt -model ./models/names -text -prompt 'adam' -n 50
```
Sample output:
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

### Inspecting the model
#### Peek into how the model processes a prompt:
```
$ go run . -mode peek -model ./models/names -prompt 'adam'
```
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/bafacbe3-312d-4e05-aa40-e1301ef82737" />

#### Inspect attention matrices:
```
$ go run . -mode peek -attention -model ./models/names -prompt 'briestyn'
```
<img width="550" height="475" alt="image" src="https://github.com/user-attachments/assets/28459b54-e3f7-4b41-aee0-7259ca48370d" />

#### Run unit tests
```
$ go test
```

## Overview
<img width="400" height="800" alt="image" src="https://github.com/user-attachments/assets/3d442d70-7092-49be-88e2-2db41df1e5cc" />
