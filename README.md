# tiny-transformers
Minimal from-scratch implementation of a simple transformer model.
- Learnable token & position embeddings
- LayerNorm
- Single-head attention
- Residuals
- MLP
- Multiple blocks

Trains using [SPSA](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation).

[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) recommended, if possible a build with AVX512 support.

Included are some simple tasks for training, eg copy/reverse/index a string, sum integers, etc. 

## Example

`make run`

Reverse-a-string task, 50k parameters, dmodel 64, context size 9, 2 blocks.

### Heatmap
Shows the transformations to the input vector as it progresses through the model.

<img width="1238" height="300" alt="image" src="https://github.com/user-attachments/assets/0ae1636f-ac12-4b69-80af-88f0306fb7d4" />


### Attention matrices
<img width="247" height="437" alt="image" src="https://github.com/user-attachments/assets/e7732d6c-ec7c-44f4-9811-a5c1aa08027d" />

<br>
<br>
<br>
<br>

A larger example with ctx 21 - nice patterns in the lower matrix.

<img width="523" height="874" alt="image" src="https://github.com/user-attachments/assets/27f76643-c57d-4cab-9429-f73c930ee8f9" />


