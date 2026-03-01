# tiny-transformers
Minimal from-scratch implementation of a simple transformer model.
- Learnable token & position embeddings
- LayerNorm
- Multi-head attention
- Residuals
- MLP
- Multiple blocks

No API calls, no external dependencies other than [gonum](https://github.com/gonum/gonum) matrix library.

[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) recommended, if possible a build with AVX512 support.

Included are some simple tasks for training, eg copy/reverse/index a string, sum integers, etc. 

## Example

#### Execute unit tests:
`go test`

#### Generate some training data. Available tasks - `reverse`, `copy`, `kv`, `index`.
`go run . -mode gen -task reverse -n 10000 -max 10 -vocab 1234567890 > data/reverse10`

#### Training. For above tasks loss of 0 is easily achievable.
`go run . -mode train -data ./data/reverse10 -model ./models/reverse10 -t 9500 -v 500 -dmodel 48 -ctx 21 -dattn 24 -attn 2 -blocks 2 -lr 0.001 -iters 500`

#### Inference.
`go run . -mode load -model ./models/reverse10 -prompt '123456789|?????????'`


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


