# PeekGPT
Minimal from-scratch implementation of a transformer with focus on interpretability.

No network/API calls, no external dependencies other than [gonum](https://github.com/gonum/gonum) matrix library.

[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) can be linked in (very recommended for 1x-10x speedups), if possible a build with AVX512 support.

## Overview
<img width="600" height="1200" alt="image" src="https://github.com/user-attachments/assets/3d442d70-7092-49be-88e2-2db41df1e5cc" />

<br/><br/>

## Example
Below is an example for an 18k parameter model trained to reverse strings up to 10 chars in length.

Trace shows path of the last 32-dimensional vector. Blue indicates negative values, red - positive. Black is near 0.

<img width="2180" height="1876" alt="image" src="https://github.com/user-attachments/assets/3b01e23b-364c-4743-a80b-907a64d7b79f" />

<br/><br/>

#### Execute unit tests:
`go test`

#### Generate some training data. Available tasks - `reverse`, `copy`, `kv`, `index`.
`go run . -mode gen -task reverse -n 10000 -max 10 -vocab 1234567890 > data/reverse10`

#### Training. For above tasks loss of 0 is easily achievable.
`go run . -mode train -data ./data/reverse10 -model ./models/reverse10 -t 9500 -v 500 -dmodel 48 -ctx 21 -dattn 24 -attn 2 -blocks 2 -lr 0.001 -iters 500`

#### Inference.
`go run . -mode load -model ./models/reverse10 -prompt '123456789|?????????'`


