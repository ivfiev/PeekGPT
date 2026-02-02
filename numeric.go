package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type (
	vector = []float64
	matrix = *mat.Dense
)

type objective interface {
	eval(vector) float64
}

func printMat(A matrix) {
	r, c := A.Dims()
	for i := range r {
		fmt.Printf("[")
		for j := range c {
			fmt.Printf("%6.3f ", A.At(i, j))
		}
		fmt.Printf("]\n")
	}
}

func printRow(A matrix, r int) {
	data, _, c, s := unmat(A)
	printVec(data[r*s : r*s+c])
}

func printVec(v vector) {
	fmt.Printf("[")
	for i := range v {
		fmt.Printf("%6.3f ", v[i])
	}
	fmt.Printf("]\n")
}

func addVec(v, u vector, k float64) {
	if len(v) != len(u) {
		log.Panicf("addVec: vectors of invalid lengths, %d & %d\n", len(v), len(u))
	}
	for i := range v {
		v[i] += u[i] * k
	}
}

func makeMat(rows, cols int) matrix {
	return mat.NewDense(rows, cols, nil)
}

func rowMax(row []float64) (float64, int) {
	rowMax := math.Inf(-1)
	i := -1
	for j, v := range row {
		if v > rowMax {
			rowMax = v
			i = j
		}
	}
	return rowMax, i
}

func unmat(A matrix) ([]float64, int, int, int) {
	raw := A.RawMatrix()
	return raw.Data, raw.Rows, raw.Cols, raw.Stride
}

func softmax(S, A matrix) {
	dataA, rA, cA, sA := unmat(A)
	dataS, rS, cS, sS := unmat(S)
	if rS != rA || cS != cA {
		log.Panicf("Softmax: incompatible matrices, A: %dx%d, S: %dx%d\n", rA, cA, rS, cS)
	}
	S.Zero()
	for i := range rA {
		triangle := i + 1
		rowA := dataA[i*sA : i*sA+triangle]
		rowS := dataS[i*sS : i*sS+triangle]
		rowMax, _ := rowMax(rowA)
		if rowMax == 0 {
			continue
		}
		var sum float64
		for j := range triangle {
			f := math.Exp(rowA[j] - rowMax)
			rowS[j] = f
			sum += f
		}
		for j := range triangle {
			rowS[j] /= sum
		}
	}
}

func layerNorm(L, X matrix, gamma, beta vector) {
	dataX, rX, cX, sX := unmat(X)
	dataL, rL, cL, sL := unmat(L)
	rows := rX
	cols := cX
	if rX != rL || cX != cL || cX != len(gamma) || cols != len(beta) {
		log.Fatalf("LayerNorm: incompatible dimensions, L: %dx%d, X: %dx%d, gamma: %d, beta: %d\n",
			rL, rL, rX, cX, len(gamma), len(beta))
	}
	for i := range rows {
		u := 0.0
		o2 := 0.0
		for j := range cols {
			u += dataX[i*sX+j]
		}
		if u == 0 {
			continue
		}
		u /= float64(cols)
		for j := range cols {
			x := dataX[i*sX+j]
			o2 += (x - u) * (x - u)
		}
		o2 /= float64(cols)
		for j := range cols {
			dataL[i*sL+j] = (dataX[i*sX+j] - u) / math.Sqrt(o2+0.00001)
			dataL[i*sL+j] *= gamma[j]
			dataL[i*sL+j] += beta[j]
		}
	}
}

func ReLU(x float64) float64 {
	return max(0, x)
}

func rademacher(v vector, rng *rand.Rand) vector {
	for i := range len(v) {
		if rng.Float32() < 0.5 {
			v[i] = -1
		} else {
			v[i] = 1
		}
	}
	return v
}

func onehot(dim, ix int) vector {
	v := make(vector, dim)
	v[ix] = 1
	return v
}

func softSample(logits vector) int {
	rm, _ := rowMax(logits)
	var sum float64 = 0
	for i := range logits {
		sum += math.Exp(logits[i] - rm)
	}
	var running float64 = 0
	r := rand.Float64()
	for i := range logits {
		running += math.Exp(logits[i]-rm) / sum
		if r < running {
			return i
		}
	}
	log.Fatal("failed to softsample")
	return -1
}

func spsa(obj objective, theta vector, iters int, lr, eps float64, rng *rand.Rand) {
	binary := make(vector, len(theta))
	delta := make(vector, len(theta))
	for range iters {
		copy(delta, theta) // noisy!
		rademacher(binary, rng)
		addVec(delta, binary, eps)
		plus := obj.eval(delta)
		addVec(delta, binary, -2*eps)
		minus := obj.eval(delta)
		d := (plus - minus) / (2 * eps)
		addVec(theta, binary, -d*lr)
	}
}
