package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

type (
	vector []float64
	matrix [][]float64
)

type objective interface {
	eval(vector) float64
}

func makeMat(n, m int) matrix {
	rows := make([][]float64, n)
	for i := range n {
		rows[i] = make([]float64, m)
	}
	return rows
}

func mulMat(c, a, b matrix) {
	if len(a[0]) != len(b) || len(c) != len(a) || len(c[0]) != len(b[0]) {
		log.Panicf("mulMat: bad matrix dimensions, A: %dx%d, B: %dx%d, C: %dx%d\n", len(a), len(a[0]), len(b), len(b[0]), len(c), len(c[0]))
	}
	for i := range a {
		for j := range b[0] {
			sum := 0.0
			for k := range b {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
}

func mulMatT(c, a, b matrix) {
	if len(a[0]) != len(b[0]) || len(c[0]) != len(b) || len(c) != len(a) {
		log.Panicf("mulMatT: bad matrix dimensions, A: %dx%d, B: %dx%d, C: %dx%d\n", len(a), len(a[0]), len(b), len(b[0]), len(c), len(c[0]))
	}
	for i := range a {
		for j := range b {
			sum := 0.0
			for k := range b[0] {
				sum += a[i][k] * b[j][k]
			}
			c[i][j] = sum
		}
	}
}

func mulMatK(a matrix, k float64) {
	for i := range a {
		for j := range a[0] {
			a[i][j] *= k
		}
	}
}

func addMatV(a matrix, v vector) {
	if len(a[0]) != len(v) {
		log.Panicf("addMatV: bad dimensions, %d + %d\n", len(a[0]), len(v))
	}
	for i := range len(a) {
		for j := range len(v) {
			a[i][j] += v[j]
		}
	}
}

func addMatM(c, a, b matrix) {
	if len(a) != len(b) || len(a[0]) != len(b[0]) || len(c) != len(a) || len(c[0]) != len(a[0]) {
		log.Panicf("addMatM: bad dimensions, A: %dx%d, B: %dx%d, C: %dx%d\n", len(a), len(a[0]), len(b), len(b[0]), len(c), len(c[0]))
	}
	for i := range len(a) {
		for j := range len(a[0]) {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
}

func mapMat(c, a matrix, f func(float64) float64) {
	if len(a) != len(c) || len(a[0]) != len(c[0]) {
		log.Panicf("mapMat: bad dimensions, A: %dx%d, C: %dx%d\n", len(a), len(a[0]), len(c), len(c[0]))
	}
	if f == nil {
		log.Panic("mapMat: nil f")
	}
	for i := range len(a) {
		for j := range len(a[0]) {
			c[i][j] = f(a[i][j])
		}
	}
}

func printMat(a matrix) {
	for _, row := range a {
		fmt.Printf("[")
		for _, Anm := range row {
			fmt.Printf("%6.3f ", Anm)
		}
		fmt.Printf("]\n")
	}
}

func printVec(v vector) {
	fmt.Printf("[")
	for _, x := range v {
		fmt.Printf("%6.3f ", x)
	}
	fmt.Printf("]\n")
}

func mulVec(v vector, k float64) {
	for i := range len(v) {
		v[i] *= k
	}
}

func addVec(v, u vector, k float64) {
	if len(v) != len(u) {
		log.Panicf("addVec: vectors of invalid lengths, %d & %d\n", len(v), len(u))
	}
	for i := range v {
		v[i] += u[i] * k
	}
}

func rowMax(row vector) (float64, int) {
	rowMax := math.Inf(-1)
	i := -1
	for j := range row {
		if row[j] > rowMax {
			rowMax = row[j]
			i = j
		}
	}
	return rowMax, i
}

func softmax(s, a matrix) {
	if len(a) != len(a[0]) || len(s) != len(s[0]) {
		log.Panicf("softmax: A & S must be square")
	}
	mulMatK(s, 0)
	for i := range a {
		triangle := i + 1
		rowMax, _ := rowMax(a[i][:triangle])
		if rowMax == 0 {
			continue
		}
		var sum float64
		for j := range triangle {
			f := math.Exp(a[i][j] - rowMax)
			s[i][j] = f
			sum += f
		}
		for j := range triangle {
			s[i][j] /= sum
		}
	}
}

func layerNorm(lnXs, xs matrix, gamma, beta vector) {
	if len(xs) != len(lnXs) || len(xs[0]) != len(lnXs[0]) || len(gamma) != len(xs[0]) || len(beta) != len(xs[0]) {
		log.Fatalf("LayerNorm: incompatible dimensions, lnXs: %dx%d, xs: %dx%d, gamma: %d, beta: %d\n",
			len(lnXs), len(lnXs[0]), len(xs), len(xs[0]), len(gamma), len(beta))
	}
	for i := range xs {
		u := 0.0
		o2 := 0.0
		for _, x := range xs[i] {
			u += x
		}
		if u == 0 {
			continue
		}
		u /= float64(len(xs[i]))
		for _, x := range xs[i] {
			o2 += (x - u) * (x - u)
		}
		o2 /= float64(len(xs[i]))
		for j := range xs[i] {
			lnXs[i][j] = (xs[i][j] - u) / math.Sqrt(o2+0.00001)
			lnXs[i][j] *= gamma[j]
			lnXs[i][j] += beta[j]
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
