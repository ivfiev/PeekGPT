package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

type (
	scalar float32
	vector []scalar
	matrix [][]scalar
)

type objective interface {
	eval(vector) scalar
}

func makeMat(n, m int) matrix {
	rows := make([][]scalar, n)
	for i := range n {
		rows[i] = make([]scalar, m)
	}
	return rows
}

func mulMat(c, a, b matrix) {
	if len(a[0]) != len(b) || len(c) != len(a) || len(c[0]) != len(b[0]) {
		log.Panicf("mulMat: bad matrix dimensions, A: %dx%d, B: %dx%d, C: %dx%d\n", len(a), len(a[0]), len(b), len(b[0]), len(c), len(c[0]))
	}
	for i := range a {
		for j := range b[0] {
			sum := scalar(0.0)
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
			sum := scalar(0.0)
			for k := range b[0] {
				sum += a[i][k] * b[j][k]
			}
			c[i][j] = sum
		}
	}
}

func mulMatK(a matrix, k scalar) {
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

func printMat(a matrix) {
	for _, row := range a {
		fmt.Printf("[ ")
		for _, Anm := range row {
			fmt.Printf("%.3f ", Anm)
		}
		fmt.Printf("]\n")
	}
}

func printVec(v vector) {
	fmt.Printf("[")
	for _, x := range v {
		fmt.Printf("%.3f ", x)
	}
	fmt.Printf("]\n")
}

func mulVec(v vector, k scalar) {
	for i := range len(v) {
		v[i] *= k
	}
}

func addVec(v, u vector, k scalar) {
	if len(v) != len(u) {
		log.Panicf("addVec: vectors of invalid lengths, %d & %d\n", len(v), len(u))
	}
	for i := range v {
		v[i] += u[i] * k
	}
}

func rowMax(row vector) (scalar, int) {
	rowMax := scalar(math.Inf(-1))
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
		triangle := i + 1 // len(a[0]) - i
		rowMax, _ := rowMax(a[i][:triangle])
		var sum scalar
		for j := range triangle {
			f := scalar(math.Exp(float64(a[i][j] - rowMax)))
			s[i][j] = f
			sum += f
		}
		for j := range triangle {
			s[i][j] /= sum
		}
	}
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
		sum += math.Exp(float64(logits[i] - rm))
	}
	var running float64 = 0
	r := rand.Float64()
	for i := range logits {
		running += math.Exp(float64(logits[i]-rm)) / sum
		if r < running {
			return i
		}
	}
	log.Fatal("failed to softsample")
	return -1
}

func spsa(obj objective, theta vector, iters int, lr, eps scalar, seed int64) {
	binary := make(vector, len(theta))
	tmp := make(vector, len(theta))
	rng := rand.New(rand.NewSource(seed))
	for i := range iters {
		if i%250 == 0 {
			current := obj.eval(theta)
			fmt.Printf("\r%%%d %.4f %.4f ",
				int(float32(i)/float32(iters)*100),
				current,
				math.Exp(-float64(current)))
		}
		copy(tmp, theta) // noisy!
		rademacher(binary, rng)
		addVec(tmp, binary, eps)
		plus := obj.eval(tmp)
		addVec(tmp, binary, -2*eps)
		minus := obj.eval(tmp)
		d := (plus - minus) / (2 * eps)
		addVec(theta, binary, -d*lr)
	}
	fmt.Printf("\r                    \r")
}
