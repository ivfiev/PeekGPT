package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

type (
	scalar float32
	vector []scalar
	matrix [][]scalar
)

type Objective interface {
	Eval(vector) scalar
	Size() int
	Clone() Objective
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

func softmax(s, a matrix) {
	if len(a) != len(a[0]) || len(s) != len(s[0]) {
		log.Panicf("softmax: A & S must be square")
	}
	mulMatK(s, 0)
	for i := range a {
		triangle := i + 1 // len(a[0]) - i
		rowMax := scalar(math.Inf(-1))
		for j := range triangle {
			if a[i][j] > rowMax {
				rowMax = a[i][j]
			}
		}
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

func spsaSample(obj Objective, theta vector, bufs []vector, eps scalar, count int, rng *rand.Rand) {
	mulVec(bufs[0], 0)
	copy(bufs[2], theta)
	for range count {
		rademacher(bufs[1], rng)
		addVec(bufs[2], bufs[1], eps)
		plus := obj.Eval(bufs[2])
		addVec(bufs[2], bufs[1], -2*eps)
		minus := obj.Eval(bufs[2])
		d := (plus - minus) / (2 * eps)
		addVec(bufs[0], bufs[1], d)
	}
	mulVec(bufs[0], 1/scalar(count))
}

func spsa(obj Objective, theta vector, iters, samples, parallel int, lRate, eps scalar) {
	objs := make([]Objective, parallel)
	bufs := make([][]vector, parallel)
	rngs := make([]*rand.Rand, parallel)
	for i := range parallel {
		objs[i] = obj.Clone()
		bufs[i] = make([]vector, 4)
		for j := range len(bufs[i]) {
			bufs[i][j] = make(vector, obj.Size())
		}
		rngs[i] = rand.New(rand.NewSource(time.Now().UnixNano() + int64(i*1000)))
	}
	var wg sync.WaitGroup
	for range iters {
		for w := range parallel {
			wg.Go(func() {
				spsaSample(objs[w], theta, bufs[w], eps, samples, rngs[w])
			})
		}
		wg.Wait()
		for i := 1; i < parallel; i++ {
			addVec(bufs[0][0], bufs[i][0], 1)
		}
		addVec(theta, bufs[0][0], -lRate/scalar(parallel))
	}
}
