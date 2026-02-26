package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type (
	vector = []float64
	matrix = *mat.Dense
)

type objective interface {
	eval2([]vector, []vector, int) (vector, vector)
}

func makeMat(r, c int) matrix {
	return mat.NewDense(r, c, nil)
}

func unmat(A matrix) ([]float64, int, int, int) {
	raw := A.RawMatrix()
	return raw.Data, raw.Rows, raw.Cols, raw.Stride
}

func mulMat(C, A, B matrix) {
	C.Mul(A, B)
}

func mulMatT(C, A, B matrix) {
	C.Mul(A, B.T())
}

func mulTmat(C, A, B matrix) {
	C.Mul(A.T(), B)
}

func mulMatK(a matrix, k float64) {
	a.Scale(k, a)
}

func addMatV(A matrix, v vector) {
	d, r, c, s := unmat(A)
	if c != len(v) {
		log.Panicf("addMatV: bad dimensions, %d + %d\n", r, len(v))
	}
	for i := range r {
		for j := range c {
			d[i*s+j] += v[j]
		}
	}
}

func addMatM(C, A, B matrix) {
	C.Add(A, B)
}

func mapMat(C, A matrix, f func(float64) float64) {
	C.Apply(func(i, j int, v float64) float64 {
		return f(v)
	}, A)
}

func catMat(B matrix, As []matrix) {
	db, rb, cb, sb := unmat(B)
	_, ra, ca, _ := unmat(As[0])
	if ra != rb || ca*len(As) != cb {
		log.Panicf("catMat: bad dims %dx%d %dx%d\n", rb, cb, ra, ca)
	}
	for _, A := range As {
		_, r, c, _ := unmat(A)
		if ra != r || ca != c {
			log.Panicf("catMat: bad dims %dx%d %dx%d\n", ra, ca, r, c)
		}
	}
	for c, A := range As {
		da, _, ca, sa := unmat(A)
		for r := range rb {
			copy(db[r*sb+c*ca:r*sb+(1+c)*ca], da[r*sa:r*sa+sa])
		}
	}
}

func printMat(A matrix) {
	d, r, c, s := unmat(A)
	for i := range r {
		fmt.Printf("[")
		for j := range c {
			fmt.Printf("%9.6f ", d[i*s+j])
		}
		fmt.Printf("]\n")
	}
}

func printVec(v vector) {
	fmt.Printf("[")
	for _, x := range v {
		fmt.Printf("%9.6f ", x)
	}
	fmt.Printf("]\n")
}

func printRow(A matrix, i int) {
	d, _, c, s := unmat(A)
	printVec(d[i*s : i*s+c])
}

func mulVec(v vector, k float64) {
	for i := range len(v) {
		v[i] *= k
	}
}

func addVec2(v, u vector, k float64) {
	if len(v) != len(u) {
		log.Panicf("addVec: vectors of invalid lengths, %d & %d\n", len(v), len(u))
	}
	for i := range v {
		v[i] += u[i] * k
	}
}

func addVec3(w, v, u vector, k float64) {
	if len(v) != len(u) || len(w) != len(u) {
		log.Panicf("addVec: vectors of invalid lengths, %d = %d + %d\n", len(w), len(v), len(u))
	}
	for i := range v {
		w[i] = v[i] + u[i]*k
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

func rowSum(row vector, rowMax float64) float64 {
	sum := 0.0
	for _, x := range row {
		sum += math.Exp(x - rowMax)
	}
	return sum
}

func softmaxT(S, A matrix) {
	dA, rA, cA, sA := unmat(A)
	dS, rS, cS, sS := unmat(S)
	if rS != rA || cS != cA {
		log.Panicf("Softmax: incompatible matrices, A: %dx%d, S: %dx%d\n", rA, cA, rS, cS)
	}
	S.Zero()
	for i := range rA {
		triangle := i + 1
		rowA := dA[i*sA : i*sA+triangle]
		rowS := dS[i*sS : i*sS+triangle]
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
	L.Zero()
	dX, rX, cX, sX := unmat(X)
	dL, rL, cL, sL := unmat(L)
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
			u += dX[i*sX+j]
		}
		if u == 0 {
			continue
		}
		u /= float64(cols)
		for j := range cols {
			x := dX[i*sX+j]
			o2 += (x - u) * (x - u)
		}
		o2 /= float64(cols)
		for j := range cols {
			dL[i*sL+j] = (dX[i*sX+j] - u) / math.Sqrt(o2+0.00001)
			dL[i*sL+j] *= gamma[j]
			dL[i*sL+j] += beta[j]
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

func softSample(logits vector) int {
	rm, _ := rowMax(logits)
	sum := rowSum(logits, rm)
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

func spsa(obj objective, theta vector, samples, iters int, lr, eps float64, seed int64) {
	ones := make([]vector, samples)
	dPlus := make([]vector, samples)
	dMinus := make([]vector, samples)
	rngs := make([]*rand.Rand, samples)
	var wg sync.WaitGroup
	for i := range samples {
		ones[i] = make(vector, len(theta))
		dPlus[i] = make(vector, len(theta))
		dMinus[i] = make(vector, len(theta))
		rngs[i] = rand.New(rand.NewSource(seed + int64((i+1)*1000)))
	}
	for iter := range iters {
		for i := range samples {
			wg.Go(func() {
				rademacher(ones[i], rngs[i])
				addVec3(dPlus[i], theta, ones[i], eps)
				addVec3(dMinus[i], theta, ones[i], -eps)
			})
		}
		wg.Wait()
		plus, minus := obj.eval2(dPlus, dMinus, iter)
		for i := range samples {
			d := (plus[i] - minus[i]) / (2 * eps)
			addVec2(theta, ones[i], -d*lr/float64(samples))
		}
	}
}
