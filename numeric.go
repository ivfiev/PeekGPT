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

func makeMat(r, c int) matrix {
	return mat.NewDense(r, c, nil)
}

func unmat(A matrix) ([]float64, int, int) {
	raw := A.RawMatrix()
	if raw.Stride != raw.Cols {
		log.Panicf("Stride %d != Cols %d", raw.Stride, raw.Cols)
	}
	return raw.Data, raw.Rows, raw.Cols
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
	d, r, c := unmat(A)
	if c != len(v) {
		log.Panicf("addMatV: bad dimensions, %d + %d\n", c, len(v))
	}
	for i := range r {
		for j := range c {
			d[i*c+j] += v[j]
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
	db, rb, cb := unmat(B)
	_, ra, ca := unmat(As[0])
	if ra != rb || ca*len(As) != cb {
		log.Panicf("catMat: bad dims %dx%d %dx%d\n", rb, cb, ra, ca)
	}
	for _, A := range As {
		_, r, c := unmat(A)
		if ra != r || ca != c {
			log.Panicf("catMat: bad dims %dx%d %dx%d\n", ra, ca, r, c)
		}
	}
	for c, A := range As {
		da, _, ca := unmat(A)
		for r := range rb {
			copy(db[r*cb+c*ca:r*cb+(1+c)*ca], da[r*ca:r*ca+ca])
		}
	}
}

func printMat(A matrix) {
	d, r, c := unmat(A)
	for i := range r {
		fmt.Printf("[")
		for j := range c {
			fmt.Printf("%9.6f ", d[i*c+j])
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
	d, _, c := unmat(A)
	printVec(d[i*c : i*c+c])
}

func zeroVec(v vector) {
	for i := range len(v) {
		v[i] = 0
	}
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

func sumCols(v vector, A matrix) {
	a, _, c := unmat(A)
	for j := range c {
		sum := 0.0
		for i := 0; i < len(a)/c; i++ {
			sum += a[i*c+j]
		}
		v[j] = sum
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
	dA, rA, cA := unmat(A)
	dS, rS, cS := unmat(S)
	if rS != rA || cS != cA {
		log.Panicf("Softmax: incompatible matrices, A: %dx%d, S: %dx%d\n", rA, cA, rS, cS)
	}
	S.Zero()
	for i := range rA {
		triangle := i + 1
		rowA := dA[i*cA : i*cA+triangle]
		rowS := dS[i*cS : i*cS+triangle]
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
	dX, rX, cX := unmat(X)
	dL, rL, cL := unmat(L)
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
			u += dX[i*cX+j]
		}
		if u == 0 {
			continue
		}
		u /= float64(cols)
		for j := range cols {
			x := dX[i*cX+j]
			o2 += (x - u) * (x - u)
		}
		o2 /= float64(cols)
		for j := range cols {
			dL[i*cL+j] = (dX[i*cX+j] - u) / math.Sqrt(o2+0.00001)
			dL[i*cL+j] *= gamma[j]
			dL[i*cL+j] += beta[j]
		}
	}
}

func ReLU(x float64) float64 {
	return max(0, x)
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

func sgd(t *training, theta vector, iters int, lr float64) {
	grad := make(vector, len(theta))
	for i := range iters {
		zeroVec(grad)
		t.eval(theta, grad, i)
		addVec2(theta, grad, -lr)
	}
}

func adam(t *training, theta vector, iters int, lr float64) {
	const (
		b1  = 0.9
		b2  = 0.999
		eps = 1e-8
	)
	grad := make(vector, len(theta))
	m := make(vector, len(theta))
	v := make(vector, len(theta))
	b1t, b2t := 1.0, 1.0
	for iter := range iters {
		zeroVec(grad)
		t.eval(theta, grad, 1+iter)
		b1t *= b1
		b2t *= b2
		for i := range grad {
			m[i] = b1*m[i] + (1-b1)*grad[i]
			v[i] = b2*v[i] + (1-b2)*grad[i]*grad[i]
			mhat := m[i] / (1 - b1t)
			vhat := v[i] / (1 - b2t)
			dt := mhat / (math.Sqrt(vhat) + eps)
			theta[i] -= lr * dt
		}
	}
}
