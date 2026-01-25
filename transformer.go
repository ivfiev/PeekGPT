package main

import (
	"fmt"
	"log"
	"math"
)

type transformer struct {
	context int
	dModel  int
	dVocab  int

	keys    matrix
	queries matrix
	values  matrix

	linear matrix
	bias   vector

	K matrix
	Q matrix
	Z matrix
	S matrix
	V matrix
	L matrix
}

func newT(ctx, dModel, dVocab int) *transformer {
	t := transformer{
		context: ctx,
		dModel:  dModel,
		dVocab:  dVocab,
	}
	t.queries = makeMat(ctx, dModel)
	t.keys = makeMat(ctx, dModel)
	t.values = makeMat(dModel, ctx)
	t.linear = makeMat(dModel, dVocab)
	t.bias = make(vector, dVocab)
	t.K = makeMat(ctx, ctx)
	t.Q = makeMat(ctx, ctx)
	t.Z = makeMat(ctx, ctx)
	t.S = makeMat(ctx, ctx)
	t.V = makeMat(ctx, dModel)
	t.L = makeMat(ctx, dVocab)
	return &t
}

func (t *transformer) run(embeds matrix) matrix {
	if len(embeds) != t.context || len(embeds[0]) != t.dModel {
		log.Fatalf("run: bad inputs")
	}
	mulMatT(t.Q, embeds, t.queries)
	mulMatT(t.K, embeds, t.keys)
	mulMatT(t.Z, t.Q, t.K)
	d := 1 / scalar(math.Sqrt(float64(t.dModel)))
	mulMatK(t.Z, d)
	softmax(t.S, t.Z)
	mulMatT(t.V, t.S, t.values)
	mulMat(t.L, t.V, t.linear)
	addMatV(t.L, t.bias)
	return t.L
}

func (t *transformer) apply(theta vector) {
	T := 0
	for i := range t.keys {
		for j := range t.keys[0] {
			t.keys[i][j] = theta[T]
			T++
		}
	}
	for i := range t.queries {
		for j := range t.queries[0] {
			t.queries[i][j] = theta[T]
			T++
		}
	}
	for i := range t.values {
		for j := range t.values[0] {
			t.values[i][j] = theta[T]
			T++
		}
	}
	for i := range t.linear {
		for j := range t.linear[0] {
			t.linear[i][j] = theta[T]
			T++
		}
	}
	for i := range t.bias {
		t.bias[i] = theta[T]
		T++
	}
	if T != len(theta) {
		log.Fatal("mismatch between len(theta) and model size")
	}
}

// training
type training struct {
	t    *transformer
	data []rune
	tok  map[rune]vector
	pos  map[int]vector
	xs   matrix
	ys   []int
}

func (tr *training) Size() int {
	t := tr.t
	return len(t.keys)*len(t.keys[0]) + len(t.queries)*len(t.queries[0]) + len(t.values)*len(t.values[0]) + len(t.linear)*len(t.linear[0]) + len(t.bias)
}

func (tr *training) Clone() Objective {
	clone := training{t: newT(tr.t.context, tr.t.dModel, tr.t.dVocab), data: tr.data, tok: tr.tok, pos: tr.pos}
	return &clone
}

func (tr *training) load(window []rune) {
	for i := range window {
		copy(tr.xs[i], tr.tok[window[i]])
		addVec(tr.xs[i], tr.pos[i], 1)
	}
}

func (tr *training) Eval(theta vector) scalar {
	if tr.xs == nil {
		tr.xs = makeMat(tr.t.context, tr.t.dModel)
	}
	if tr.ys == nil {
		tr.ys = make([]int, tr.t.context)
	}
	tr.t.apply(theta)
	loss := scalar(0.0)
	for w := range len(tr.data) - tr.t.context {
		tr.load(tr.data[w : w+tr.t.context])
		yrs := tr.data[w+1 : w+tr.t.context+1]
		for i := range yrs {
			tr.ys[i] = int(yrs[i] - rune('a'))
		}
		ys := tr.t.run(tr.xs)
		for i := range len(ys) {
			rowMax := scalar(math.Inf(-1))
			for j := range len(ys[0]) {
				if ys[i][j] > rowMax {
					rowMax = ys[i][j]
				}
			}
			sum := 0.0
			for j := range len(ys[0]) {
				sum += math.Exp(float64(ys[i][j] - rowMax))
			}
			loss += -ys[i][tr.ys[i]] + rowMax + scalar(math.Log(sum))
		}
	}
	loss /= scalar(tr.t.context * (len(tr.data) - tr.t.context))
	println(loss)
	return scalar(loss)
}

func (tr *training) Predict(ctx []rune) {
	tr.load(ctx)
	tr.t.run(tr.xs)
	printMat(tr.t.L)
	rowMax := scalar(-9999)
	maxI := -1
	row := tr.t.L[len(ctx)-1]
	for i, v := range row {
		if v > rowMax {
			rowMax = v
			maxI = i
		}
	}
	fmt.Printf("%c\n", tr.data[maxI])
}
