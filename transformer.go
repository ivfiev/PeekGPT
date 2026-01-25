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

	K  matrix
	Q  matrix
	QK matrix
	S  matrix
	V  matrix
	L  matrix // logits

	// training/predictions
	data []rune
	tok  map[rune]vector
	pos  map[int]vector
	xs   matrix
	ys   []int
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
	t.QK = makeMat(ctx, ctx)
	t.S = makeMat(ctx, ctx)
	t.V = makeMat(ctx, dModel)
	t.L = makeMat(ctx, dVocab)
	t.xs = makeMat(t.context, t.dModel)
	t.ys = make([]int, t.context)
	return &t
}

func (t *transformer) run() {
	mulMatT(t.Q, t.xs, t.queries)
	mulMatT(t.K, t.xs, t.keys)
	mulMatT(t.QK, t.Q, t.K)
	d := 1 / scalar(math.Sqrt(float64(t.dModel)))
	mulMatK(t.QK, d)
	softmax(t.S, t.QK)
	mulMatT(t.V, t.S, t.values)
	mulMat(t.L, t.V, t.linear)
	addMatV(t.L, t.bias)
}

func (t *transformer) eval(theta vector) scalar {
	t.apply(theta)
	loss := scalar(0.0)
	for w := range len(t.data) - t.context { // this assumes full-context training
		t.loadXs(t.data[w : w+t.context])
		t.loadYs(t.data[w+1 : w+t.context+1])
		t.run()
		for i := range len(t.L) {
			rowMax, _ := rowMax(t.L[i])
			sum := 0.0
			for j := range len(t.L[i]) {
				sum += math.Exp(float64(t.L[i][j] - rowMax))
			}
			loss += -t.L[i][t.ys[i]] + rowMax + scalar(math.Log(sum))
		}
	}
	loss /= scalar(t.context * (len(t.data) - t.context))
	return scalar(loss)
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

func (t *transformer) size() int {
	return len(t.keys)*len(t.keys[0]) + len(t.queries)*len(t.queries[0]) + len(t.values)*len(t.values[0]) + len(t.linear)*len(t.linear[0]) + len(t.bias)
}

func (t *transformer) clone() objective {
	clone := newT(t.context, t.dModel, t.dVocab)
	clone.data = t.data
	clone.tok = t.tok
	clone.pos = t.pos
	return clone
}

func (t *transformer) loadXs(window []rune) {
	if len(window) > t.context {
		log.Fatal("too long xs")
	}
	for i := range window {
		copy(t.xs[i], t.tok[window[i]])
		addVec(t.xs[i], t.pos[i], 1)
	}
}

func (t *transformer) loadYs(window []rune) {
	if len(window) > t.context {
		log.Fatal("too long ys")
	}
	for i := range window {
		t.ys[i] = int(window[i] - rune('a'))
	}
}

func (t *transformer) predict(ctx []rune) {
	t.loadXs(ctx)
	t.run()
	printMat(t.L)
	_, i := rowMax(t.L[len(ctx)-1])
	fmt.Printf("%c\n", t.data[i])
}
