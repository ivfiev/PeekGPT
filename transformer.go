package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"slices"
	"sync"
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
	voc  []rune
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
	d := 1 / math.Sqrt(float64(t.dModel))
	mulMatK(t.QK, d)
	softmax(t.S, t.QK)
	mulMatT(t.V, t.S, t.values)
	mulMat(t.L, t.V, t.linear)
	addMatV(t.L, t.bias)
}

func (t *transformer) eval(theta vector) float64 {
	t.apply(theta)
	loss := 0.0
	for w := range len(t.data) - t.context { // this assumes full-context training
		t.loadXs(t.data[w : w+t.context])
		t.loadYs(t.data[w+1 : w+t.context+1])
		t.run()
		for i := range len(t.L) {
			rowMax, _ := rowMax(t.L[i])
			sum := 0.0
			for j := range len(t.L[i]) {
				sum += math.Exp(t.L[i][j] - rowMax)
			}
			loss += -t.L[i][t.ys[i]] + rowMax + math.Log(sum)
		}
	}
	loss /= float64(t.context * (len(t.data) - t.context))
	return loss
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

func (t *transformer) loadXs(window []rune) {
	if len(window) > t.context {
		log.Fatal("too long xs")
	}
	for i := range window {
		tok, ok := t.tok[window[i]]
		if !ok {
			log.Fatalf("loadXs: token %c is invalid", window[i])
		}
		copy(t.xs[i], tok)
		addVec(t.xs[i], t.pos[i], 1)
	}
}

func (t *transformer) loadYs(window []rune) {
	if len(window) > t.context {
		log.Fatal("too long ys")
	}
	for i := range window {
		ix := slices.Index(t.voc, window[i])
		if ix == -1 {
			log.Fatalf("loadYs: token %c is invalid", window[i])
		}
		t.ys[i] = ix
	}
}

func (t *transformer) predict(ctx []rune) {
	t.loadXs(ctx)
	t.run()
	printMat(t.L)
	tokIx := len(ctx) - 1
	rm, i := rowMax(t.L[tokIx])
	sum := 0.0
	for j := range t.L[tokIx] {
		sum += math.Exp(t.L[tokIx][j] - rm)
	}
	prob := math.Exp(t.L[tokIx][i]-rm) / sum
	fmt.Printf("%s -> %c (%.3f)\n", string(ctx), t.voc[i], prob)
}

func (t *transformer) generate(ctx []rune, n int) {
	fmt.Printf("%s", string(ctx))
	for range n {
		t.loadXs(ctx)
		t.run()
		tokIx := len(ctx) - 1
		// printVec(t.L[tokIx])
		i := softSample(t.L[tokIx])
		fmt.Printf("%c", t.voc[i])
		ctx = append(ctx, t.voc[i])
		ctx = ctx[max(0, len(ctx)-t.context):]
	}
	println()
}

func (t *transformer) peek(ctx []rune) {
	mulMatK(t.xs, 0)
	t.loadXs(ctx)
	t.run()
	println()
	fmt.Println("Full breakdown:")
	fmt.Println("Input embeddings (xs)")
	printMat(t.xs)
	println()
	fmt.Println("Queries")
	printMat(t.queries)
	println()
	fmt.Println("Q (xs * queries)")
	printMat(t.Q)
	println()
	fmt.Println("Keys")
	printMat(t.keys)
	println()
	fmt.Println("K (xs * keys)")
	printMat(t.K)
	println()
	fmt.Println("QK")
	printMat(t.QK)
	println()
	fmt.Println("S (triangular softmax QK)")
	printMat(t.S)
	println()
	fmt.Println("Values")
	printMat(t.values)
	println()
	fmt.Println("V (softmax * values)")
	printMat(t.V)
	println()
	fmt.Println("Linear")
	printMat(t.linear)
	println()
	fmt.Println("Bias")
	printVec(t.bias)
	println()
	fmt.Println("Logits (V * Linear + Bias)")
	printMat(t.L)
	println()
	fmt.Println("--------------------------------------")
	fmt.Println("Detailed breakdown for the last token:")
	lastIx := len(ctx) - 1
	fmt.Println("Last token's embedding:")
	printVec(t.xs[lastIx])
	println()
	fmt.Println("Last token's Query:")
	printVec(t.Q[lastIx])
	println()
	fmt.Println("Available Keys:")
	printMat(t.K)
	println()
	fmt.Println("Raw scores against Keys (QKT):")
	printVec(t.QK[lastIx])
	println()
	fmt.Println("Normalized Softmax scores:")
	printVec(t.S[lastIx])
	println()
	fmt.Println("Linearly combine Value columns:")
	printMat(t.values)
	println()
	fmt.Println("To get the final Value:")
	printVec(t.V[lastIx])
	println()
	fmt.Println("Lineraly combine Linear layer rows:")
	printMat(t.linear)
	println()
	fmt.Println("And add Bias:")
	printVec(t.bias)
	println()
	fmt.Println("To get the final Logits:")
	printVec(t.L[lastIx])
	println()
	fmt.Printf("Input: [%s]\n", string(ctx))
	fmt.Println("Next token probabilities:")
	rm, _ := rowMax(t.L[lastIx])
	sum := 0.0
	for _, x := range t.L[lastIx] {
		sum += math.Exp(x - rm)
	}
	for i, x := range t.L[lastIx] {
		fmt.Printf("[%c] -> %.6f\n", t.voc[i], math.Exp(x-rm)/sum)
	}
	println()
}

func embeds(vocab, ctx int, toks []rune) (map[rune]vector, map[int]vector) {
	dModel := vocab + ctx
	tokens := map[rune]vector{}
	pos := map[int]vector{}
	for i := range vocab {
		tokens[toks[i]] = onehot(dModel, i)
	}
	for i := range ctx {
		pos[i] = onehot(dModel, vocab+i)
	}
	return tokens, pos
}

func trainModel(context int, data []rune, parallelism int, seed int64, iters int, lr, eps float64) *transformer {
	vocab := make([]rune, 0, len(data))
	vocmap := map[rune]struct{}{}
	for _, tok := range data {
		_, ok := vocmap[tok]
		if !ok {
			vocab = append(vocab, tok)
			vocmap[tok] = struct{}{}
		}
	}
	tokens, positions := embeds(len(vocab), context, vocab)
	models := make([]*transformer, parallelism)
	thetas := make([]vector, parallelism)
	rngs := make([]*rand.Rand, parallelism)
	for i := range parallelism {
		t := newT(context, len(vocab)+context, len(vocab))
		t.data = data
		t.voc = vocab
		t.pos = positions
		t.tok = tokens
		models[i] = t
		rngs[i] = rand.New(rand.NewSource(seed + 1000*int64(i)))
		thetas[i] = make(vector, t.size())
		for t := range len(thetas[i]) {
			thetas[i][t] = rngs[i].Float64() - 0.5
		}
	}
	var wg sync.WaitGroup
	for i := range parallelism {
		wg.Go(func() {
			spsa(models[i], thetas[i], iters, lr, eps, rngs[i])
		})
	}
	wg.Wait()
	winner := -1
	minLoss := math.Inf(1)
	for i := range parallelism {
		loss := models[i].eval(thetas[i])
		if loss < minLoss {
			minLoss = loss
			winner = i
		}
		fmt.Printf("Model %d has loss %.4f\n", i, loss)
	}
	fmt.Printf("The winner is %d! (%d parameters, d_model %d, ctx %d, vocab %d, %.4f loss)\n",
		winner,
		models[winner].size(),
		models[winner].dModel,
		models[winner].context,
		models[winner].dVocab,
		models[winner].eval(thetas[winner]))
	fmt.Printf("Seed used %d\n\n", seed)
	// models[winner].apply(thetas[winner])
	return models[winner]
}
