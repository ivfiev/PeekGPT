package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"slices"
	"sync"
	"time"
)

type transformer struct {
	context int
	dModel  int
	dVocab  int

	// pre-attention LayerNorm parameters
	gamma1 vector
	beta1  vector

	// attention parameters
	keys    matrix
	queries matrix
	values  matrix

	// pre-MLP LayerNorm parameters
	gamma2 vector
	beta2  vector

	// MLP
	input      matrix
	activation func(float64) float64
	hidden     matrix

	// to-logit map parameters
	linear matrix
	bias   vector

	// post-LN xs
	xs1 matrix
	xs2 matrix

	// attention values
	K  matrix
	Q  matrix
	QK matrix
	S  matrix
	V  matrix

	// residual values
	R1 matrix
	R2 matrix

	// MLP intermediary states
	I matrix
	A matrix
	H matrix

	// logits
	L matrix

	// training/predictions
	voc  []rune
	data []rune
	tok  map[rune]vector
	pos  map[int]vector
	xs   matrix // inputs
	ys   []int
}

func newT(ctx, dModel, dVocab int) *transformer {
	t := transformer{
		context: ctx,
		dModel:  dModel,
		dVocab:  dVocab,
	}

	t.xs = makeMat(t.context, t.dModel)

	t.gamma1 = make(vector, dModel)
	t.beta1 = make(vector, dModel)
	t.xs1 = makeMat(t.context, t.dModel)

	t.queries = makeMat(ctx, dModel)
	t.keys = makeMat(ctx, dModel)
	t.values = makeMat(dModel, ctx)

	t.gamma2 = make(vector, dModel)
	t.beta2 = make(vector, dModel)
	t.xs2 = makeMat(t.context, t.dModel)

	t.input = makeMat(2*dModel, dModel)
	t.hidden = makeMat(dModel, 2*dModel)

	t.linear = makeMat(dModel, dVocab)
	t.bias = make(vector, dVocab)

	t.K = makeMat(ctx, ctx)
	t.Q = makeMat(ctx, ctx)
	t.QK = makeMat(ctx, ctx)
	t.S = makeMat(ctx, ctx)
	t.V = makeMat(ctx, dModel)
	t.L = makeMat(ctx, dVocab)
	t.R1 = makeMat(t.context, t.dModel)
	t.R2 = makeMat(t.context, t.dModel)
	t.I = makeMat(ctx, 2*dModel)
	t.A = makeMat(ctx, 2*dModel)
	t.H = makeMat(ctx, dModel)

	t.ys = make([]int, t.context)

	return &t
}

func (t *transformer) run() {
	layerNorm(t.xs1, t.xs, t.gamma1, t.beta1)
	mulMatT(t.Q, t.xs1, t.queries)
	mulMatT(t.K, t.xs1, t.keys)
	mulMatT(t.QK, t.Q, t.K)
	d := 1 / math.Sqrt(float64(t.dModel))
	mulMatK(t.QK, d)
	softmax(t.S, t.QK)
	mulMatT(t.V, t.S, t.values)
	addMatM(t.R1, t.xs, t.V)

	layerNorm(t.xs2, t.R1, t.gamma2, t.beta2)
	mulMatT(t.I, t.xs2, t.input)
	mapMat(t.A, t.I, t.activation)
	mulMatT(t.H, t.A, t.hidden)
	addMatM(t.R2, t.R1, t.H)

	mulMat(t.L, t.R2, t.linear)
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
	for i := range t.gamma1 {
		t.gamma1[i] = theta[T]
		T++
	}
	for i := range t.beta1 {
		t.beta1[i] = theta[T]
		T++
	}
	for i := range t.gamma2 {
		t.gamma2[i] = theta[T]
		T++
	}
	for i := range t.beta2 {
		t.beta2[i] = theta[T]
		T++
	}
	for i := range t.keys {
		for j := range t.keys[i] {
			t.keys[i][j] = theta[T]
			T++
		}
	}
	for i := range t.queries {
		for j := range t.queries[i] {
			t.queries[i][j] = theta[T]
			T++
		}
	}
	for i := range t.values {
		for j := range t.values[i] {
			t.values[i][j] = theta[T]
			T++
		}
	}
	for i := range t.input {
		for j := range t.input[i] {
			t.input[i][j] = theta[T]
			T++
		}
	}
	for i := range t.hidden {
		for j := range t.hidden[i] {
			t.hidden[i][j] = theta[T]
			T++
		}
	}
	for i := range t.linear {
		for j := range t.linear[i] {
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
	return len(t.gamma1) + len(t.beta1) +
		len(t.keys)*len(t.keys[0]) +
		len(t.queries)*len(t.queries[0]) +
		len(t.values)*len(t.values[0]) +
		len(t.gamma2) + len(t.beta2) +
		len(t.input)*len(t.input[0]) + len(t.hidden)*len(t.hidden[0]) +
		len(t.linear)*len(t.linear[0]) + len(t.bias)
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
	mulMatK(t.xs1, 0)
	mulMatK(t.xs2, 0)
	mulMatK(t.S, 0)
	mulMatK(t.QK, 0)
	mulMatK(t.K, 0)
	mulMatK(t.Q, 0)
	mulMatK(t.V, 0)
	mulMatK(t.I, 0)
	mulMatK(t.H, 0)
	t.loadXs(ctx)
	t.run()
	println()
	fmt.Println("Full breakdown:")
	fmt.Println("Input embeddings (xs)")
	printMat(t.xs)
	println()
	fmt.Println("First LayerNorm Gamma & Beta:")
	printVec(t.gamma1)
	printVec(t.beta1)
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
	fmt.Println("Second LayerNorm Gamma & Beta:")
	printVec(t.gamma2)
	printVec(t.beta2)
	println()
	fmt.Println("MLP Input layer")
	printMat(t.input)
	println()
	fmt.Println("MLP Hidden layer")
	printMat(t.hidden)
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
	fmt.Println("After first LayerNorm:")
	printVec(t.xs1[lastIx])
	println()
	fmt.Println("Token's Query:")
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
	fmt.Println("Dot product with Value rows:")
	printMat(t.values)
	println()
	fmt.Println("To get the final Value:")
	printVec(t.V[lastIx])
	println()
	fmt.Println("Residual stream:")
	printVec(t.xs[lastIx])
	println("+")
	printVec(t.V[lastIx])
	println("=")
	printVec(t.R1[lastIx])
	println()
	fmt.Println("After second LayerNorm:")
	printVec(t.xs2[lastIx])
	println()
	fmt.Println("Pass through Input layer:")
	printVec(t.I[lastIx])
	println()
	fmt.Println("Activation:")
	printVec(t.A[lastIx])
	println()
	fmt.Println("Pass through Hidden layer:")
	printVec(t.H[lastIx])
	println()
	fmt.Println("Residual stream:")
	printVec(t.R1[lastIx])
	println("+")
	printVec(t.H[lastIx])
	println("=")
	printVec(t.R2[lastIx])
	println()
	fmt.Println("Dot product with Linear layer rows:")
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

func (t *transformer) rand(rng *rand.Rand) {
	T := 0
	for i := range t.gamma1 {
		t.gamma1[i] = 1
		T++
	}
	for i := range t.beta1 {
		t.beta1[i] = 0
		T++
	}
	for i := range t.gamma2 {
		t.gamma2[i] = 1
		T++
	}
	for i := range t.beta2 {
		t.beta2[i] = 0
		T++
	}
	for i := range t.keys {
		for j := range t.keys[i] {
			t.keys[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.queries {
		for j := range t.queries[i] {
			t.queries[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.values {
		for j := range t.values[i] {
			t.values[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.input {
		for j := range t.input[i] {
			t.input[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.hidden {
		for j := range t.hidden[i] {
			t.hidden[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.linear {
		for j := range t.linear[i] {
			t.linear[i][j] = rng.Float64() - 0.5
			T++
		}
	}
	for i := range t.bias {
		t.bias[i] = rng.Float64() - 0.5
		T++
	}
}

func (t *transformer) dump(theta vector) {
	T := 0
	for i := range t.gamma1 {
		theta[T] = t.gamma1[i]
		T++
	}
	for i := range t.beta1 {
		theta[T] = t.beta1[i]
		T++
	}
	for i := range t.gamma2 {
		theta[T] = t.gamma2[i]
		T++
	}
	for i := range t.beta2 {
		theta[T] = t.beta2[i]
		T++
	}
	for i := range t.keys {
		for j := range t.keys[i] {
			theta[T] = t.keys[i][j]
			T++
		}
	}
	for i := range t.queries {
		for j := range t.queries[i] {
			theta[T] = t.queries[i][j]
			T++
		}
	}
	for i := range t.values {
		for j := range t.values[i] {
			theta[T] = t.values[i][j]
			T++
		}
	}
	for i := range t.input {
		for j := range t.input[i] {
			theta[T] = t.input[i][j]
			T++
		}
	}
	for i := range t.hidden {
		for j := range t.hidden[i] {
			theta[T] = t.hidden[i][j]
			T++
		}
	}
	for i := range t.linear {
		for j := range t.linear[i] {
			theta[T] = t.linear[i][j]
			T++
		}
	}
	for i := range t.bias {
		theta[T] = t.bias[i]
		T++
	}
	if T != len(theta) {
		log.Fatal("mismatch between len(theta) and model size")
	}
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
	now := time.Now()
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
		t.activation = ReLU
		rngs[i] = rand.New(rand.NewSource(seed + 1000*int64(i)))
		thetas[i] = make(vector, t.size())
		t.rand(rngs[i])
		t.dump(thetas[i])
		models[i] = t
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
	fmt.Printf("Trained in %.3f seconds\n", float64(time.Now().UnixMilli()-now.UnixMilli())/1000)
	// models[winner].apply(thetas[winner])
	return models[winner]
}
