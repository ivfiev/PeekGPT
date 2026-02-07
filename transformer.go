package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"slices"
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
	vocab []rune
	tok   map[rune]vector
	pos   map[int]vector

	// prompts/inputs
	prompt []rune
	xs     matrix // inputs

	// graphical output
	heatmap matrix
}

func newT(dModel, dVocab, ctx int, activation func(float64) float64) *transformer {
	t := transformer{
		context: ctx,
		dModel:  dModel,
		dVocab:  dVocab,
	}
	t.xs = makeMat(t.context, t.dModel)

	t.gamma1 = make(vector, dModel)
	t.beta1 = make(vector, dModel)
	t.xs1 = makeMat(ctx, dModel)

	t.queries = makeMat(ctx, dModel)
	t.keys = makeMat(ctx, dModel)
	t.values = makeMat(dModel, ctx)

	t.gamma2 = make(vector, dModel)
	t.beta2 = make(vector, dModel)
	t.xs2 = makeMat(t.context, t.dModel)

	t.input = makeMat(dModel, dModel)
	t.activation = activation
	t.hidden = makeMat(dModel, dModel)

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
	t.I = makeMat(ctx, dModel)
	t.A = makeMat(ctx, dModel)
	t.H = makeMat(ctx, dModel)

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

func (t *transformer) loss(ys []int) float64 {
	loss := 0.0
	for i := range len(t.L) {
		rowMax, _ := rowMax(t.L[i])
		sum := 0.0
		for j := range len(t.L[i]) {
			sum += math.Exp(t.L[i][j] - rowMax)
		}
		loss += -t.L[i][ys[i]] + rowMax + math.Log(sum)
	}
	return loss / float64(t.context)
}

func (t *transformer) clone() *transformer {
	clone := newT(t.dModel, t.dVocab, t.context, t.activation)
	clone.tok = t.tok
	clone.pos = t.pos
	clone.vocab = t.vocab
	return clone
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

func (t *transformer) loadXs(prompt []rune) {
	if len(prompt) > t.context {
		log.Fatal("too long xs")
	}
	for i := range prompt {
		tok, ok := t.tok[prompt[i]]
		if !ok {
			log.Panicf("loadXs: token %c is invalid", prompt[i])
		}
		pos, ok := t.pos[i]
		if !ok {
			log.Panicf("loadXs: pos %d is invalid", i)
		}
		addVec3(t.xs[i], tok, pos, 1)
	}
	t.prompt = prompt
}

func (t *transformer) predict(ctx []rune) (rune, float64) {
	t.loadXs(ctx)
	t.run()
	tokIx := len(ctx) - 1
	rm, i := rowMax(t.L[tokIx])
	sum := 0.0
	for j := range t.L[tokIx] {
		sum += math.Exp(t.L[tokIx][j] - rm)
	}
	prob := math.Exp(t.L[tokIx][i]-rm) / sum
	return t.vocab[i], prob
}

func (t *transformer) generate(ctx []rune, n int) {
	fmt.Printf("%s", string(ctx))
	for range n {
		t.loadXs(ctx)
		t.run()
		tokIx := len(ctx) - 1
		// printVec(t.L[tokIx])
		i := softSample(t.L[tokIx])
		fmt.Printf("%c", t.vocab[i])
		ctx = append(ctx, t.vocab[i])
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
	rm, rmix := rowMax(t.L[lastIx])
	sum := 0.0
	for _, x := range t.L[lastIx] {
		sum += math.Exp(x - rm)
	}
	for i, x := range t.L[lastIx] {
		fmt.Printf("[%c] -> %.6f\n", t.vocab[i], math.Exp(x-rm)/sum)
	}
	println()
	println()
	t.printAttention()
	println()
	println()
	t.printHeatmap(lastIx, rmix)
	println()
}

func (t *transformer) printAttention() {
	for i := range t.prompt {
		fmt.Printf("%c ", t.prompt[i])
		for j := range t.prompt {
			fg, bg := 0, 0
			if j <= i {
				fg = int(255 * t.S[i][j])
			}
			fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm███\x1b[0m", fg, fg, fg, bg, bg, bg)
		}
		println()
	}
	fmt.Printf("   ")
	for _, c := range t.prompt {
		fmt.Printf("%c  ", c)
	}
}

func (t *transformer) printHeatmap(lastIx, rmix int) {
	t.heatmap = nil
	lin := make(vector, len(t.linear))
	for i := range lin {
		lin[i] = t.linear[i][rmix]
	}
	maxProd := math.Inf(-1)
	minProd := math.Inf(1)
	for i := range lin {
		prods := vector{
			t.R2[lastIx][i] * lin[i],
			t.R1[lastIx][i] * lin[i],
			t.xs[lastIx][i] * lin[i],
			t.V[lastIx][i] * lin[i],
			t.H[lastIx][i] * lin[i],
		}
		maxProd = max(maxProd, slices.Max(prods))
		minProd = min(minProd, slices.Min(prods))
	}
	printHeatmap := func(xs matrix) {
		rgb := make(vector, t.dModel) // len(lin)
		for i := range lin {
			red, blue, bg := 0, 0, 0
			prod := xs[lastIx][i] * lin[i]
			if prod > 0 {
				rgb[i] = prod / (maxProd - minProd)
				red = int(rgb[i] * 255)
			} else {
				rgb[i] = prod / (maxProd - minProd)
				blue = int(-rgb[i] * 255)
			}
			fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm██\x1b[0m", red, 0, blue, bg, bg, bg)
		}
		t.heatmap = append(t.heatmap, rgb)
	}
	printHeatmap(t.xs)
	println("  Original")
	println()
	printHeatmap(t.V)
	println("  Attention Δ")
	println()
	printHeatmap(t.R1)
	println("  Post-attention")
	println()
	printHeatmap(t.H)
	println("  MLP Δ")
	println()
	printHeatmap(t.R2)
	println("  Post-MLP")
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
	mat := func(m matrix) {
		for i := range m {
			for j := range m[i] {
				m[i][j] = rng.Float64() - 0.5
			}
		}
	}
	mat(t.keys)
	mat(t.queries)
	mat(t.values)
	mat(t.input)
	mat(t.hidden)
	mat(t.linear)
	for i := range t.bias {
		t.bias[i] = rng.Float64() - 0.5
		T++
	}
}

func (t *transformer) apply(theta vector) {
	T := 0
	vec := func(v vector) {
		for i := range v {
			v[i] = theta[T]
			T++
		}
	}
	mat := func(m matrix) {
		for i := range m {
			for j := range m[i] {
				m[i][j] = theta[T]
				T++
			}
		}
	}
	vec(t.gamma1)
	vec(t.beta1)
	vec(t.gamma2)
	vec(t.beta2)
	mat(t.keys)
	mat(t.queries)
	mat(t.values)
	mat(t.input)
	mat(t.hidden)
	mat(t.linear)
	vec(t.bias)
	if T != len(theta) {
		log.Fatal("mismatch between len(theta) and model size")
	}
}

func (t *transformer) dump(theta vector) {
	T := 0
	vec := func(v vector) {
		for i := range v {
			theta[T] = v[i]
			T++
		}
	}
	mat := func(m matrix) {
		for i := range m {
			for j := range m[i] {
				theta[T] = m[i][j]
				T++
			}
		}
	}
	vec(t.gamma1)
	vec(t.beta1)
	vec(t.gamma2)
	vec(t.beta2)
	mat(t.keys)
	mat(t.queries)
	mat(t.values)
	mat(t.input)
	mat(t.hidden)
	mat(t.linear)
	vec(t.bias)
	if T != len(theta) {
		log.Fatal("mismatch between len(theta) and model size")
	}
}
