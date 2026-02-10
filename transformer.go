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

	tokens    matrix // token embeddings
	positions matrix // position embeddings

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

	// vocabulary
	vocab []rune

	// prompts/inputs
	prompt []rune
	xs     matrix // inputs
	ys     []int  // outputs, used for loss

	// graphical output
	heatmap []vector
}

func newT(dModel, dVocab, ctx int, activation func(float64) float64) *transformer {
	t := transformer{
		context: ctx,
		dModel:  dModel,
		dVocab:  dVocab,
	}
	t.tokens = makeMat(dVocab, dModel)
	t.positions = makeMat(ctx, dModel)

	t.xs = makeMat(t.context, t.dModel)
	t.ys = make([]int, ctx)

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
	softmaxT(t.S, t.QK)
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

func (t *transformer) loss() float64 {
	loss := 0.0
	count := 0
	d, r, c, s := unmat(t.L)
	for i := range r {
		if t.ys[i] == -1 {
			continue
		}
		count++
		rowMax, _ := rowMax(d[i*s : i*s+c])
		sum := 0.0
		for j := range c {
			sum += math.Exp(d[i*s+j] - rowMax)
		}
		loss += -d[i*s+t.ys[i]] + rowMax + math.Log(sum)
	}
	return loss / float64(count)
}

func (t *transformer) clone() *transformer {
	clone := newT(t.dModel, t.dVocab, t.context, t.activation)
	clone.vocab = t.vocab
	return clone
}

func (t *transformer) size() int {
	return len(t.tokens.RawMatrix().Data) +
		len(t.positions.RawMatrix().Data) +
		len(t.gamma1) + len(t.beta1) +
		len(t.gamma2) + len(t.beta2) +
		len(t.queries.RawMatrix().Data) +
		len(t.keys.RawMatrix().Data) +
		len(t.values.RawMatrix().Data) +
		len(t.input.RawMatrix().Data) +
		len(t.hidden.RawMatrix().Data) +
		len(t.linear.RawMatrix().Data) +
		len(t.bias)
}

func (t *transformer) loadXs(prompt []rune) {
	if len(prompt) > t.context {
		log.Fatal("too long xs")
	}
	t.xs.Zero()
	dx, _, _, sx := unmat(t.xs)
	dt, _, _, st := unmat(t.tokens)
	dp, _, _, sp := unmat(t.positions)
	for posIx := range prompt {
		vocIx := slices.Index(t.vocab, prompt[posIx])
		if vocIx == -1 {
			log.Panicf("loadXs: token %c is invalid", prompt[posIx])
		}
		for j := range t.dModel {
			dx[posIx*sx+j] = dt[vocIx*st+j] + dp[posIx*sp+j]
		}
	}
	t.prompt = prompt
}

func (t *transformer) predict(ctx []rune) ([]rune, vector) {
	t.loadXs(ctx)
	t.run()
	d, _, c, s := unmat(t.L)
	nexts := make([]rune, len(ctx))
	probs := make(vector, len(ctx))
	for tokIx := range len(ctx) {
		rm, i := rowMax(d[s*tokIx : s*tokIx+c])
		sum := 0.0
		for j := range c {
			sum += math.Exp(d[tokIx*s+j] - rm)
		}
		prob := math.Exp(d[tokIx*s+i]-rm) / sum
		nexts[tokIx] = t.vocab[i]
		probs[tokIx] = prob
	}
	return nexts, probs
}

func (t *transformer) generate(ctx []rune, n int) {
	fmt.Printf("%s", string(ctx))
	d, _, c, s := unmat(t.L)
	for range n {
		t.loadXs(ctx)
		t.run()
		tokIx := len(ctx) - 1
		// printVec(t.L[tokIx])
		i := softSample(d[tokIx*s : tokIx*s+c])
		fmt.Printf("%c", t.vocab[i])
		ctx = append(ctx, t.vocab[i])
		ctx = ctx[max(0, len(ctx)-t.context):]
	}
	println()
}

func (t *transformer) solve(ctx []rune) {
	t.loadXs(ctx)
	t.run()
	d, _, c, s := unmat(t.L)
	i := 1 + slices.Index(ctx, '|')
	prediction := make([]rune, 0)
	fmt.Println(string(t.vocab))
	for ; i < len(ctx); i++ {
		printVec(d[i*s : i*s+c])
		_, j := rowMax(d[i*s : i*s+c])
		prediction = append(prediction, t.vocab[j])
	}
	fmt.Println(string(prediction))
	println()
	t.printAttention()
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
	printRow(t.xs, lastIx)
	println()
	fmt.Println("After first LayerNorm:")
	printRow(t.xs1, lastIx)
	println()
	fmt.Println("Token's Query:")
	printRow(t.Q, lastIx)
	println()
	fmt.Println("Available Keys:")
	printMat(t.K)
	println()
	fmt.Println("Raw scores against Keys (QKT):")
	printRow(t.QK, lastIx)
	println()
	fmt.Println("Normalized Softmax scores:")
	printRow(t.S, lastIx)
	println()
	fmt.Println("Dot product with Value rows:")
	printMat(t.values)
	println()
	fmt.Println("To get the final Value:")
	printRow(t.V, lastIx)
	println()
	fmt.Println("Residual stream:")
	printRow(t.xs, lastIx)
	println("+")
	printRow(t.V, lastIx)
	println("=")
	printRow(t.R1, lastIx)
	println()
	fmt.Println("After second LayerNorm:")
	printRow(t.xs2, lastIx)
	println()
	fmt.Println("Pass through Input layer:")
	printRow(t.I, lastIx)
	println()
	fmt.Println("Activation:")
	printRow(t.A, lastIx)
	println()
	fmt.Println("Pass through Hidden layer:")
	printRow(t.H, lastIx)
	println()
	fmt.Println("Residual stream:")
	printRow(t.R1, lastIx)
	println("+")
	printRow(t.H, lastIx)
	println("=")
	printRow(t.R2, lastIx)
	println()
	fmt.Println("Dot product with Linear layer rows:")
	printMat(t.linear)
	println()
	fmt.Println("And add Bias:")
	printVec(t.bias)
	println()
	fmt.Println("To get the final Logits:")
	printRow(t.L, lastIx)
	println()
	fmt.Printf("Input: [%s]\n", string(ctx))
	fmt.Println("Next token probabilities:")
	d, _, c, s := unmat(t.L)
	rm, rmix := rowMax(d[lastIx*s : lastIx*s+c])
	sum := 0.0
	for i := range c {
		x := d[lastIx*s+i]
		sum += math.Exp(x - rm)
	}
	for i := range c {
		x := d[lastIx*s+i]
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
				fg = int(255 * t.S.At(i, j))
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
	lin := make(vector, t.linear.RawMatrix().Rows)
	for i := range lin {
		lin[i] = t.linear.At(i, rmix)
	}
	maxProd := math.Inf(-1)
	minProd := math.Inf(1)
	for i := range lin {
		prods := vector{
			t.R2.At(lastIx, i) * lin[i],
			t.R1.At(lastIx, i) * lin[i],
			t.xs.At(lastIx, i) * lin[i],
			t.V.At(lastIx, i) * lin[i],
			t.H.At(lastIx, i) * lin[i],
		}
		maxProd = max(maxProd, slices.Max(prods))
		minProd = min(minProd, slices.Min(prods))
	}
	printHeatmap := func(xs matrix) {
		rgb := make(vector, t.dModel) // len(lin)
		for i := range lin {
			red, blue, bg := 0, 0, 0
			prod := xs.At(lastIx, i) * lin[i]
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
	mat := func(m matrix, scale float64) {
		d, r, c, s := unmat(m)
		for i := range r {
			for j := range c {
				d[i*s+j] = scale * 2 * (rng.Float64() - 0.5)
			}
		}
	}
	mat(t.tokens, 0.25)
	mat(t.positions, 0.25)
	mat(t.keys, 0.5)
	mat(t.queries, 0.5)
	mat(t.values, 0.5)
	mat(t.input, 0.25)
	mat(t.hidden, 0.25)
	mat(t.linear, 0.25)
	for i := range t.bias {
		t.bias[i] = 0
		T++
	}
}

func (t *transformer) apply(theta vector) {
	T := 0
	vec := func(v vector) {
		copy(v, theta[T:T+len(v)])
		T += len(v)
	}
	mat := func(m matrix) {
		d, _, _, _ := unmat(m)
		copy(d, theta[T:T+len(d)])
		T += len(d)
	}
	vec(t.gamma1)
	vec(t.beta1)
	vec(t.gamma2)
	vec(t.beta2)
	mat(t.tokens)
	mat(t.positions)
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
		copy(theta[T:T+len(v)], v)
		T += len(v)
	}
	mat := func(m matrix) {
		d, _, _, _ := unmat(m)
		copy(theta[T:T+len(d)], d)
		T += len(d)
	}
	vec(t.gamma1)
	vec(t.beta1)
	vec(t.gamma2)
	vec(t.beta2)
	mat(t.tokens)
	mat(t.positions)
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
