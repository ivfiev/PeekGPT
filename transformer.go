package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"slices"
)

type transformer struct {
	dModel  int
	context int

	vocab     []rune // vocabulary
	tokens    matrix // token embeddings
	positions matrix // position embeddings

	prompt []rune
	XS     matrix // inputs
	ys     []int  // outputs, used for loss

	blocks []*block

	// to-logit map parameters
	linear matrix
	bias2  vector

	// logits
	L matrix
}

type block struct {
	dModel int
	// pre-attention LayerNorm parameters
	gamma0 vector
	beta0  vector

	// attention parameters
	keys    matrix
	queries matrix
	values  matrix
	proj    matrix

	// pre-MLP LayerNorm parameters
	gamma1 vector
	beta1  vector

	// MLP
	input      matrix
	bias0      vector
	activation func(float64) float64
	hidden     matrix
	bias1      vector

	// inputs
	XS0 matrix
	XS1 matrix
	XS2 matrix

	// attention values
	K  matrix
	Q  matrix
	V  matrix
	QK matrix
	S  matrix
	SV matrix
	P  matrix

	// residual values
	R0 matrix
	R1 matrix

	// MLP intermediary states
	I matrix
	A matrix
	H matrix
}

func newT(dModel, ctx, blocks int, vocab []rune) *transformer {
	dVocab := len(vocab)
	t := transformer{
		context: ctx,
		dModel:  dModel,
	}
	t.tokens = makeMat(dVocab, dModel)
	t.positions = makeMat(ctx, dModel)
	t.XS = makeMat(ctx, dModel)

	t.blocks = make([]*block, blocks)
	for i := range blocks {
		t.blocks[i] = newB(dModel, ctx, ReLU)
	}

	t.linear = makeMat(dModel, dVocab)
	t.bias2 = make(vector, dVocab)
	t.L = makeMat(ctx, dVocab)

	t.ys = make([]int, ctx)
	t.vocab = vocab

	return &t
}

func newB(dModel, ctx int, activation func(float64) float64) *block {
	b := block{
		dModel: dModel,
	}

	b.XS0 = makeMat(ctx, dModel)

	b.gamma0 = make(vector, dModel)
	b.beta0 = make(vector, dModel)
	b.XS1 = makeMat(ctx, dModel)

	b.queries = makeMat(dModel, dModel)
	b.keys = makeMat(dModel, dModel)
	b.values = makeMat(dModel, dModel)
	b.proj = makeMat(dModel, dModel)

	b.gamma1 = make(vector, dModel)
	b.beta1 = make(vector, dModel)
	b.XS2 = makeMat(ctx, dModel)

	b.input = makeMat(dModel, dModel)
	b.bias0 = make(vector, dModel)
	b.activation = activation
	b.hidden = makeMat(dModel, dModel)
	b.bias1 = make(vector, dModel)

	b.K = makeMat(ctx, dModel)
	b.Q = makeMat(ctx, dModel)
	b.V = makeMat(ctx, dModel)
	b.QK = makeMat(ctx, ctx)
	b.S = makeMat(ctx, ctx)
	b.SV = makeMat(ctx, dModel)
	b.P = makeMat(ctx, dModel)
	b.R0 = makeMat(ctx, dModel)
	b.R1 = makeMat(ctx, dModel)
	b.I = makeMat(ctx, dModel)
	b.A = makeMat(ctx, dModel)
	b.H = makeMat(ctx, dModel)

	return &b
}

func (t *transformer) run() {
	xs := t.XS
	for _, b := range t.blocks {
		b.loadXs(xs)
		b.run()
		xs = b.R1
	}
	mulMat(t.L, xs, t.linear)
	addMatV(t.L, t.bias2)
}

func (b *block) run() {
	// attention
	layerNorm(b.XS1, b.XS0, b.gamma0, b.beta0)
	mulMat(b.Q, b.XS1, b.queries)
	mulMat(b.K, b.XS1, b.keys)
	mulMat(b.V, b.XS1, b.values)
	mulMatT(b.QK, b.Q, b.K)
	d := 1 / math.Sqrt(float64(b.dModel))
	mulMatK(b.QK, d)
	softmaxT(b.S, b.QK)
	mulMat(b.SV, b.S, b.V)
	mulMat(b.P, b.SV, b.proj)
	addMatM(b.R0, b.XS0, b.P)

	// mlp
	layerNorm(b.XS2, b.R0, b.gamma1, b.beta1)
	mulMat(b.I, b.XS2, b.input)
	addMatV(b.I, b.bias0)
	mapMat(b.A, b.I, b.activation)
	mulMat(b.H, b.A, b.hidden)
	addMatV(b.H, b.bias1)
	addMatM(b.R1, b.R0, b.H)
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
		row := d[i*s : i*s+c]
		rowMax, _ := rowMax(row)
		sum := rowSum(row, rowMax)
		loss += -d[i*s+t.ys[i]] + rowMax + math.Log(sum)
	}
	return loss / float64(count)
}

func (t *transformer) clone() *transformer {
	return newT(t.dModel, t.context, len(t.blocks), t.vocab)
}

func (t *transformer) size() int {
	blocks := 0
	for _, b := range t.blocks {
		blocks += b.size()
	}
	return blocks +
		len(t.tokens.RawMatrix().Data) +
		len(t.positions.RawMatrix().Data) +
		len(t.linear.RawMatrix().Data) +
		len(t.bias2)
}

func (b *block) size() int {
	return len(b.gamma0) + len(b.beta0) +
		len(b.gamma1) + len(b.beta1) +
		len(b.queries.RawMatrix().Data) +
		len(b.keys.RawMatrix().Data) +
		len(b.values.RawMatrix().Data) +
		len(b.proj.RawMatrix().Data) +
		len(b.input.RawMatrix().Data) +
		len(b.bias0) +
		len(b.hidden.RawMatrix().Data) +
		len(b.bias1)
}

func (t *transformer) loadXs(prompt []rune) {
	if len(prompt) > t.context {
		log.Fatal("too long xs")
	}
	t.XS.Zero()
	for _, b := range t.blocks {
		b.XS0.Zero()
		b.XS1.Zero()
		b.XS2.Zero()
	}
	dx, _, _, sx := unmat(t.XS)
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

func (b *block) loadXs(xs matrix) {
	dxs, rxs, cxs, _ := unmat(xs)
	dxs0, rxs0, cxs0, _ := unmat(b.XS0)
	if rxs != rxs0 || cxs != cxs0 {
		log.Panic("Incompatible XS")
	}
	copy(dxs0, dxs)
}

func (t *transformer) predict(ctx []rune) ([]rune, vector) {
	t.loadXs(ctx)
	t.run()
	d, _, c, s := unmat(t.L)
	nexts := make([]rune, len(ctx))
	probs := make(vector, len(ctx))
	for tokIx := range len(ctx) {
		rm, i := rowMax(d[s*tokIx : s*tokIx+c])
		sum := rowSum(d[tokIx*s:tokIx*s+c], rm)
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
	// fmt.Println(string(t.vocab))
	xs := []int{}
	for ; i < len(ctx); i++ {
		// printVec(d[i*s : i*s+c])
		_, j := rowMax(d[i*s : i*s+c])
		prediction = append(prediction, t.vocab[j])
		xs = append(xs, i)
	}
	println()
	t.printHeatmap(xs)
	println()
	t.printAttention()
	println()
	fmt.Printf("Input: [%s]\nPrediction: [%s]\n\n", string(ctx), string(prediction))
}

func (t *transformer) rand(rng *rand.Rand) {
	mat := func(m matrix, scale float64) {
		d, r, c, s := unmat(m)
		for i := range r {
			for j := range c {
				d[i*s+j] = scale * 2 * (rng.Float64() - 0.5)
			}
		}
	}
	mat(t.tokens, 0.2)
	mat(t.positions, 0.2)
	for _, b := range t.blocks {
		for i := range b.gamma0 {
			b.gamma0[i] = 1
		}
		for i := range b.beta0 {
			b.beta0[i] = 0
		}
		for i := range b.gamma1 {
			b.gamma1[i] = 1
		}
		for i := range b.beta1 {
			b.beta1[i] = 0
		}
		mat(b.keys, 0.2)
		mat(b.queries, 0.2)
		mat(b.values, 0.2)
		mat(b.proj, 0.2)
		mat(b.input, 0.2)
		mat(b.hidden, 0.2)
		for i := range b.bias0 {
			b.bias0[i] = 0
		}
		for i := range b.bias1 {
			b.bias1[i] = 0
		}
	}
	mat(t.linear, 0.2)
	for i := range t.bias2 {
		t.bias2[i] = 0
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
	mat(t.tokens)
	mat(t.positions)
	for _, b := range t.blocks {
		vec(b.gamma0)
		vec(b.beta0)
		vec(b.gamma1)
		vec(b.beta1)
		mat(b.keys)
		mat(b.queries)
		mat(b.values)
		mat(b.proj)
		mat(b.input)
		vec(b.bias0)
		mat(b.hidden)
		vec(b.bias1)
	}
	mat(t.linear)
	vec(t.bias2)
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
	mat(t.tokens)
	mat(t.positions)
	for _, b := range t.blocks {
		vec(b.gamma0)
		vec(b.beta0)
		vec(b.gamma1)
		vec(b.beta1)
		mat(b.keys)
		mat(b.queries)
		mat(b.values)
		mat(b.proj)
		mat(b.input)
		vec(b.bias0)
		mat(b.hidden)
		vec(b.bias1)
	}
	mat(t.linear)
	vec(t.bias2)
	if T != len(theta) {
		log.Fatal("mismatch between len(theta) and model size")
	}
}
