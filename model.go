package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"slices"
)

type model struct {
	dModel  int
	context int
	dAttn   int
	attn    int
	mlp     int

	vocab     []rune // vocabulary
	tokens    matrix // token embeddings
	positions matrix // position embeddings

	prompt []rune
	XS     matrix // inputs
	ys     []int  // output targets

	blocks []*block

	// pre-logit layernorm
	gamma2 vector
	beta2  vector
	XS3    matrix
	hatXS3 matrix
	// to-logit map parameters
	unembed matrix
	bias2   vector

	L matrix // logits
	S matrix // output probs

	// derivatives
	dunembed    matrix
	dbias2      vector
	dgamma2     vector
	dbeta2      vector
	dL          matrix
	dXS3        matrix
	dhatXS3     matrix
	dXS3ThatXS3 matrix

	dtokens    matrix
	dpositions matrix
}

type block struct {
	dModel  int
	context int
	dAttn   int
	attn    int
	// pre-attention LayerNorm parameters
	gamma0 vector
	beta0  vector

	// attention parameters
	queries []matrix
	keys    []matrix
	values  []matrix
	proj    matrix

	// pre-MLP LayerNorm parameters
	gamma1 vector
	beta1  vector

	// MLP
	input  matrix
	bias0  vector
	hidden matrix
	bias1  vector

	// inputs
	XS0 matrix
	XS1 matrix
	XS2 matrix

	hatXS1 matrix // layernorm intermediaries
	hatXS2 matrix // layernorm intermediaries

	// attention values
	Q  []matrix
	K  []matrix
	V  []matrix
	QK []matrix
	S  []matrix
	SV []matrix
	CV matrix
	P  matrix

	// residual values
	R0 matrix
	R1 matrix

	// MLP intermediary states
	I matrix
	A matrix
	H matrix

	// derivatives
	dR1 matrix
	dH  matrix

	dhidden matrix
	dbias1  vector
	dA      matrix
	dI      matrix
	dinput  matrix
	dbias0  vector

	dR0         matrix
	dXS2        matrix
	dXS2ThatXS2 matrix
	dhatXS2     matrix
	dgamma1     vector
	dbeta1      vector

	dP       matrix
	dXS0     matrix
	dCV      matrix
	dproj    matrix
	dSV      []matrix
	dS       []matrix
	dQ       []matrix
	dK       []matrix
	dV       []matrix
	dQK      []matrix
	dqueries []matrix
	dkeys    []matrix
	dvalues  []matrix
	dXS1q    []matrix
	dXS1k    []matrix
	dXS1v    []matrix

	dXS1        matrix
	dhatXS1     matrix
	dXS1ThatXS1 matrix
	dgamma0     vector
	dbeta0      vector
}

func newModel(dModel, ctx, dAttn, attn, mlp, blocks int, vocab []rune) *model {
	dVocab := len(vocab)
	m := model{
		context: ctx,
		dModel:  dModel,
		dAttn:   dAttn,
		attn:    attn,
		mlp:     mlp,
	}
	m.tokens = makeMat(dVocab, dModel)
	m.positions = makeMat(ctx, dModel)
	m.XS = makeMat(ctx, dModel)

	m.blocks = make([]*block, blocks)
	for i := range blocks {
		m.blocks[i] = newBlock(dModel, ctx, dAttn, attn, mlp)
	}

	m.gamma2 = make(vector, dModel)
	m.beta2 = make(vector, dModel)
	m.unembed = makeMat(dModel, dVocab)
	m.bias2 = make(vector, dVocab)
	m.XS3 = makeMat(ctx, dModel)
	m.L = makeMat(ctx, dVocab)
	m.S = makeMat(ctx, dVocab)

	m.ys = make([]int, ctx)
	m.vocab = vocab

	m.dgamma2 = make(vector, dModel)
	m.dbeta2 = make(vector, dModel)

	m.dunembed = makeMat(dModel, dVocab)
	m.dbias2 = make(vector, dVocab)

	m.dtokens = makeMat(dVocab, dModel)
	m.dpositions = makeMat(ctx, dModel)

	m.dL = makeMat(ctx, dVocab)
	m.hatXS3 = makeMat(ctx, dModel)
	m.dhatXS3 = makeMat(ctx, dModel)
	m.dXS3 = makeMat(ctx, dModel)
	m.dXS3ThatXS3 = makeMat(ctx, dModel)

	return &m
}

func newBlock(dModel, ctx, dAttn, attn, mlp int) *block {
	b := block{
		dModel:  dModel,
		context: ctx,
		dAttn:   dAttn,
		attn:    attn,
	}

	b.XS0 = makeMat(ctx, dModel)

	b.gamma0 = make(vector, dModel)
	b.beta0 = make(vector, dModel)
	b.XS1 = makeMat(ctx, dModel)

	b.queries = make([]matrix, attn)
	b.keys = make([]matrix, attn)
	b.values = make([]matrix, attn)
	for i := range attn {
		b.queries[i] = makeMat(dModel, dAttn)
		b.keys[i] = makeMat(dModel, dAttn)
		b.values[i] = makeMat(dModel, dAttn)
	}
	b.proj = makeMat(attn*dAttn, dModel)

	b.gamma1 = make(vector, dModel)
	b.beta1 = make(vector, dModel)
	b.XS2 = makeMat(ctx, dModel)

	b.input = makeMat(dModel, mlp*dModel)
	b.bias0 = make(vector, mlp*dModel)
	b.hidden = makeMat(mlp*dModel, dModel)
	b.bias1 = make(vector, dModel)

	b.Q = make([]matrix, attn)
	b.K = make([]matrix, attn)
	b.V = make([]matrix, attn)
	b.QK = make([]matrix, attn)
	b.S = make([]matrix, attn)
	b.SV = make([]matrix, attn)
	for i := range attn {
		b.Q[i] = makeMat(ctx, dAttn)
		b.K[i] = makeMat(ctx, dAttn)
		b.V[i] = makeMat(ctx, dAttn)
		b.QK[i] = makeMat(ctx, ctx)
		b.S[i] = makeMat(ctx, ctx)
		b.SV[i] = makeMat(ctx, dAttn)
	}
	b.CV = makeMat(ctx, attn*dAttn)
	b.P = makeMat(ctx, dModel)
	b.R0 = makeMat(ctx, dModel)
	b.R1 = makeMat(ctx, dModel)
	b.I = makeMat(ctx, mlp*dModel)
	b.A = makeMat(ctx, mlp*dModel)
	b.H = makeMat(ctx, dModel)

	b.dR1 = makeMat(ctx, dModel)
	b.dH = makeMat(ctx, dModel)
	b.dhidden = makeMat(mlp*dModel, dModel)
	b.dbias1 = make(vector, dModel)
	b.dA = makeMat(ctx, mlp*dModel)
	b.dI = makeMat(ctx, mlp*dModel)
	b.dinput = makeMat(dModel, mlp*dModel)
	b.dbias0 = make(vector, mlp*dModel)

	b.dR0 = makeMat(ctx, dModel)
	b.dXS2 = makeMat(ctx, dModel)
	b.hatXS2 = makeMat(ctx, dModel)
	b.dXS2ThatXS2 = makeMat(ctx, dModel)
	b.dhatXS2 = makeMat(ctx, dModel)
	b.dgamma1 = make(vector, dModel)
	b.dbeta1 = make(vector, dModel)

	b.dP = makeMat(ctx, dModel)
	b.dXS0 = makeMat(ctx, dModel)
	b.dCV = makeMat(ctx, attn*dAttn)
	b.dproj = makeMat(attn*dAttn, dModel)
	b.dSV = make([]matrix, attn)
	b.dS = make([]matrix, attn)
	b.dQ = make([]matrix, attn)
	b.dK = make([]matrix, attn)
	b.dV = make([]matrix, attn)
	b.dQK = make([]matrix, attn)
	b.dqueries = make([]matrix, attn)
	b.dkeys = make([]matrix, attn)
	b.dvalues = make([]matrix, attn)
	b.dXS1q = make([]matrix, attn)
	b.dXS1k = make([]matrix, attn)
	b.dXS1v = make([]matrix, attn)
	for i := range attn {
		b.dSV[i] = makeMat(ctx, dAttn)
		b.dS[i] = makeMat(ctx, ctx)
		b.dQ[i] = makeMat(ctx, dAttn)
		b.dK[i] = makeMat(ctx, dAttn)
		b.dV[i] = makeMat(ctx, dAttn)
		b.dQK[i] = makeMat(ctx, ctx)
		b.dqueries[i] = makeMat(dModel, dAttn)
		b.dkeys[i] = makeMat(dModel, dAttn)
		b.dvalues[i] = makeMat(dModel, dAttn)
		b.dXS1q[i] = makeMat(ctx, dModel)
		b.dXS1k[i] = makeMat(ctx, dModel)
		b.dXS1v[i] = makeMat(ctx, dModel)
	}
	b.dXS1 = makeMat(ctx, dModel)
	b.hatXS1 = makeMat(ctx, dModel)
	b.dXS1ThatXS1 = makeMat(ctx, dModel)
	b.dhatXS1 = makeMat(ctx, dModel)
	b.dgamma0 = make(vector, dModel)
	b.dbeta0 = make(vector, dModel)

	return &b
}

func (m *model) forward() {
	xs := m.XS
	for _, b := range m.blocks {
		b.loadXs(xs)
		b.forward()
		xs = b.R1
	}
	layerNorm(m.XS3, m.hatXS3, xs, m.gamma2, m.beta2)
	mulMat(m.L, m.XS3, m.unembed)
	addMatV(m.L, m.bias2)
	softmax(m.S, m.L, false)
}

func (b *block) forward() {
	// attention
	layerNorm(b.XS1, b.hatXS1, b.XS0, b.gamma0, b.beta0)
	for i := range b.attn {
		mulMat(b.Q[i], b.XS1, b.queries[i])
		mulMat(b.K[i], b.XS1, b.keys[i])
		mulMat(b.V[i], b.XS1, b.values[i])
		mulMatT(b.QK[i], b.Q[i], b.K[i])
		d := 1 / math.Sqrt(float64(b.dAttn))
		mulMatK(b.QK[i], d)
		softmax(b.S[i], b.QK[i], true)
		mulMat(b.SV[i], b.S[i], b.V[i])
	}
	catMat(b.CV, b.SV)
	mulMat(b.P, b.CV, b.proj)
	addMatM(b.R0, b.XS0, b.P)
	// mlp
	layerNorm(b.XS2, b.hatXS2, b.R0, b.gamma1, b.beta1)
	mulMat(b.I, b.XS2, b.input)
	addMatV(b.I, b.bias0)
	mapMat(b.A, b.I, ReLU)
	mulMat(b.H, b.A, b.hidden)
	addMatV(b.H, b.bias1)
	addMatM(b.R1, b.R0, b.H)
}

func (m *model) loss() float64 {
	loss := 0.0
	count := 0
	s, r, c := unmat(m.S)
	for i := range r {
		if m.ys[i] == -1 {
			continue
		}
		count++
		p := s[i*c+m.ys[i]]
		loss += -math.Log(p)
	}
	return loss / float64(count)
}

func (m *model) size() int {
	blocks := 0
	for _, b := range m.blocks {
		blocks += b.size()
	}
	return blocks +
		len(m.tokens.RawMatrix().Data) +
		len(m.positions.RawMatrix().Data) +
		len(m.gamma2) + len(m.beta2) +
		len(m.unembed.RawMatrix().Data) + len(m.bias2)
}

func (b *block) size() int {
	return len(b.gamma0) + len(b.beta0) +
		len(b.gamma1) + len(b.beta1) +
		len(b.queries[0].RawMatrix().Data)*b.attn +
		len(b.keys[0].RawMatrix().Data)*b.attn +
		len(b.values[0].RawMatrix().Data)*b.attn +
		len(b.proj.RawMatrix().Data) +
		len(b.input.RawMatrix().Data) +
		len(b.bias0) +
		len(b.hidden.RawMatrix().Data) +
		len(b.bias1)
}

func (m *model) clone() *model {
	model := newModel(m.dModel, m.context, m.dAttn, m.attn, m.mlp, len(m.blocks), m.vocab)
	return model
}

func (m *model) loadXs(prompt []rune) {
	if len(prompt) > m.context {
		log.Fatal("too long xs")
	}
	m.XS.Zero()
	for _, b := range m.blocks {
		b.XS0.Zero()
		b.XS1.Zero()
		b.XS2.Zero()
	}
	dx, _, cx := unmat(m.XS)
	dt, _, ct := unmat(m.tokens)
	dp, _, cp := unmat(m.positions)
	for posIx := range prompt {
		vocIx := slices.Index(m.vocab, prompt[posIx])
		if vocIx == -1 {
			log.Panicf("loadXs: token %c is invalid", prompt[posIx])
		}
		for j := range m.dModel {
			dx[posIx*cx+j] = dt[vocIx*ct+j] + dp[posIx*cp+j]
		}
	}
	m.prompt = prompt
}

func (b *block) loadXs(xs matrix) {
	dxs, rxs, cxs := unmat(xs)
	dxs0, rxs0, cxs0 := unmat(b.XS0)
	if rxs != rxs0 || cxs != cxs0 {
		log.Panic("Incompatible XS")
	}
	copy(dxs0, dxs)
}

func (m *model) predict(ctx []rune) ([]rune, vector) {
	m.loadXs(ctx)
	m.forward()
	s, _, c := unmat(m.S)
	nexts := make([]rune, len(ctx))
	probs := make(vector, len(ctx))
	for tokIx := range len(ctx) {
		_, i := rowMax(s[c*tokIx : c*tokIx+c])
		nexts[tokIx] = m.vocab[i]
		probs[tokIx] = s[tokIx*c+i]
	}
	return nexts, probs
}

func (m *model) generate(ctx []rune, n int) {
	fmt.Printf("%s", string(ctx))
	d, _, c := unmat(m.S)
	for range n {
		m.loadXs(ctx)
		m.forward()
		tokIx := len(ctx) - 1
		i := sample(d[tokIx*c : tokIx*c+c])
		fmt.Printf("%c", m.vocab[i])
		ctx = append(ctx, m.vocab[i])
		ctx = ctx[max(0, len(ctx)-m.context):]
	}
	println()
}

func (m *model) peek(ctx []rune, i int, peekMode string) {
	m.loadXs(ctx)
	m.forward()
	switch peekMode {
	case "heatmap":
		m.printHeatmap([]int{i})
		fmt.Printf("Highlighted prompt [%s]\n", tokenHighlight(ctx, i))
		m.printNextTokenProbs(i)
	case "attention":
		m.printAttention()
	default:
		log.Panicf("Unrecognised peek mode '%s'", peekMode)
	}
}

func (m *model) rand(rng *rand.Rand) {
	mat := func(m matrix, std float64) {
		d, r, c := unmat(m)
		for i := range r {
			for j := range c {
				d[i*c+j] = rng.NormFloat64() * std
			}
		}
	}
	std := 1 / math.Sqrt(float64(m.dModel))
	stdmlp := 1 / math.Sqrt(float64(m.dModel*m.mlp))
	mat(m.tokens, math.Sqrt(0.5))
	mat(m.positions, math.Sqrt(0.5))
	for _, b := range m.blocks {
		for i := range b.gamma0 {
			b.gamma0[i] = 1
			b.beta0[i] = 0
			b.gamma1[i] = 1
			b.beta1[i] = 0
		}
		for i := range b.attn {
			mat(b.queries[i], std)
			mat(b.keys[i], std)
			mat(b.values[i], std)
		}
		mat(b.proj, std)
		mat(b.input, std)
		mat(b.hidden, stdmlp)
		for i := range b.bias0 {
			b.bias0[i] = 0
		}
		for i := range b.bias1 {
			b.bias1[i] = 0
		}
	}
	for i := range m.gamma2 {
		m.gamma2[i] = 1
		m.beta2[i] = 0
	}
	mat(m.unembed, std)
	for i := range m.bias2 {
		m.bias2[i] = 0
	}
}

func (m *model) apply(theta vector) {
	M := 0
	vec := func(v vector) {
		copy(v, theta[M:M+len(v)])
		M += len(v)
	}
	mat := func(m matrix) {
		d, _, _ := unmat(m)
		copy(d, theta[M:M+len(d)])
		M += len(d)
	}
	mat(m.tokens)
	mat(m.positions)
	for _, b := range m.blocks {
		vec(b.gamma0)
		vec(b.beta0)
		vec(b.gamma1)
		vec(b.beta1)
		for i := range b.attn {
			mat(b.queries[i])
			mat(b.keys[i])
			mat(b.values[i])
		}
		mat(b.proj)
		mat(b.input)
		vec(b.bias0)
		mat(b.hidden)
		vec(b.bias1)
	}
	vec(m.gamma2)
	vec(m.beta2)
	mat(m.unembed)
	vec(m.bias2)
	if M != len(theta) {
		log.Panic("apply: mismatch between len(theta) and model size")
	}
}

func (m *model) dump(theta vector) {
	M := 0
	vec := func(v vector) {
		copy(theta[M:M+len(v)], v)
		M += len(v)
	}
	mat := func(m matrix) {
		d, _, _ := unmat(m)
		copy(theta[M:M+len(d)], d)
		M += len(d)
	}
	mat(m.tokens)
	mat(m.positions)
	for _, b := range m.blocks {
		vec(b.gamma0)
		vec(b.beta0)
		vec(b.gamma1)
		vec(b.beta1)
		for i := range b.attn {
			mat(b.queries[i])
			mat(b.keys[i])
			mat(b.values[i])
		}
		mat(b.proj)
		mat(b.input)
		vec(b.bias0)
		mat(b.hidden)
		vec(b.bias1)
	}
	vec(m.gamma2)
	vec(m.beta2)
	mat(m.unembed)
	vec(m.bias2)
	if M != len(theta) {
		log.Panic("dump: mismatch between len(theta) and model size")
	}
}

func (m *model) grad(theta vector, k float64) {
	M := 0
	vec := func(v vector) {
		if k == 0 {
			copy(theta[M:M+len(v)], v)
		} else {
			for i := range v {
				theta[M+i] += k * v[i]
			}
		}
		M += len(v)
	}
	mat := func(m matrix) {
		d, _, _ := unmat(m)
		if k == 0 {
			copy(theta[M:M+len(d)], d)
		} else {
			for i := range d {
				theta[M+i] += k * d[i]
			}
		}
		M += len(d)
	}
	mat(m.dtokens)
	mat(m.dpositions)
	for _, b := range m.blocks {
		vec(b.dgamma0)
		vec(b.dbeta0)
		vec(b.dgamma1)
		vec(b.dbeta1)
		for i := range b.attn {
			mat(b.dqueries[i])
			mat(b.dkeys[i])
			mat(b.dvalues[i])
		}
		mat(b.dproj)
		mat(b.dinput)
		vec(b.dbias0)
		mat(b.dhidden)
		vec(b.dbias1)
	}
	vec(m.dgamma2)
	vec(m.dbeta2)
	mat(m.dunembed)
	vec(m.dbias2)
	if M != len(theta) {
		log.Panic("grad: mismatch between len(theta) and model size")
	}
}
