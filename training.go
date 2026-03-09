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

type tmode int

const (
	task tmode = iota
	text
)

type trainer struct {
	models []*model
	theta0 vector

	mode       tmode
	training   [][]rune
	validation [][]rune

	iters     int
	ubatches  []int
	evalSteps int

	rng *rand.Rand
}

func newTrainer(m *model, parallel int) *trainer {
	copies := make([]*model, parallel)
	for i := range copies {
		if i == 0 {
			copies[i] = m
		} else {
			copies[i] = m.clone()
		}
	}
	return &trainer{
		models: copies,
		theta0: make(vector, m.size()),
	}
}

func getVocab(data [][]rune, mode tmode) []rune {
	vocab := make([]rune, 0)
	for _, str := range data {
		for _, tok := range str {
			if tok == '=' && mode == task {
				continue
			}
			if slices.Index(vocab, tok) == -1 {
				vocab = append(vocab, tok)
			}
		}
	}
	return vocab
}

func (t *trainer) loadBatch() {
	switch t.mode {
	case task:
		for i := range t.ubatches {
			ix := t.rng.Int() % len(t.training)
			t.ubatches[i] = ix
		}
	case text:
		for i := range t.ubatches {
			ix := t.rng.Int() % (len(t.training[0]) - t.models[0].context - 1)
			t.ubatches[i] = ix
		}
	default:
		log.Panic("invalid tmode")
	}
}

func (t *trainer) validate(m *model) float64 {
	loss := 0.0
	switch t.mode {
	case task:
		for _, data := range t.validation {
			loss += t.pointLoss(m, data)
		}
		loss /= float64(len(t.validation))
	case text:
		for i := range len(t.validation[0]) - m.context {
			loss += t.pointLoss(m, t.validation[0][i:i+m.context+1])
		}
		loss /= float64(len(t.validation[0]) - m.context)
	default:
		log.Panic("invalid tmode")
	}
	return loss
}

func (t *trainer) pointLoss(m *model, data []rune) float64 {
	switch t.mode {
	case task:
		separator := slices.Index(data, '|')
		target := slices.Index(data, '=')
		if separator == -1 || target == -1 {
			log.Fatalf("bad pipe/eq indexes")
		}
		m.loadXs(data[:target])
		t.loadYs(m, data, 1+separator, 1+target, len(data)-target-1)
	case text:
		m.loadXs(data[:len(data)-1])
		t.loadYs(m, data, 0, 1, len(data)-1)
	default:
		log.Panic("invalid tmode")
	}
	m.forward()
	return m.loss()
}

func (t *trainer) loadYs(m *model, data []rune, x, y, k int) {
	for i := range m.ys {
		m.ys[i] = -1
	}
	for i := range k {
		m.ys[x+i] = slices.Index(m.vocab, data[y+i])
		if m.ys[x+i] == -1 {
			fmt.Printf("%s - %s\n", string(data), string(m.vocab))
			log.Panicf("loadYs - bad token %c", data[y+i])
		}
	}
}

func (t *trainer) eval(theta, grad vector, iter int) {
	t.loadBatch()
	ubInv := 1 / float64(len(t.ubatches))
	tLoss := 0.0
	for _, m := range t.models {
		m.apply(theta)
	}
	var wg sync.WaitGroup
	threads := 0
	for i, u := range t.ubatches {
		model := t.models[threads]
		wg.Go(func() {
			var data []rune
			switch t.mode {
			case task:
				data = t.training[u]
			case text:
				data = t.training[0][u : u+model.context+1] // 0..x, 1..y+1
			default:
				log.Panic("invalid tmode")
			}
			// TODO tLoss race cond
			tLoss += t.pointLoss(model, data) * ubInv
			model.backward()
		})
		threads++
		if threads == len(t.models) || (i+1) == len(t.ubatches) {
			wg.Wait()
			for r := range threads {
				t.models[r].grad(grad, ubInv)
			}
			threads = 0
		}
	}
	if t.evalSteps > 0 && (iter%t.evalSteps == 0 || iter == t.iters) {
		vLoss := t.validate(t.models[0])
		fmt.Println("-")
		t.printBlockStats()
		fmt.Println("-")
		fmt.Printf("Iteration %d\n", iter)
		fmt.Printf("Training loss: %.3f\n", tLoss)
		fmt.Printf("Validation loss: %.3f\n", vLoss)
		fmt.Printf("%d%% done\n", int(float64(iter)/float64(t.iters)*100))
		if iter == t.iters {
			fmt.Println("-")
		}
	}
}

func (t *trainer) printBlockStats() {
	model0 := t.models[0]
	model1 := t.models[1]
	model0.apply(t.theta0)
	vec := func(w any) vector { // TODO mb move out
		switch w := w.(type) {
		case matrix:
			v, _, _ := unmat(w)
			return v
		case vector:
			return w
		}
		log.Panic("vec: bad input")
		return nil
	}
	stats := func(label string, x0, x1 any) {
		w0 := vec(x0)
		w1 := vec(x1)
		if len(w0) != len(w1) {
			log.Panicf("deltas: incompatible lengths %d != %d", len(w0), len(w1))
		}
		delta := 0.0
		for i := range w1 {
			delta += (w1[i] - w0[i]) * (w1[i] - w0[i]) // TODO mb move out
		}
		delta = math.Sqrt(delta / float64(len(w1)))
		u, o2 := meanStd(w1)
		fmt.Printf("%s: u[%.4f], o[%.4f], d[%.4f]\n", label, u, o2, delta)
	}
	stats("tokens   ", model0.tokens, model1.tokens)
	stats("positions", model0.positions, model1.positions)
	for bi := range model1.blocks {
		b1 := model1.blocks[bi]
		b0 := model0.blocks[bi]
		stats(fmt.Sprintf("blocks[%d].gamma0", bi), b0.gamma0, b1.gamma0)
		stats(fmt.Sprintf("blocks[%d].beta0 ", bi), b0.beta0, b1.beta0)
		println()
		for a := range b1.attn {
			stats(fmt.Sprintf("blocks[%d].queries[%d]", bi, a), b0.queries[a], b1.queries[a])
			stats(fmt.Sprintf("blocks[%d].keys[%d]   ", bi, a), b0.keys[a], b1.keys[a])
			stats(fmt.Sprintf("blocks[%d].values[%d] ", bi, a), b0.values[a], b1.values[a])
			println()
		}
		stats(fmt.Sprintf("blocks[%d].proj ", bi), b0.proj, b1.proj)
		println()
		stats(fmt.Sprintf("blocks[%d].gamma1", bi), b0.gamma1, b1.gamma1)
		stats(fmt.Sprintf("blocks[%d].beta1 ", bi), b0.beta1, b1.beta1)
		println()
		stats(fmt.Sprintf("blocks[%d].input ", bi), b0.input, b1.input)
		stats(fmt.Sprintf("blocks[%d].bias0 ", bi), b0.bias0, b1.bias0)
		stats(fmt.Sprintf("blocks[%d].hidden ", bi), b0.hidden, b1.hidden)
		stats(fmt.Sprintf("blocks[%d].bias1 ", bi), b0.bias1, b1.bias1)
		println()
	}
	println()
	stats("gamma2", model0.gamma2, model1.gamma2)
	stats("beta2", model0.beta2, model1.beta2)
	println()
	stats("unembed", model0.unembed, model1.unembed)
	stats("bias2", model0.bias2, model1.bias2)
}

func train(
	dModel, context, dAttn, attn, mlp, blocks int,
	data, validation [][]rune, evalSteps int,
	iters, ubatches, parallel int,
	lr float64,
	seed int64,
	checkpoint *model,
) *model {
	mode := task
	if len(data) == 1 {
		mode = text
	}
	vocab := getVocab(slices.Concat(data, validation), mode)
	m := checkpoint
	if checkpoint == nil {
		m = newModel(dModel, context, dAttn, attn, mlp, blocks, vocab)
	}
	t := newTrainer(m, parallel)
	t.mode = mode
	t.iters = iters
	t.evalSteps = evalSteps
	now := time.Now().UnixMilli()
	if !slices.Equal(vocab, m.vocab) {
		log.Panicf("Incompatible vocabs: %s != %s\n", string(vocab), string(m.vocab))
	}
	t.training = data
	t.validation = validation
	m.vocab = vocab
	theta := make(vector, m.size())
	rng := rand.New(rand.NewSource(seed))
	if checkpoint == nil {
		m.rand(rng)
	}
	t.rng = rng
	m.dump(theta)
	m.dump(t.theta0)
	t.ubatches = make([]int, ubatches)
	adam(t, theta, iters, lr)
	m.apply(theta)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", m.size(), float64(time.Now().UnixMilli()-now)/1000)
	return t.models[0]
}
