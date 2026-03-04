package main

import (
	"fmt"
	"log"
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

type training struct {
	models []*model
	grad   vector

	mode       tmode
	training   [][]rune
	validation [][]rune

	iters     int
	ubatches  []int
	evalSteps int

	rng *rand.Rand
}

func newTraining(m *model, parallel int) *training {
	copies := make([]*model, parallel)
	for i := range copies {
		if i == 0 {
			copies[i] = m
		} else {
			copies[i] = m.clone()
		}
	}
	return &training{
		models: copies,
		grad:   make(vector, m.size()),
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

func (t *training) loadBatch() {
	if t.mode == task {
		for i := range t.ubatches {
			ix := t.rng.Int() % len(t.training)
			t.ubatches[i] = ix
		}
	} else {
		for i := range t.ubatches {
			ix := t.rng.Int() % (len(t.training[0]) - t.models[0].context - 1)
			t.ubatches[i] = ix
		}
	}
}

func (t *training) validate(m *model) float64 {
	loss := 0.0
	if t.mode == task {
		for _, data := range t.validation {
			loss += t.pointLoss(m, data)
		}
		loss /= float64(len(t.validation))
	} else {
		for i := range len(t.validation[0]) - m.context {
			loss += t.pointLoss(m, t.validation[0][i:i+m.context+1])
		}
		loss /= float64(len(t.validation[0]) - m.context)
	}
	return loss
}

func (t *training) pointLoss(m *model, data []rune) float64 {
	if t.mode == task {
		separator := slices.Index(data, '|')
		target := slices.Index(data, '=')
		if separator == -1 || target == -1 {
			log.Fatalf("bad pipe/eq indexes")
		}
		m.loadXs(data[:target])
		t.loadYs(m, data, 1+separator, 1+target, len(data)-target-1)
	} else {
		m.loadXs(data[:len(data)-1])
		t.loadYs(m, data, 0, 1, len(data)-1)
	}
	m.forward()
	return m.loss()
}

func (t *training) loadYs(m *model, data []rune, x, y, k int) {
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

func (t *training) eval(theta, grad vector, iter int) {
	t.loadBatch()
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
			if t.mode == task {
				data = t.training[u]
			} else {
				data = t.training[0][u : u+model.context+1] // 0..x, 1..y+1
			}
			// TODO tLoss race cond
			tLoss += t.pointLoss(model, data) / float64(len(t.ubatches))
			model.backward()
		})
		threads++
		if threads == len(t.models) || (i+1) == len(t.ubatches) {
			wg.Wait()
			for r := range threads {
				t.models[r].grad(t.grad)
				addVec2(grad, t.grad, 1/float64(len(t.ubatches)))
			}
			threads = 0
		}
	}
	if t.evalSteps > 0 && (iter%t.evalSteps == 0 || iter == t.iters) {
		vLoss := t.validate(t.models[0])
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
	t := newTraining(m, parallel)
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
	t.ubatches = make([]int, ubatches)
	adam(t, theta, iters, lr)
	m.apply(theta)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", m.size(), float64(time.Now().UnixMilli()-now)/1000)
	return t.models[0]
}
