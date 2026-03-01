package main

import (
	"fmt"
	"log"
	"math/rand"
	"slices"
	"time"
)

type tmode int

const (
	task tmode = iota
	text
)

type training struct {
	model *model
	grad  vector

	mode       tmode
	training   [][]rune
	validation [][]rune

	iters    int
	ubatches []int

	rng *rand.Rand
}

func newTraining(m *model) *training {
	models := make([]*model, 64)
	models[0] = m
	return &training{
		model: m,
		grad:  make(vector, m.size()),
	}
}

func (t *training) train(
	data, validation [][]rune,
	iters, ubatches int,
	lr float64,
	seed int64,
) *model {
	m := t.model
	vocab := getVocab(data, t.mode)
	if !slices.Equal(vocab, m.vocab) {
		log.Panicf("Incompatible vocabs: %s != %s\n", string(vocab), string(m.vocab))
	}
	t.training = data
	t.validation = validation
	m.vocab = vocab
	theta := make(vector, m.size())
	rng := rand.New(rand.NewSource(seed))
	m.rand(rng)
	m.dump(theta)
	t.ubatches = make([]int, ubatches)
	t.rng = rng
	adam(t, theta, iters, lr)
	m.apply(theta)
	return m
}

func getVocab(data [][]rune, mode tmode) []rune {
	vocab := make([]rune, 0, len(data))
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
			ix := t.rng.Int() % (len(t.training[0]) - t.model.context - 1)
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

func (t *training) eval(theta, grad vector, i int) {
	t.loadBatch()
	m := t.model
	m.apply(theta)
	for _, i := range t.ubatches {
		var data []rune
		if t.mode == task {
			data = t.training[i]
		} else {
			data = t.training[0][i : i+m.context+1] // 0..x, 1..y+1
		}
		t.pointLoss(m, data)
		m.backward()
		m.grad(t.grad)
		addVec2(grad, t.grad, 1/float64(len(t.ubatches)))
	}
	if i%100 == 0 {
		loss := t.validate(m)
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", loss, int(float64(i)/float64(t.iters)*100))
	}
}

func train(
	dModel, context, dAttn, attn, blocks int,
	data, validation [][]rune,
	iters, ubatches int,
	lr float64,
	seed int64,
) *model {
	mode := task
	if len(data) == 1 {
		mode = text
	}
	m := newModel(dModel, context, dAttn, attn, blocks, getVocab(data, mode))
	t := newTraining(m)
	t.mode = mode
	t.iters = iters
	now := time.Now().UnixMilli()
	t.train(data, validation, iters, ubatches, lr, seed)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", m.size(), float64(time.Now().UnixMilli()-now)/1000)
	return t.model
}
