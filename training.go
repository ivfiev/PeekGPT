package main

import (
	"fmt"
	"log"
	"math/rand"
	"slices"
	"sync"
	"time"
)

type training struct {
	models []*model // copies for parallelism

	training   [][]rune
	validation [][]rune

	iters    int
	ubatches []int
	uiters   int

	rng *rand.Rand

	spsa int
	us   vector
	vs   vector
}

func newTraining(m *model) *training {
	models := make([]*model, 32)
	models[0] = m
	return &training{
		models: models,
		us:     make(vector, 32),
		vs:     make(vector, 32),
	}
}

func (t *training) train(
	data, validation [][]rune,
	seed int64,
	spsaSamples, iters, ubatches, uiters int,
	lr, eps float64,
) *model {
	model := t.models[0]
	vocab := getVocab(data)
	if !slices.Equal(vocab, model.vocab) {
		log.Panicf("Incompatible vocabs: %s != %s\n", string(vocab), string(model.vocab))
	}
	t.training = data
	t.validation = validation
	model.vocab = vocab
	theta := make(vector, model.size())
	rng := rand.New(rand.NewSource(seed))
	model.rand(rng)
	model.dump(theta)
	t.ubatches = make([]int, ubatches)
	for i := range spsaSamples * 2 {
		if t.models[i] == nil {
			t.models[i] = model.clone()
		}
	}
	t.spsa = spsaSamples
	t.rng = rng
	t.uiters = uiters
	spsa(t, theta, spsaSamples, iters, lr, eps, seed)
	model.apply(theta)
	return model
}

func getVocab(data [][]rune) []rune {
	vocab := make([]rune, 0, len(data))
	for _, task := range data {
		for _, tok := range task {
			if tok == '=' {
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
	for i := range t.ubatches {
		ix := t.rng.Int() % len(t.training)
		t.ubatches[i] = ix
	}
}

func (t *training) eval(m *model, theta vector) float64 {
	m.apply(theta)
	loss := 0.0
	for _, i := range t.ubatches {
		data := t.training[i]
		loss += t.pointLoss(m, data)
	}
	return loss / float64(len(t.ubatches))
}

func (t *training) validate(m *model, theta vector) float64 {
	m.apply(theta)
	loss := 0.0
	for _, data := range t.validation {
		loss += t.pointLoss(m, data)
	}
	return loss / float64(len(t.validation))
}

func (t *training) pointLoss(m *model, data []rune) float64 {
	separator := slices.Index(data, '|')
	target := slices.Index(data, '=')
	if separator == -1 || target == -1 {
		log.Fatalf("bad pipe/eq indexes")
	}
	m.loadXs(data[:target])
	t.loadYs(m, data, 1+separator, 1+target, len(data)-target-1)
	m.run()
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

func (t *training) eval2(us, vs []vector, i int) (vector, vector) {
	if i%t.uiters == 0 {
		t.loadBatch()
	}
	var wg sync.WaitGroup
	for i := range t.spsa {
		wg.Go(func() {
			t.us[i] = t.eval(t.models[2*i], us[i])
		})
		wg.Go(func() {
			t.vs[i] = t.eval(t.models[2*i+1], vs[i])
		})
	}
	wg.Wait()
	if i%250 == 0 {
		w := us[0]
		if i%500 == 0 {
			w = vs[0]
		}
		loss := t.validate(t.models[0], w)
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", loss, int(float64(i)/float64(t.iters)*100))
	}
	return t.us, t.vs
}

func train(
	dModel, context, blocks int,
	data, validation [][]rune,
	spsaSamples, iters, ubatches, uiters int,
	lr, eps float64,
	seed int64,
) *model {
	m := newModel(dModel, context, blocks, getVocab(data))
	t := newTraining(m)
	t.iters = iters
	now := time.Now().UnixMilli()
	t.train(data, validation, seed, spsaSamples, iters, ubatches, uiters, lr, eps)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", m.size(), float64(time.Now().UnixMilli()-now)/1000)
	return t.models[0]
}
