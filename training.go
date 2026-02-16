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
	ts []*transformer // copies for parallelism

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

func newTraining(t *transformer) *training {
	ts := make([]*transformer, 32)
	ts[0] = t
	return &training{
		ts: ts,
		us: make(vector, 32),
		vs: make(vector, 32),
	}
}

func (tr *training) train(
	data, validation [][]rune,
	seed int64,
	spsaSamples, iters, ubatches, uiters int,
	lr, eps float64,
) *transformer {
	t := tr.ts[0]
	vocab := getVocab(data)
	if !slices.Equal(vocab, t.vocab) {
		log.Panicf("Incompatible vocabs: %s != %s\n", string(vocab), string(t.vocab))
	}
	tr.training = data
	tr.validation = validation
	t.vocab = vocab
	theta := make(vector, t.size())
	rng := rand.New(rand.NewSource(seed))
	t.rand(rng)
	t.dump(theta)
	tr.ubatches = make([]int, ubatches)
	for i := range spsaSamples * 2 {
		if tr.ts[i] == nil {
			tr.ts[i] = t.clone()
		}
	}
	tr.spsa = spsaSamples
	tr.rng = rng
	tr.uiters = uiters
	spsa(tr, theta, spsaSamples, iters, lr, eps, seed)
	t.apply(theta)
	return t
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

func (tr *training) loadBatch() {
	for i := range tr.ubatches {
		ix := tr.rng.Int() % len(tr.training)
		tr.ubatches[i] = ix
	}
}

func (tr *training) eval(t *transformer, theta vector) float64 {
	t.apply(theta)
	loss := 0.0
	for _, i := range tr.ubatches {
		data := tr.training[i]
		loss += tr.pointLoss(t, data)
	}
	return loss / float64(len(tr.ubatches))
}

func (tr *training) validate(t *transformer, theta vector) float64 {
	t.apply(theta)
	loss := 0.0
	for _, data := range tr.validation {
		loss += tr.pointLoss(t, data)
	}
	return loss / float64(len(tr.validation))
}

func (tr *training) pointLoss(t *transformer, data []rune) float64 {
	separator := slices.Index(data, '|')
	target := slices.Index(data, '=')
	if separator == -1 || target == -1 {
		log.Fatalf("bad pipe/ex indexes")
	}
	t.loadXs(data[:target])
	tr.loadYs(t, data, 1+separator, 1+target, len(data)-target-1)
	t.run()
	return t.loss()
}

func (tr *training) loadYs(t *transformer, data []rune, x, y, k int) {
	for i := range t.ys {
		t.ys[i] = -1
	}
	for i := range k {
		t.ys[x+i] = slices.Index(t.vocab, data[y+i])
		if t.ys[x+i] == -1 {
			fmt.Printf("%s - %s\n", string(data), string(t.vocab))
			log.Panicf("loadYs - bad token %c", data[y+i])
		}
	}
}

func (tr *training) eval2(us, vs []vector, i int) (vector, vector) {
	if i%tr.uiters == 0 {
		tr.loadBatch()
	}
	var wg sync.WaitGroup
	for i := range tr.spsa {
		wg.Go(func() {
			tr.us[i] = tr.eval(tr.ts[2*i], us[i])
		})
		wg.Go(func() {
			tr.vs[i] = tr.eval(tr.ts[2*i+1], vs[i])
		})
	}
	wg.Wait()
	if i%250 == 0 {
		w := us[0]
		if i%500 == 0 {
			w = vs[0]
		}
		loss := tr.validate(tr.ts[0], w)
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", loss, int(float64(i)/float64(tr.iters)*100))
	}
	return tr.us, tr.vs
}

func train(
	dModel, context, blocks int,
	data, validation [][]rune,
	spsaSamples, iters, ubatches, uiters int,
	lr, eps float64,
	seed int64,
) *transformer {
	t := newT(dModel, context, blocks, getVocab(data))
	tr := newTraining(t)
	tr.iters = iters
	now := time.Now().UnixMilli()
	tr.train(data, validation, seed, spsaSamples, iters, ubatches, uiters, lr, eps)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", t.size(), float64(time.Now().UnixMilli()-now)/1000)
	return tr.ts[0]
}
