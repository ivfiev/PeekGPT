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
	t1, t2 *transformer // copies for parallelism

	training   [][]rune
	validation [][]rune

	iters    int
	ubatches []int
	uiters   int

	rng *rand.Rand
}

func newTraining(t *transformer) *training {
	return &training{
		t1: t,
	}
}

func (tr *training) train(
	data, validation [][]rune,
	seed int64,
	iters, ubatches, uiters int,
	lr, eps float64,
) *transformer {
	t := tr.t1
	vocab := getVocab(data)
	if len(vocab) != t.dVocab {
		log.Panicf("incompatible vocab %d != %d\n", len(vocab), t.dVocab)
	}
	tr.training = data
	tr.validation = validation
	t.vocab = vocab
	theta := make(vector, t.size())
	rng := rand.New(rand.NewSource(seed))
	t.rand(rng)
	t.dump(theta)
	tr.ubatches = make([]int, ubatches)
	tr.t2 = t.clone()
	tr.rng = rng
	tr.uiters = uiters
	spsa(tr, theta, iters, lr, eps, rng)
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

func (tr *training) eval2(u, v vector, i int) (float64, float64) {
	if i%tr.uiters == 0 {
		tr.loadBatch()
	}
	var wg sync.WaitGroup
	yu, yv := 0.0, 0.0
	wg.Go(func() {
		yu = tr.eval(tr.t1, u)
	})
	wg.Go(func() {
		yv = tr.eval(tr.t2, v)
	})
	wg.Wait()
	if i%250 == 0 {
		w := u
		if i%500 == 0 {
			w = v
		}
		loss := tr.validate(tr.t1, w)
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", loss, int(float64(i)/float64(tr.iters)*100))
	}
	return yu, yv
}

func train(
	dModel, context, blocks int,
	data, validation [][]rune,
	iters, ubatches, uiters int,
	lr, eps float64,
	seed int64,
) *transformer {
	t := newT(dModel, context, blocks, getVocab(data))
	tr := newTraining(t)
	tr.iters = iters
	now := time.Now().UnixMilli()
	tr.train(data, validation, seed, iters, ubatches, uiters, lr, eps)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", t.size(), float64(time.Now().UnixMilli()-now)/1000)
	return tr.t1
}
