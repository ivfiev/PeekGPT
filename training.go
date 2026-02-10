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
	t      *transformer
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
		t: t,
	}
}

func (tr *training) train(data [][]rune, seed int64, iters, ubatches, uiters int, lr, eps float64) *transformer {
	T := tr.t
	vocab := make([]rune, 0, len(data))
	for _, task := range data {
		for _, tok := range task {
			if tok == '=' {
				break
			}
			if slices.Index(vocab, tok) == -1 {
				vocab = append(vocab, tok)
			}
		}
	}
	if len(vocab) != T.dVocab {
		log.Panicf("incompatible vocab %d != %d\n", len(vocab), T.dVocab)
	}
	tr.training = data
	T.vocab = vocab
	theta := make(vector, T.size())
	rng := rand.New(rand.NewSource(seed))
	T.rand(rng)
	T.dump(theta)
	tr.ubatches = make([]int, ubatches)
	tr.t1 = T.clone()
	tr.t2 = T.clone()
	tr.rng = rng
	tr.uiters = uiters
	spsa(tr, theta, iters, lr, eps, rng)
	T.apply(theta)
	return T
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
		example := tr.training[i]
		pipeIx := slices.Index(example, '|')
		eqIx := slices.Index(example, '=')
		if pipeIx == -1 || eqIx == -1 {
			log.Fatalf("bad pipe/ex indexes")
		}
		t.loadXs(example[:eqIx])
		tr.loadYs(t, example, 1+pipeIx, 1+eqIx, len(example)-eqIx-1)
		t.run()
		loss += t.loss()
	}
	loss /= float64(len(tr.ubatches))
	return loss
}

func (tr *training) loadYs(t *transformer, datum []rune, x, y, k int) {
	for i := range tr.t.ys {
		t.ys[i] = -1
	}
	for i := range k {
		t.ys[x+i] = slices.Index(tr.t.vocab, datum[y+i])
		if tr.t.ys[x+i] == -1 {
			fmt.Printf("%s - %s\n", string(datum), string(t.vocab))
			log.Panicf("loadYs - bad token %c", datum[y+i])
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
	if i%100 == 0 {
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", (yu+yv)/2, int(float64(i)/float64(tr.iters)*100))
	}
	return yu, yv
}

func train(dModel, dVocab, context int, data, validation [][]rune, iters, ubatches, uiters int, lr, eps float64, seed int64) *transformer {
	t := newT(dModel, dVocab, context, ReLU)
	tr := newTraining(t)
	tr.iters = iters
	now := time.Now().UnixMilli()
	tr.train(data, seed, iters, ubatches, uiters, lr, eps)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", t.size(), float64(time.Now().UnixMilli()-now)/1000)
	return tr.t
}

func onehotEmbeds(vocab, ctx int, toks []rune) (map[rune]vector, map[int]vector) {
	dModel := vocab + ctx
	tokens := map[rune]vector{}
	pos := map[int]vector{}
	for i := range vocab {
		tokens[toks[i]] = onehot(dModel, i)
	}
	for i := range ctx {
		pos[i] = onehot(dModel, vocab+i)
	}
	return tokens, pos
}
