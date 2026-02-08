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
	t        *transformer
	t1, t2   *transformer
	data     []rune
	ubatches []int
	uiters   int
	ys       []int
	rng      *rand.Rand
	iters    int
}

func newTraining(t *transformer) *training {
	return &training{
		t:        t,
		t1:       nil,
		t2:       nil,
		data:     nil,
		ubatches: nil,
		ys:       nil,
	}
}

func (tr *training) train(data []rune, seed int64, iters, ubatches, uiters int, lr, eps float64) *transformer {
	T := tr.t
	vocab := make([]rune, 0, len(data))
	for _, tok := range data {
		if slices.Index(vocab, tok) == -1 {
			vocab = append(vocab, tok)
		}
	}
	if len(vocab) != T.dVocab {
		log.Panicf("incompatible vocab %d != %d\n", len(vocab), T.dVocab)
	}
	tr.data = data
	T.vocab = vocab
	theta := make(vector, T.size())
	rng := rand.New(rand.NewSource(seed))
	T.rand(rng)
	T.dump(theta)
	tr.ubatches = make([]int, ubatches)
	tr.uiters = uiters
	tr.ys = make([]int, len(data))
	for i := range tr.ys {
		ix := slices.Index(vocab, data[i])
		tr.ys[i] = ix
	}
	tr.t1 = T.clone()
	tr.t2 = T.clone()
	tr.rng = rng
	spsa(tr, theta, iters, lr, eps, rng)
	T.apply(theta)
	return T
}

func (tr *training) eval(t *transformer, theta vector) float64 {
	if len(tr.ubatches) == 0 {
		log.Panic("empty ubatches")
	}
	t.apply(theta)
	loss := 0.0
	for _, b := range tr.ubatches {
		t.loadXs(tr.data[b : b+t.context])
		t.run()
		loss += t.loss(tr.ys[1+b : 1+b+t.context])
	}
	loss /= float64(len(tr.ubatches))
	return loss
}

func (tr *training) eval2(u, v vector, i int) (float64, float64) {
	if i%tr.uiters == 0 {
		for i := range tr.ubatches {
			tr.ubatches[i] = tr.rng.Int() % (len(tr.data) - tr.t.context)
		}
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

func train(dModel, dVocab, context int, data []rune, iters, ubatches, uiters int, lr, eps float64, seed int64) *transformer {
	t := newT(dModel, dVocab, context, ReLU)
	tr := newTraining(t)
	tr.iters = iters
	now := time.Now().UnixMilli()
	tr.train(data, seed, iters, ubatches, uiters, lr, eps)
	fmt.Printf("\nTrained %d parameters in %.3f seconds.\n", t.size(), float64(time.Now().UnixMilli()-now)/1000)
	return tr.t
}

func embeds(vocab, ctx int, toks []rune) (map[rune]vector, map[int]vector) {
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
