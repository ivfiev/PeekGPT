package main

import (
	"log"
	"math/rand"
	"slices"
	"sync"
)

type training struct {
	t      *transformer
	t1, t2 *transformer
	data   []rune
	// ubatches []int
	ys []int
}

func newTraining(t *transformer) *training {
	return &training{
		t:    t,
		t1:   nil,
		t2:   nil,
		data: nil,
		// ubatches: nil,
		ys: nil,
	}
}

func (tr *training) train(data []rune, seed int64, iters int, lr, eps float64) *transformer {
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
	toks, pos := embeds(len(vocab), T.context, vocab)
	tr.data = data
	T.tok = toks
	T.pos = pos
	T.vocab = vocab
	theta := make(vector, T.size())
	rng := rand.New(rand.NewSource(seed))
	T.rand(rng)
	T.dump(theta)
	tr.ys = make([]int, len(data))
	for i := range tr.ys {
		ix := slices.Index(vocab, data[i])
		tr.ys[i] = ix
	}
	tr.t1 = T.clone()
	tr.t2 = T.clone()
	spsa(tr, theta, iters, lr, eps, rng)
	T.apply(theta)
	return T
}

func (tr *training) eval(t *transformer, theta vector) float64 {
	// if len(tr.ubatches) == 0 {
	// 	log.Panic("empty ubatches")
	// }
	t.apply(theta)
	loss := 0.0
	windows := len(tr.data) - t.context
	for w := range windows { // this assumes full-context training
		t.loadXs(tr.data[w : w+t.context])
		t.run()
		loss += t.loss(tr.ys[w+1 : w+t.context+1])
	}
	loss /= float64(windows)
	return loss
}

func (tr *training) eval2(u, v vector, i int) (float64, float64) {
	var wg sync.WaitGroup
	yu, yv := 0.0, 0.0
	wg.Go(func() {
		yu = tr.eval(tr.t1, u)
	})
	wg.Go(func() {
		yv = tr.eval(tr.t2, v)
	})
	wg.Wait()
	return yu, yv
}

func train(dVocab, context int, data []rune, iters int, lr, eps float64, seed int64) *transformer {
	t := newT(dVocab+context, dVocab, context, ReLU)
	tr := newTraining(t)
	tr.train(data, seed, iters, lr, eps)
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
