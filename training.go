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
	t1, t2 *transformer
	data   [][]rune
	ys     [][]int
	rng    *rand.Rand
	iters  int
}

func newTraining(t *transformer) *training {
	return &training{
		t:    t,
		t1:   nil,
		t2:   nil,
		data: nil,
		ys:   nil,
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
	tr.data = data
	T.vocab = vocab
	theta := make(vector, T.size())
	rng := rand.New(rand.NewSource(seed))
	T.rand(rng)
	T.dump(theta)
	tr.ys = make([][]int, len(data))
	for i := range tr.ys {
		tr.ys[i] = make([]int, 0, len(data[i]))
		ix := slices.Index(data[i], '=')
		for j := 1 + ix; j < len(data[i]); j++ {
			vocIx := slices.Index(vocab, data[i][j])
			if vocIx == -1 {
				log.Panicf("train - bad token %c", data[i][j])
			}
			tr.ys[i] = append(tr.ys[i], vocIx)
		}
	}
	tr.t1 = T.clone()
	tr.t2 = T.clone()
	tr.rng = rng
	spsa(tr, theta, iters, lr, eps, rng)
	T.apply(theta)
	return T
}

func (tr *training) eval(t *transformer, theta vector) float64 {
	t.apply(theta)
	loss := 0.0
	for i, example := range tr.data {
		pipeIx := slices.Index(example, '|')
		eqIx := slices.Index(example, '=')
		if pipeIx == -1 || eqIx == -1 {
			log.Fatalf("bad pipe/ex indexes")
		}
		t.loadXs(example[:eqIx])
		t.run()
		loss += t.loss(tr.ys[i], pipeIx+1, eqIx)
	}
	loss /= float64(len(tr.data))
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
	if i%100 == 0 {
		fmt.Printf("\r              ")
		fmt.Printf("\r%.3f  %d%%", (yu+yv)/2, int(float64(i)/float64(tr.iters)*100))
	}
	return yu, yv
}

func train(dModel, dVocab, context int, data [][]rune, iters, ubatches, uiters int, lr, eps float64, seed int64) *transformer {
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
