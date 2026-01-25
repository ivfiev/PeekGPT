package main

import (
	"fmt"
	"math/rand"
	"time"
)

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

func main() {
	toks := []rune("123")
	dVocab := 3
	context := 8
	tokens, positions := embeds(dVocab, context, toks)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	data := []rune("123123123123123123")
	t := newT(context, dVocab+context, dVocab)
	t.data = data
	t.voc = toks
	t.pos = positions
	t.tok = tokens
	theta := make(vector, t.size())
	for i := range len(theta) {
		theta[i] = scalar(rng.Float32() - 0.5)
	}
	spsa(spsaArgs{
		obj:      t,
		theta:    theta,
		iters:    10000,
		samples:  1,
		parallel: 1,
		lr:       0.1,
		eps:      0.0001,
		seed:     seed,
	})
	fmt.Printf("%v\n", len(theta))
	loss := t.eval(theta)
	println(loss)
	t.generate([]rune("12312312"), 10)

	// todos
	// print attn
	// run spsa for a few hundred, print stats, switch batch, run again
	// data := []rune("x|1,xx|2,xxx|3,xxxx|4,xxxxx|5,") train ctx "xxxx|(?)"
	// persist theta to file, or open a session
}
