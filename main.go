package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	toks := []rune("123")
	dVocab := 3
	context := 4
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
		theta[i] = rng.Float64() - 0.5
	}
	spsa(t, theta, 10000, 0.1, 0.0001, seed)
	fmt.Printf("%v\n", len(theta))
	loss := t.eval(theta)
	t.peek()
	println(loss)
	t.generate([]rune("1231"), 10)
	t.predict([]rune("1231"))

	// todos
	// print attn, detailed breakdown of last token
	// multiple models can be trained in parallel. quality evaluation. storage of weights?
	// run spsa for a few hundred, print stats, switch batch, run again
	// data := []rune("x|1,xx|2,xxx|3,xxxx|4,xxxxx|5,") train ctx "xxxx|(?)"
}
