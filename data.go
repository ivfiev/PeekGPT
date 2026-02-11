package main

import (
	"fmt"
	"math/rand"
	"strings"
)

func generateCopyTask(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	data := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		datum := make([]rune, k)
		for i := range k {
			datum[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(datum)
		qs := strings.Repeat("?", len(str))
		data = append(data, []rune(fmt.Sprintf("%s|%s=%s", str, qs, str)))
	}
	return data
}
