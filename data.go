package main

import (
	"fmt"
	"math/rand"
	"strings"
)

func generateCopyTask(vocab []rune, maxLen, n int) [][]rune {
	data := make([][]rune, 0, n)
	for range n {
		k := 1 + rand.Int()%maxLen
		datum := make([]rune, k)
		for i := range k {
			datum[i] = vocab[rand.Int()%len(vocab)]
		}
		str := string(datum)
		qs := strings.Repeat("?", len(str))
		data = append(data, []rune(fmt.Sprintf("%s|%s=%s", str, qs, str)))
	}
	return data
}
