package main

import (
	"fmt"
	"math/rand"
	"slices"
	"strings"
)

func generateCopyTask(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		qs := strings.Repeat("?", len(str))
		dataset = append(dataset, []rune(fmt.Sprintf("%s|%s=%s", str, qs, str)))
	}
	return dataset
}

func generateReverseTask(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		slices.Reverse(data)
		rev := string(data)
		qs := strings.Repeat("?", len(str))
		dataset = append(dataset, []rune(fmt.Sprintf("%s|%s=%s", str, qs, rev)))
	}
	return dataset
}

func generateIndexTask(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		ix := rng.Int() % k
		ch := data[ix]
		dataset = append(dataset, []rune(fmt.Sprintf("%d%s|?=%c", ix, str, ch)))
	}
	return dataset
}

func generateSumTask(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		sum := 0
		for _, r := range data {
			sum += int(r - '0')
		}
		qs := strings.Repeat("?", len(fmt.Sprint(sum)))
		dataset = append(dataset, []rune(fmt.Sprintf("%s|%s=%d", str, qs, sum)))
	}
	return dataset
}
