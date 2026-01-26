package main

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestAbcdefgh(te *testing.T) {
	tokens := map[rune]vector{
		'a': onehot(16, 0),
		'b': onehot(16, 1),
		'c': onehot(16, 2),
		'd': onehot(16, 3),
		'e': onehot(16, 4),
		'f': onehot(16, 5),
		'g': onehot(16, 6),
		'h': onehot(16, 7),
	}
	positions := map[int]vector{
		0: onehot(16, 8),
		1: onehot(16, 9),
		2: onehot(16, 10),
		3: onehot(16, 11),
		4: onehot(16, 12),
		5: onehot(16, 13),
		6: onehot(16, 14),
		7: onehot(16, 15),
	}
	var seed int64 = 7357
	rng := rand.New(rand.NewSource(seed))
	data := []rune("abcdefghabcdefgh")
	t := newT(8, 16, 8)
	t.data = data
	t.pos = positions
	t.tok = tokens
	t.voc = []rune("abcdefgh")
	theta := make(vector, t.size())
	for i := range len(theta) {
		theta[i] = rng.Float64() - 0.5
	}
	spsa(t, theta, 10000, 0.1, 0.0001, seed)
	fmt.Printf("%v\n", len(theta))
	loss := t.eval(theta)
	fmt.Printf("%.70f\n", loss)
	if loss != 0.52463790839985147140822618894162587821483612060546875 {
		te.Error("loss has changed")
	} else {
		t.predict([]rune("defg"))
	}
}
