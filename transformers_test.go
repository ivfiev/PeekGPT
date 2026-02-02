package main

import (
	"fmt"
	"testing"
)

func TestAbcdefgh(te *testing.T) {
	var seed int64 = 7357
	t := trainModel(10, []rune("abcdefgghabcdefggh"), 4, seed, 200000, 0.0075, 0.0001)
	theta := make(vector, t.size())
	t.dump(theta)
	loss := t.eval(theta)
	fmt.Printf("%5.80f\n", loss)
	// 0.1262216915046103782316322394763119518756866455078125
	// 0.12622169137027217100666121041285805404186248779296875
	if loss != 0.12622169137027217100666121041285805404186248779296875 {
		te.Fatal("Loss has changed")
	}
	next, prob := t.predict([]rune("bcdefg"))
	if next != 'g' || prob < 0.99 {
		te.Fatalf("Bad prediction %c, %.4f", next, prob)
	}
	next, prob = t.predict([]rune("bcdefgg"))
	if next != 'h' || prob < 0.99 {
		te.Fatalf("Bad prediction %c, %.4f", next, prob)
	}
}
