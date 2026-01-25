package main

import (
	"fmt"
	"math/rand"
)

func main() {
	// // a := makeMat(2, 3)
	// // b := makeMat(2, 3)
	// // c := makeMat(2, 2)
	// // a[0][0] = 1
	// // a[0][2] = 2
	// // a[1][2] = 2
	// // b[0][0] = 2
	// // b[1][2] = 4
	// // mulMatT(c, a, b)
	// // printMat(a)
	// // printMat(b)
	// // printMat(c)
	// a := makeMat(3, 3)
	// a[0][0] = 1
	// a[0][1] = 2
	// a[0][2] = 3
	// a[1][0] = 2
	// a[1][1] = 3
	// a[1][2] = 1
	// a[2][0] = 3
	// a[2][1] = 1
	// a[2][2] = 2
	// b := makeMat(3, 3)
	// softmax(b, a)
	// printMat(b)
	// t := newT(3, 3, 4)
	// xs := matrix([][]scalar{{0, 1, 1}, {1, 0, 2}, {1, 1, 3}})
	// tr := training{t: t, xs: xs, ys: []int{1, 2, 0}}
	// theta := make(vector, tr.Size())
	// for i := range len(theta) {
	// 	theta[i] = scalar(rand.Float32() - 0.5)
	// }
	// spsa(&tr, theta, 2000, 1, 1, 0.005)
	// fmt.Printf("%v\n", theta)
	// loss := tr.Eval(theta)
	// fmt.Printf("%v\n", loss)
	// ys := t.run(xs)
	// printMat(ys)
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
	data := []rune("abcdefghabcdefgh")
	t := newT(8, 16, 8)
	tr := training{t: t, data: data, pos: positions, tok: tokens}
	theta := make(vector, tr.Size())
	for i := range len(theta) {
		theta[i] = scalar(rand.Float32() - 0.5)
	}
	spsa(&tr, theta, 10000, 1, 4, 0.1, 0.0001)
	fmt.Printf("%v\n", len(theta))
	loss := tr.Eval(theta)
	println(loss)
	tr.Predict([]rune("abcdefgh"))
	println()
	tr.Predict([]rune("bcdefgha"))
	println()
	tr.Predict([]rune("fghabcde"))
	tr.Predict([]rune("defg"))
}

func onehot(dim, ix int) vector {
	v := make(vector, dim)
	v[ix] = 1
	return v
}
