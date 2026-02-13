package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func testMat(data [][]float64) matrix {
	A := makeMat(len(data), len(data[0]))
	for i := range data {
		for j := range data[i] {
			A.Set(i, j, data[i][j])
		}
	}
	return A
}

func assertEq(actual matrix, expected any, err string, te *testing.T) {
	switch expected := expected.(type) {
	case [][]float64:
		fail := func() {
			println("\nactual:")
			printMat(actual)
			println("expected:")
			printMat(testMat(expected))
			println()
			te.Fatal(err)
		}
		_, r, c, _ := unmat(actual)
		if r != len(expected) {
			fail()
		}
		for i := range expected {
			if c != len(expected[i]) {
				fail()
			}
			for j := range expected[i] {
				if math.Abs(actual.At(i, j)-expected[i][j]) > 0.001 {
					fail()
				}
			}
		}
	case matrix:
		fail := func() {
			println("\nactual:")
			printMat(actual)
			println("expected:")
			printMat(expected)
			println()
			te.Fatal(err)
		}
		da, ra, ca, _ := unmat(actual)
		de, re, ce, _ := unmat(expected)
		if ra != re || ca != ce || len(da) != len(de) {
			fail()
		}
		for i := range da {
			if math.Abs(da[i]-de[i]) > 0.001 {
				fail()
			}
		}
	default:
		te.Fatal("assertEq: bad types")
	}
}

type testquad struct {
	minimum vector
}

func (q *testquad) eval(theta vector) float64 {
	sum := 0.0
	for i := range theta {
		sum += (theta[i] - q.minimum[i]) * (theta[i] - q.minimum[i])
	}
	return sum
}

func (q *testquad) eval2(u, v vector, i int) (float64, float64) {
	return q.eval(u), q.eval(v)
}

func TestSPSA(te *testing.T) {
	const size = 100
	t := testquad{minimum: make(vector, size)}
	theta := make(vector, size)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	for i := range size {
		t.minimum[i] = rng.Float64() * 10
		theta[i] = -rng.Float64() * 10
	}
	spsa(&t, theta, 2000, 0.005, 0.0001, rng)
	loss := t.eval(theta)
	if loss > 0.001 {
		fmt.Printf("SPSA loss: %f, seed: %d\n", loss, seed)
		te.Fatal("SPSA failed to find minima")
	}
}

func TestIntegration(te *testing.T) {
	var seed int64 = 7357
	rng := rand.New(rand.NewSource(seed))
	t := train(16, 5, 2,
		generateCopyTask([]rune("123"), 2, 30, rng),
		generateCopyTask([]rune("123"), 2, 10, rng),
		5000, 8, 4, 0.001, 0.00001, seed)
	assert := func(expected string) {
		ctx := []rune(fmt.Sprintf("%s|%s", expected, strings.Repeat("?", len(expected))))
		toks, _ := t.predict(ctx)
		actual := string(toks[len(expected)+1 : len(expected)*2+1])
		if expected != actual {
			te.Fatalf("Integration %s != %s", expected, actual)
		}
	}
	assert("21")
	assert("32")
	assert("11")
	assert("22")
	assert("33")
	assert("13")
	assert("31")
	assert("3")
	assert("2")
	assert("1")
}

func TestLayerNorm(te *testing.T) {
	xs := testMat([][]float64{
		{1, 0, 0, 1},
		{0.5, -0.4, 0.7, 0},
		{5, 10, -15, 20},
	})
	ys := makeMat(3, 4)
	gamma := vector{1, 1.5, 2, 1}
	beta := vector{1, -1, -1, 1}
	layerNorm(ys, xs, gamma, beta)
	assertEq(ys, [][]float64{
		{2, -2.5, -3, 2},
		{1.697, -3.092, 1.325, 0.535},
		{1, -0.412, -4.138, 2.177},
	}, "LayerNorm", te)
}

func TestSoftmax(te *testing.T) {
	xs := testMat([][]float64{
		{1, 1, 1, 1},
		{1002, 1004, -7, 3},
		{-4, 1, -5, 2},
		{4, 2, -5, -1},
	})
	ys := makeMat(4, 4)
	softmaxT(ys, xs)
	assertEq(ys, [][]float64{
		{1, 0, 0, 0},
		{0.119, 0.881, 0, 0},
		{0.007, 0.991, 0.002, 0},
		{0.876, 0.118, 0, 0.006},
	}, "Softmax", te)
}

func TestLoadingXs(te *testing.T) {
	t := newT(4, 3, 1, []rune("123"))
	t.xs = testMat([][]float64{
		{1, 0, 0, -0.5},
		{0, 1, 0, 0.0},
		{0, 0, 1, 0.5},
	})
	t.run()
	assertEq(t.blocks[0].xs0, [][]float64{
		{1, 0, 0, -0.5},
		{0, 1, 0, 0.0},
		{0, 0, 1, 0.5},
	}, "xs0", te)
}

func TestBlockLayerNorm(te *testing.T) {
	b := newB(4, 3, ReLU)
	b.xs0 = testMat([][]float64{
		{1, 0, 0, 0},
		{0, 0, 0, 1},
		{0, 0, 0, 0},
	})
	b.gamma1 = vector{1.1, 1.2, 1.3, 1.4}
	b.beta1 = vector{0.5, 0.6, 0.7, -0.5}
	b.run()
	assertEq(b.xs1, [][]float64{
		{2.405, -0.093, -0.051, -1.308},
		{-0.135, -0.093, -0.051, 1.925},
		{0.000, 0.000, 0.000, 0.000},
	}, "xs1", te)
}

func TestBlockAttention(te *testing.T) {
	t := newT(4, 3, 2, []rune("abc"))
	t.rand(rand.New(rand.NewSource(7357)))
	t.xs = testMat([][]float64{
		{0.2, -0.34, 1.2, -0.5},
		{-0.6, 0.1, 0.2, 0.6},
		{2, 1, 0, -1},
	})
	mat34 := makeMat(3, 4)
	mat33 := makeMat(3, 3)
	t.run()

	assertEq(t.blocks[0].xs0, t.xs, "0.xs0", te)
	layerNorm(mat34, t.blocks[0].xs0, t.blocks[0].gamma1, t.blocks[0].beta1)
	assertEq(mat34, t.blocks[0].xs1, "0.xs1", te)

	mulMat(mat34, t.blocks[0].xs1, t.blocks[0].queries)
	assertEq(mat34, t.blocks[0].Q, "0.Q", te)

	mulMat(mat34, t.blocks[0].xs1, t.blocks[0].keys)
	assertEq(mat34, t.blocks[0].K, "0.K", te)

	mulMat(mat34, t.blocks[0].xs1, t.blocks[0].values)
	assertEq(mat34, t.blocks[0].V, "0.V", te)

	mulMatT(mat33, t.blocks[0].Q, t.blocks[0].K)
	mulMatK(mat33, 1/math.Sqrt(4))
	assertEq(mat33, t.blocks[0].QK, "0.QK", te)

	softmaxT(mat33, t.blocks[0].QK)
	assertEq(mat33, t.blocks[0].S, "0.S", te)

	mulMat(mat34, t.blocks[0].S, t.blocks[0].V)
	assertEq(mat34, t.blocks[0].SV, "0.SV", te)

	mat34.Zero()
	addMatM(mat34, t.blocks[0].xs0, t.blocks[0].SV)
	assertEq(mat34, t.blocks[0].R1, "0.R1", te)

	mat34.Zero()
	layerNorm(mat34, t.blocks[0].R1, t.blocks[0].gamma2, t.blocks[0].beta2)
	assertEq(mat34, t.blocks[0].xs2, "0.xs2", te)

	mulMat(mat34, t.blocks[0].xs2, t.blocks[0].input)
	assertEq(mat34, t.blocks[0].I, "0.I", te)

	mapMat(mat34, t.blocks[0].I, ReLU)
	assertEq(mat34, t.blocks[0].A, "0.A", te)

	mulMat(mat34, t.blocks[0].A, t.blocks[0].hidden)
	assertEq(mat34, t.blocks[0].H, "0.H", te)

	addMatM(mat34, t.blocks[0].H, t.blocks[0].R1)
	assertEq(mat34, t.blocks[0].R2, "0.R2", te)

	assertEq(t.blocks[1].xs0, t.blocks[0].R2, "1.xs0", te)

	mulMat(mat33, t.blocks[1].R2, t.linear)
	assertEq(t.L, mat33, "t.L", te)

	d, _, _, _ := unmat(t.L)
	for _, x := range d {
		if math.Abs(x) < 0.001 {
			te.Fatal("t.L = 0")
		}
	}
}

// func TestHeatmaps(te *testing.T) {
// 	t := newT(3, 3, 3, ReLU)
// 	t.xs = testMat([][]float64{
// 		{0, 0, 0},
// 		{0, 1, 0},
// 		{0, 0, 0},
// 	})
// 	t.V = testMat([][]float64{
// 		{0, 0, 0},
// 		{2, -1, 3},
// 		{0, 0, 0},
// 	})
// 	t.R1 = testMat([][]float64{
// 		{0, 0, 0},
// 		{-1, 1.5, 0.5},
// 		{0, 0, 0},
// 	})
// 	t.H = testMat([][]float64{
// 		{0, 0, 0},
// 		{-1, 4, 2},
// 		{0, 0, 0},
// 	})
// 	t.R2 = testMat([][]float64{
// 		{0, 0, 0},
// 		{1, -1, -0.5},
// 		{0, 0, 0},
// 	})
// 	t.linear = testMat([][]float64{
// 		{0, 1, 0},
// 		{0, 1, 0},
// 		{0, 1, 0},
// 	})
// 	t.printHeatmap(1, 1)
// 	// printMat(t.heatmap)
// 	assertEq(testMat(t.heatmap), [][]float64{
// 		{0.000, 0.200, 0.000},
// 		{0.400, -0.200, 0.600},
// 		{-0.200, 0.300, 0.100},
// 		{-0.200, 0.800, 0.400},
// 		{0.200, -0.200, -0.100},
// 	}, "heatmap", te)
// }

func TestLoss(te *testing.T) {
	t := newT(4, 4, 1, []rune("abcd"))
	t.L = testMat([][]float64{
		{1, 2, 3, -9},
		{2, 1.6, 1, 0.1},
		{5, -8, -13},
		{-2, 100, -1},
		{-3.14, 7.77, 0},
	})
	p := func(r, c int) float64 {
		d, _, cols, s := unmat(t.L)
		sum := 0.0
		for i := range cols {
			sum += math.Exp(d[r*s+i])
		}
		return -math.Log(math.Exp(float64(d[r*s+c])) / sum)
	}
	t.ys = []int{1, 0, 2, 0, 2}
	loss := t.loss()
	expected := (p(0, 1) + p(1, 0) + p(2, 2) + p(3, 0) + p(4, 2)) / 5.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
	t.ys = []int{-1, 1, 2, -1, -1}
	loss = t.loss()
	expected = (p(1, 1) + p(2, 2)) / 2.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
}

func TestLoadXs(te *testing.T) {
	t := newT(4, 3, 1, []rune("abc"))
	t.tokens = testMat([][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	})
	t.positions = testMat([][]float64{
		{0, 0, 0, -0.5},
		{0, -0.1, 0.1, 0},
		{0, 0, 0, 0.5},
	})
	t.loadXs([]rune("bac"))
	assertEq(t.xs, [][]float64{
		{0, 1, 0, -0.5},
		{1, -0.1, 0.1, 0},
		{0, 0, 1, 0.5},
	}, "LoadXS", te)
	t.loadXs([]rune("cb"))
	assertEq(t.xs, [][]float64{
		{0, 0, 1, -0.5},
		{0, 0.9, 0.1, 0},
		{0, 0, 0, 0},
	}, "LoadXS", te)
}

func TestMatrixInit(te *testing.T) {
	assert := func(X any, p func(float64) bool) {
		switch X := X.(type) {
		case matrix:
			_, rows, cols, _ := unmat(X)
			for i := range rows {
				for j := range cols {
					if !p(X.At(i, j)) {
						te.Fatalf("matrix property failed at position %d %d", i, j)
					}
				}
			}
		case vector:
			for i := range X {
				if !p(X[i]) {
					te.Fatalf("vector property failed at position %d", i)
				}
			}
		}
	}
	t := newT(4, 3, 2, []rune("abc"))
	assert(t.tokens, func(f float64) bool { return f == 0 })
	assert(t.xs, func(f float64) bool { return f == 0 })

	t.rand(rand.New(rand.NewSource(7357)))
	assert(t.tokens, func(f float64) bool { return f != 0 })
	assert(t.positions, func(f float64) bool { return f != 0 })
	for _, b := range t.blocks {
		assert(b.gamma1, func(f float64) bool { return f == 1 })
		assert(b.beta1, func(f float64) bool { return f == 0 })
		assert(b.keys, func(f float64) bool { return f != 0 })
		assert(b.queries, func(f float64) bool { return f != 0 })
		assert(b.values, func(f float64) bool { return f != 0 })
		assert(b.gamma2, func(f float64) bool { return f == 1 })
		assert(b.beta2, func(f float64) bool { return f == 0 })
		assert(b.input, func(f float64) bool { return f != 0 })
		assert(b.hidden, func(f float64) bool { return f != 0 })
	}
	assert(t.linear, func(f float64) bool { return f != 0 })
	assert(t.bias, func(f float64) bool { return f == 0 })

	theta := make(vector, t.size())
	for i := range theta {
		theta[i] = 7357
	}
	t.apply(theta)
	assert(t.tokens, func(f float64) bool { return f == 7357 })
	assert(t.positions, func(f float64) bool { return f == 7357 })
	for _, b := range t.blocks {
		assert(b.gamma1, func(f float64) bool { return f == 7357 })
		assert(b.beta1, func(f float64) bool { return f == 7357 })
		assert(b.keys, func(f float64) bool { return f == 7357 })
		assert(b.queries, func(f float64) bool { return f == 7357 })
		assert(b.values, func(f float64) bool { return f == 7357 })
		assert(b.gamma2, func(f float64) bool { return f == 7357 })
		assert(b.beta2, func(f float64) bool { return f == 7357 })
		assert(b.input, func(f float64) bool { return f == 7357 })
		assert(b.hidden, func(f float64) bool { return f == 7357 })
	}
	assert(t.linear, func(f float64) bool { return f == 7357 })
	assert(t.bias, func(f float64) bool { return f == 7357 })

	for i := range theta {
		theta[i] = 0
	}
	assert(theta, func(f float64) bool { return f == 0 })
	t.dump(theta)
	assert(theta, func(f float64) bool { return f == 7357 })
}
