package main

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
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

func assertEq(actual matrix, expected [][]float64, err string, te *testing.T) {
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
	t := train(16, 4, 5, [][]rune{
		[]rune("ab|??=ab"),
		[]rune("ba|??=ba"),
		[]rune("ab|??=ab"),
		[]rune("ba|??=ba"),
		[]rune("b|?=b"),
		[]rune("a|?=a"),
		[]rune("ba|??=ba"),
		[]rune("a|?=a"),
		[]rune("aa|??=aa"),
		[]rune("bb|??=bb"),
		[]rune("ba|??=ba"),
		[]rune("b|?=b"),
		[]rune("aa|??=aa"),
		[]rune("bb|??=bb"),
	}, [][]rune{
		[]rune("a|?=a"),
		[]rune("b|?=b"),
		[]rune("ab|??=ab"),
		[]rune("ba|??=ba"),
		[]rune("aa|??=aa"),
		[]rune("bb|??=bb"),
	}, 10000, 8, 4, 0.001, 0.00001, seed)
	toks, _ := t.predict([]rune("ab|??"))
	if !slices.Equal(toks[3:5], []rune("ab")) {
		te.Fatalf("Integration ab != %s", string(toks[3:5]))
	}
	toks, _ = t.predict([]rune("ba|??"))
	if !slices.Equal(toks[3:5], []rune("ba")) {
		te.Fatalf("Integration ba != %s", string(toks[3:5]))
	}
	toks, _ = t.predict([]rune("a|?"))
	if !slices.Equal(toks[2:3], []rune("a")) {
		te.Fatalf("Integration ba != %s", string(toks[2:4]))
	}
	toks, _ = t.predict([]rune("b|?"))
	if !slices.Equal(toks[2:3], []rune("b")) {
		te.Fatalf("Integration ba != %s", string(toks[2:4]))
	}
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

func TestRun(te *testing.T) {
	t := newT(5, 3, 4, ReLU)
	t.xs = testMat([][]float64{
		{1, 0, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 0, 1, 1},
		{0, 0, 0, 0, 0},
	})
	t.gamma1 = vector{1, 1, 1, 1, 1}
	t.beta1 = vector{0, 0, 0, 0, 0}
	t.gamma2 = vector{1, 1, 1, 1, 1}
	t.beta2 = vector{0, 0, 0, 0, 0}
	t.queries = testMat([][]float64{
		{0.5, -0.5, -1, 2.5, 1},
		{0, 1, -1, 0.5, 1},
		{1, 0, -1, 0.5, 1},
		{-1, 0.5, -1, 0.5, 1},
		{0, 0.5, 1, -0.5, 0},
	})
	t.keys = testMat([][]float64{
		{1, 0.5, 1, 0, 1},
		{0, 0.5, 1, -0.5, 0},
		{0.5, -1, -1, 0.5, 1},
		{-1, -1, -1, 2, 1},
		{1, 0, -2, 0.5, 0},
	})
	t.values = testMat([][]float64{
		{2, 0, -0.5, 1, -0.5},
		{-1, 2, -1, 1, 0.5},
		{-1, 0, 0.5, 1, 0.5},
		{0.5, -2, 0.5, 1, -0.5},
		{0.5, -2, 0.5, 1, 0},
	})
	t.input = testMat([][]float64{
		{-0.5, 1, 0.5, 2, -2},
		{0.5, 2, -2, 2, 1},
		{0.5, -1, 2, 1, 1},
		{-1, 1, 1, 1, 0},
		{0.1, 0, 2, 0, 3.5},
	})
	t.hidden = testMat([][]float64{
		{-0.5, -1, 0.5, 2, -2},
		{2, 2, 2, -2, 1.5},
		{0.5, 1.2, 2, 1, 1},
		{-1, 0, 1, 1, 0},
		{0.1, 0.5, 1, 0, 3.5},
	})
	t.linear = testMat([][]float64{
		{0.5, -0.5, 3},
		{1, 1, 1},
		{3.1, -0.9, 0},
		{0.04, 0, 1},
		{0.7, -2, 1},
	})
	t.bias = vector{0.5, -0.5, 0.3}
	t.run()
	assertEq(t.xs1, [][]float64{
		{1.225, -0.816, -0.816, 1.225, -0.816},
		{-0.816, 1.225, -0.816, 1.225, -0.816},
		{-0.816, -0.816, -0.816, 1.225, 1.225},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "xs1", te)
	assertEq(t.Q, [][]float64{
		{-1.429, -1.225, -1.633, 3.266, 0.816},
		{-2.449, 1.837, -1.633, -0.816, 0.816},
		{-2.449, 0.816, 2.449, -2.858, -1.225},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "Q", te)
	assertEq(t.K, [][]float64{
		{-1.225, -0.204, 1.633, 2.041, 1.633},
		{-3.266, -0.204, 1.633, 1.021, -0.408},
		{-1.225, -1.225, -4.491, 3.062, -0.408},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "K", te)
	assertEq(t.QK, [][]float64{
		{3.279, 2.348, 9.056, 0.000},
		{-0.168, 1.696, 2.348, 0.000},
		{-0.447, 4.211, -7.714, 0.000},
		{0.000, 0.000, 0.000, 0.000},
	}, "K", te)
	assertEq(t.QK, [][]float64{
		{3.279, 2.348, 9.056, 0.000},
		{-0.168, 1.696, 2.348, 0.000},
		{-0.447, 4.211, -7.714, 0.000},
		{0.000, 0.000, 0.000, 0.000},
	}, "K", te)
	assertEq(t.S, [][]float64{
		{1.000, 0.000, 0.000, 0.000},
		{0.134, 0.866, 0.000, 0.000},
		{0.009, 0.991, 0.000, 0.000},
		{0.000, 0.000, 0.000, 0.000},
	}, "K", te)
	assertEq(t.V, [][]float64{
		{4.287, -2.449, 0.000, 0.000, -2.041},
		{-1.837, 1.633, -1.021, 0.000, 0.000},
		{1.225, -6.532, 2.041, 0.000, -1.021},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "V", te)
	assertEq(t.R1, [][]float64{
		{5.287, -2.449, 0.000, 1.000, -2.041},
		{-1.015, 2.085, -0.884, 1.000, -0.274},
		{-1.780, 1.595, -1.011, 1.000, 0.981},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "R1", te)
	assertEq(t.I, [][]float64{
		{0.227, -1.281, 1.007, -2.687, -3.110},
		{3.799, 5.482, -3.594, 2.398, -3.239},
		{1.421, 5.158, -2.349, 2.329, 0.270},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "I", te)
	assertEq(t.A, [][]float64{
		{0.227, 0.000, 1.007, 0.000, 0.000},
		{3.799, 5.482, 0.000, 2.398, 0.000},
		{1.421, 5.158, 0.000, 2.329, 0.270},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "I", te)
	assertEq(t.H, [][]float64{
		{0.390, 2.470, 2.129, 0.780, 1.030},
		{-2.585, 13.766, 10.876, -1.401, 3.121},
		{-1.751, 8.905, 9.499, 0.908, 3.667},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "H", te)
	assertEq(t.R2, [][]float64{
		{5.677, 0.020, 2.129, 1.780, -1.011},
		{-3.600, 15.851, 9.993, -0.401, 2.847},
		{-3.530, 10.499, 8.488, 1.908, 4.648},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "R2", te)
	assertEq(t.L, [][]float64{
		{9.320, -3.212, 18.119},
		{47.505, 2.463, 7.798},
		{38.877, -5.170, 6.764},
		{0.500, -0.500, 0.300},
	}, "L", te)
}

func TestHeatmaps(te *testing.T) {
	t := newT(3, 3, 3, ReLU)
	t.xs = testMat([][]float64{
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 0},
	})
	t.V = testMat([][]float64{
		{0, 0, 0},
		{2, -1, 3},
		{0, 0, 0},
	})
	t.R1 = testMat([][]float64{
		{0, 0, 0},
		{-1, 1.5, 0.5},
		{0, 0, 0},
	})
	t.H = testMat([][]float64{
		{0, 0, 0},
		{-1, 4, 2},
		{0, 0, 0},
	})
	t.R2 = testMat([][]float64{
		{0, 0, 0},
		{1, -1, -0.5},
		{0, 0, 0},
	})
	t.linear = testMat([][]float64{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	})
	t.printHeatmap(1, 1)
	// printMat(t.heatmap)
	assertEq(testMat(t.heatmap), [][]float64{
		{0.000, 0.200, 0.000},
		{0.400, -0.200, 0.600},
		{-0.200, 0.300, 0.100},
		{-0.200, 0.800, 0.400},
		{0.200, -0.200, -0.100},
	}, "heatmap", te)
}

func TestLoss(te *testing.T) {
	t := newT(4, 4, 5, nil)
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
	t := newT(4, 3, 3, nil)
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
	t.vocab = []rune("abc")
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
	t := newT(4, 3, 5, ReLU)
	assert(t.tokens, func(f float64) bool { return f == 0 })
	assert(t.xs, func(f float64) bool { return f == 0 })

	t.rand(rand.New(rand.NewSource(7357)))
	assert(t.tokens, func(f float64) bool { return f != 0 })
	assert(t.positions, func(f float64) bool { return f != 0 })
	assert(t.gamma1, func(f float64) bool { return f == 1 })
	assert(t.beta1, func(f float64) bool { return f == 0 })
	assert(t.keys, func(f float64) bool { return f != 0 })
	assert(t.queries, func(f float64) bool { return f != 0 })
	assert(t.values, func(f float64) bool { return f != 0 })
	assert(t.gamma2, func(f float64) bool { return f == 1 })
	assert(t.beta2, func(f float64) bool { return f == 0 })
	assert(t.input, func(f float64) bool { return f != 0 })
	assert(t.hidden, func(f float64) bool { return f != 0 })
	assert(t.linear, func(f float64) bool { return f != 0 })
	assert(t.bias, func(f float64) bool { return f == 0 })

	theta := make(vector, t.size())
	for i := range theta {
		theta[i] = 7357
	}
	t.apply(theta)
	assert(t.tokens, func(f float64) bool { return f == 7357 })
	assert(t.positions, func(f float64) bool { return f == 7357 })
	assert(t.gamma1, func(f float64) bool { return f == 7357 })
	assert(t.beta1, func(f float64) bool { return f == 7357 })
	assert(t.keys, func(f float64) bool { return f == 7357 })
	assert(t.queries, func(f float64) bool { return f == 7357 })
	assert(t.values, func(f float64) bool { return f == 7357 })
	assert(t.gamma2, func(f float64) bool { return f == 7357 })
	assert(t.beta2, func(f float64) bool { return f == 7357 })
	assert(t.input, func(f float64) bool { return f == 7357 })
	assert(t.hidden, func(f float64) bool { return f == 7357 })
	assert(t.linear, func(f float64) bool { return f == 7357 })
	assert(t.bias, func(f float64) bool { return f == 7357 })

	for i := range theta {
		theta[i] = 0
	}
	assert(theta, func(f float64) bool { return f == 0 })
	t.dump(theta)
	assert(theta, func(f float64) bool { return f == 7357 })
}
