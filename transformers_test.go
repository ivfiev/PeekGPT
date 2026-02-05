package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func testMat(data matrix) matrix {
	A := makeMat(len(data), len(data[0]))
	for i := range data {
		copy(A[i], data[i])
	}
	return A
}

func assertEq(actual, expected matrix, err string, te *testing.T) {
	fail := func() {
		println("\nactual:")
		printMat(actual)
		println("expected:")
		printMat(expected)
		println()
		te.Fatal(err)
	}
	if len(actual) != len(expected) {
		fail()
	}
	for i := range actual {
		if len(actual[i]) != len(expected[i]) {
			fail()
		}
		for j := range actual[i] {
			if math.Abs(actual[i][j]-expected[i][j]) > 0.001 {
				fail()
			}
		}
	}
}

type testquad struct {
	minimum vector
}

func (q *testquad) clone() objective {
	return q
}

func (q *testquad) eval(theta vector) float64 {
	sum := 0.0
	for i := range theta {
		sum += (theta[i] - q.minimum[i]) * (theta[i] - q.minimum[i])
	}
	return sum
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

func TestAbcdefggh(te *testing.T) {
	var seed int64 = 7357
	t := trainModel(8, []rune("aa|bb|aa|bb|aa|bb|"), 4, seed, 5000, 0.025, 0.0001)
	theta := make(vector, t.size())
	t.dump(theta)
	loss := t.eval(theta)
	targetLoss := 0.083747851122179362004516178785706870257854461669921875000000
	if loss != targetLoss {
		fmt.Printf("%.60f\n", targetLoss)
		fmt.Printf("%.60f\n", loss)
		te.Fatal("Loss has changed")
	}
}

func TestLayerNorm(te *testing.T) {
	xs := testMat(matrix{
		{1, 0, 0, 1},
		{0.5, -0.4, 0.7, 0},
		{5, 10, -15, 20},
	})
	ys := makeMat(len(xs), len(xs[0]))
	gamma := vector{1, 1.5, 2, 1}
	beta := vector{1, -1, -1, 1}
	layerNorm(ys, xs, gamma, beta)
	assertEq(ys, matrix{
		{2, -2.5, -3, 2},
		{1.697, -3.092, 1.325, 0.535},
		{1, -0.412, -4.138, 2.177},
	}, "LayerNorm", te)
}

func TestSoftmax(te *testing.T) {
	xs := testMat(matrix{
		{1, 1, 1, 1},
		{1002, 1004, -7, 3},
		{-4, 1, -5, 2},
		{4, 2, -5, -1},
	})
	ys := makeMat(len(xs), len(xs[0]))
	softmax(ys, xs)
	assertEq(ys, matrix{
		{1, 0, 0, 0},
		{0.119, 0.881, 0, 0},
		{0.007, 0.991, 0.002, 0},
		{0.876, 0.118, 0, 0.006},
	}, "Softmax", te)
}

func TestRun(te *testing.T) {
	t := newT(4, 5, 3, nil, nil, nil, nil, ReLU)
	t.handicap = 0.5
	t.xs = testMat(matrix{
		{1, 0, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 0, 1, 1},
		{0, 0, 0, 0, 0},
	})
	t.gamma1 = vector{1, 1, 1, 1, 1}
	t.beta1 = vector{0, 0, 0, 0, 0}
	t.gamma2 = vector{1, 1, 1, 1, 1}
	t.beta2 = vector{0, 0, 0, 0, 0}
	t.queries = testMat(matrix{
		{0.5, -0.5, -1, 2.5, 1},
		{0, 1, -1, 0.5, 1},
		{1, 0, -1, 0.5, 1},
		{-1, 0.5, -1, 0.5, 1},
	})
	t.keys = testMat(matrix{
		{1, 0.5, 1, 0, 1},
		{0.5, -1, -1, 0.5, 1},
		{-1, -1, -1, 2, 1},
		{1, 0, -2, 0.5, 0},
	})
	t.values = testMat(matrix{
		{2, 0, -0.5, 1},
		{-1, 2, -1, 1},
		{-1, 0, 0.5, 1},
		{0.5, -2, 0.5, 1},
		{0.5, -2, 0.5, 1},
	})
	t.input = testMat(matrix{
		{-0.5, 1, 0.5, 2, -2},
		{0.5, 2, -2, 2, 1},
		{0.5, -1, 2, 1, 1},
		{-1, 1, 1, 1, 0},
		{0.1, 0, 2, 0, 3.5},
	})
	t.hidden = testMat(matrix{
		{-0.5, -1, 0.5, 2, -2},
		{2, 2, 2, -2, 1.5},
		{0.5, 1.2, 2, 1, 1},
		{-1, 0, 1, 1, 0},
		{0.1, 0.5, 1, 0, 3.5},
	})
	t.linear = testMat(matrix{
		{0.5, -0.5, 3},
		{1, 1, 1},
		{3.1, -0.9, 0},
		{0.04, 0, 1},
		{0.7, -2, 1},
	})
	t.bias = vector{0.5, -0.5, 0.3}
	t.run()
	assertEq(t.Q, matrix{
		{4.082, -0.204, 1.837, -1.021},
		{2.041, 1.837, -0.204, 2.041},
		{5.103, 1.837, 1.837, 3.062},
		{0.000, 0.000, 0.000, 0.000},
	}, "Q", te)
	assertEq(t.K, matrix{
		{-0.816, 2.041, 2.041, 3.470},
		{-1.837, -1.021, 2.041, 1.429},
		{-0.816, 3.062, 6.124, 1.429},
		{0.000, 0.000, 0.000, 0.000},
	}, "K", te)
	assertEq(t.QK, matrix{
		{-1.584, -2.236, 2.609, 0},
		{3.913, -1.397, 2.515, 0},
		{6.242, -1.397, 7.640, 0},
		{0.000, 0.000, 0.000, 0},
	}, "K", te)
	assertEq(t.QK, matrix{
		{-1.584, -2.236, 2.609, 0},
		{3.913, -1.397, 2.515, 0},
		{6.242, -1.397, 7.640, 0},
		{0.000, 0.000, 0.000, 0},
	}, "K", te)
	assertEq(t.S, matrix{
		{1, 0, 0, 0},
		{0.995, 0.005, 0, 0},
		{0.198, 0, 0.802, 0},
		{0.000, 0.000, 0.000, 0},
	}, "K", te)
	assertEq(t.V, matrix{
		{2, -1, -1, 0.5, 0.5},
		{1.990, -0.985, -0.995, 0.488, 0.488},
		{-0.004, -1, 0.203, 0.5, 0.5},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "V", te)
	assertEq(t.R1, matrix{
		{3.000, -1.000, -1.000, 1.500, 0.500},
		{1.990, 0.015, -0.995, 1.488, 0.488},
		{-0.004, -1.000, 0.203, 1.500, 1.500},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "R1", te)
	assertEq(t.I, matrix{
		{-1.046, 1.896, 0.261, -3.072, -2.164},
		{-0.071, 4.139, -1.060, -2.525, -3.233},
		{-1.397, 0.579, 2.994, -0.180, 3.338},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "I", te)
	assertEq(t.A, matrix{
		{0.000, 1.896, 0.261, 0.000, 0.000},
		{0.000, 4.139, 0.000, 0.000, 0.000},
		{0.000, 0.579, 2.994, 0.000, 3.338},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "I", te)
	assertEq(t.H, matrix{
		{-0.883, 2.157, 1.399, 0.131, 0.605},
		{-2.070, 4.139, 2.483, 0.000, 1.035},
		{-2.879, 6.077, 5.011, 1.497, 7.484},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "H", te)
	assertEq(t.R2, matrix{
		{2.117, 1.157, 0.399, 1.631, 1.105},
		{-0.079, 4.154, 1.488, 1.488, 1.522},
		{-2.883, 5.077, 5.214, 2.997, 8.983},
		{0.000, 0.000, 0.000, 0.000, 0.000},
	}, "R2", te)
	assertEq(t.L, matrix{
		{4.791, -2.970, 10.545},
		{10.353, -0.691, 7.226},
		{26.706, -16.640, 8.707},
		{0.500, -0.500, 0.300},
	}, "L", te)
}

func TestHeatmaps(te *testing.T) {
	t := newT(3, 3, 3, nil, nil, nil, nil, ReLU)
	t.xs = testMat(matrix{
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 0},
	})
	t.V = testMat(matrix{
		{0, 0, 0},
		{2, -1, 3},
		{0, 0, 0},
	})
	t.R1 = testMat(matrix{
		{0, 0, 0},
		{-1, 1.5, 0.5},
		{0, 0, 0},
	})
	t.H = testMat(matrix{
		{0, 0, 0},
		{-1, 4, 2},
		{0, 0, 0},
	})
	t.R2 = testMat(matrix{
		{0, 0, 0},
		{1, -1, -0.5},
		{0, 0, 0},
	})
	t.linear = testMat(matrix{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	})
	t.printHeatmap(1, 1)
	// printMat(t.heatmap)
	assertEq(t.heatmap, matrix{
		{0.000, 0.200, 0.000},
		{0.400, -0.200, 0.600},
		{-0.200, 0.300, 0.100},
		{-0.200, 0.800, 0.400},
		{0.200, -0.200, -0.100},
	}, "heatmap", te)
}

func TestLoss(te *testing.T) {
	t := newT(5, 4, 4, nil, nil, nil, nil, nil)
	t.L = testMat(matrix{
		{1, 2, 3, -9},
		{2, 1.6, 1, 0.1},
		{5, -8, -13},
		{-2, 100, -1},
		{-3.14, 7.77, 0},
	})
	p := func(r, c int) float64 {
		sum := 0.0
		for _, l := range t.L[r] {
			sum += math.Exp(l)
		}
		return -math.Log(math.Exp(float64(t.L[r][c])) / sum)
	}
	t.ys = []int{1, 0, 2, 0, 2}
	loss := t.loss()
	expected := (p(0, 1) + p(1, 0) + p(2, 2) + p(3, 0) + p(4, 2)) / 5.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
}
