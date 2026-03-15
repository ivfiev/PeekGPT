package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
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
		_, r, c := unmat(actual)
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
		da, ra, ca := unmat(actual)
		de, re, ce := unmat(expected)
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

func assertValidation(model *model, mode tmode, targetLoss float64, data [][]rune, te *testing.T) {
	tr := newTrainer(model, 1)
	tr.validation = data
	tr.mode = mode
	loss := tr.validate(model)
	if math.Abs(loss-targetLoss) > 1e-13 {
		te.Fatalf("validation loss\n%.13f\n!=\n%.13f", targetLoss, loss)
	}
}

func TestIntegrationTask(te *testing.T) {
	var seed int64 = 7359
	rng := rand.New(rand.NewSource(seed))
	data := genCopyDataset([]rune("1234"), 4, 500, rng)
	m := train(16, 9, 8, 2, 2, 2,
		data[:450], data[450:], 0,
		150, 32, 8, 0.01, seed, nil)
	assert := func(expected string) {
		ctx := []rune(fmt.Sprintf("%s|%s", expected, strings.Repeat("?", len(expected))))
		toks, _ := m.predict(ctx)
		actual := string(toks[len(expected)+1 : len(expected)*2+1])
		if expected != actual {
			te.Fatalf("Integration %s != %s", expected, actual)
		}
	}
	assert("21")
	assert("322")
	assert("111")
	assert("2211")
	assert("3341")
	assert("13")
	assert("31")
	assert("3")
	assert("2")
	const target = 0.0011941290520
	assertValidation(m, task, target, data, te)
}

func TestIntegrationTextgenRaceCond(te *testing.T) {
	var seed int64 = 7359
	data := []rune(`
Humpty Dumpty sat on a wall.
Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again.
`)
	mPar := train(16, 8, 8, 2, 2, 2,
		[][]rune{data}, [][]rune{[]rune("Humpty Dumpty sat on a wall.")}, 0,
		133, 17, 13, 0.01, seed, nil)
	mSeq := train(16, 8, 8, 2, 2, 2,
		[][]rune{data}, [][]rune{[]rune("Humpty Dumpty sat on a wall.")}, 0,
		133, 17, 1, 0.01, seed, nil)
	const target = 0.6264098015485
	assertValidation(mPar, text, target, [][]rune{data[:30]}, te)
	assertValidation(mSeq, text, target, [][]rune{data[:30]}, te)
}

func TestAllParametersActuallyTrain(te *testing.T) {
	var seed int64 = 7357
	data := []rune(`
Humpty Dumpty sat on a wall.
Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again.
`)
	m1 := newModel(8, 8, 8, 2, 2, 2, getVocab([][]rune{data}, text))
	m1.rand(rand.New(rand.NewSource(seed)))
	initial := make(vector, m1.size())
	m1.dump(initial)
	m2 := train(8, 8, 8, 2, 2, 2,
		[][]rune{data}, [][]rune{[]rune("Humpty Dumpty sat on a wall.")}, 0,
		37, 16, 8, 0.01, seed, m1)
	trained := make(vector, m2.size())
	m2.dump(trained)
	for i := range m1.size() {
		if math.Abs(initial[i]-trained[i]) < 0.000001 {
			te.Errorf("%d: %.12f != %.12f\n", i, initial[i], trained[i])
		}
	}
}

func TestPointLoss(te *testing.T) {
	m := newModel(4, 7, 4, 3, 2, 2, []rune("abc|?"))
	m.rand(rand.New(rand.NewSource(7357)))
	t := newTrainer(m, 8)
	p := func(r, c int) float64 {
		d, _, cols := unmat(m.L)
		sum := 0.0
		for i := range cols {
			sum += math.Exp(d[r*cols+i])
		}
		return -math.Log(math.Exp(float64(d[r*cols+c])) / sum)
	}
	actual := t.pointLoss(m, []rune("bc|??=bc"))
	expected := (p(3, 1) + p(4, 2)) / 2.0
	if math.Abs(actual-expected) > 0.000000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
	actual = t.pointLoss(m, []rune("cba|???=cba"))
	expected = (p(4, 2) + p(5, 1) + p(6, 0)) / 3.0
	if math.Abs(actual-expected) > 0.000000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
	actual = t.pointLoss(m, []rune("a|?=a"))
	expected = (p(2, 0)) / 1.0
	if math.Abs(actual-expected) > 0.000000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
}

func TestLoadYs(te *testing.T) {
	m := newModel(4, 9, 4, 3, 2, 2, []rune("012345|?"))
	m.rand(rand.New(rand.NewSource(7357)))
	t := newTrainer(m, 8)
	t.pointLoss(m, []rune("4012345|?=4"))
	expected := []int{-1, -1, -1, -1, -1, -1, -1, -1, 4}
	if len(m.ys) != len(expected) {
		te.Fatalf("%d != %d", len(m.ys), len(expected))
	}
	for i := range expected {
		if m.ys[i] != expected[i] {
			te.Fatalf("%d != %d", m.ys[i], expected[i])
		}
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
	layerNorm(ys, ys, xs, gamma, beta)
	assertEq(ys, [][]float64{
		{2, -2.5, -3, 2},
		{1.697, -3.092, 1.325, 0.535},
		{1, -0.412, -4.138, 2.177},
	}, "LayerNorm", te)
}

func TestSoftmaxT(te *testing.T) {
	xs := testMat([][]float64{
		{1, 1, 1, 1},
		{1002, 1004, -7, 3},
		{-4, 1, -5, 2},
		{4, 2, -5, -1},
	})
	ys := makeMat(4, 4)
	softmax(ys, xs, true)
	assertEq(ys, [][]float64{
		{1, 0, 0, 0},
		{0.119, 0.881, 0, 0},
		{0.007, 0.991, 0.002, 0},
		{0.876, 0.118, 0, 0.006},
	}, "Softmax", te)
}

func TestSoftmax(te *testing.T) {
	xs := testMat([][]float64{
		{1, 1, 1, 1},
		{1002, 1004, -7, 1002},
		{-4, 1, -5, 2},
		{4, 2, -5, -1},
	})
	ys := makeMat(4, 4)
	softmax(ys, xs, false)
	assertEq(ys, [][]float64{
		{0.25, 0.25, 0.25, 0.25},
		{0.107, 0.786, 0, 0.107},
		{0.002, 0.268, 0.001, 0.730},
		{0.876, 0.118, 0, 0.006},
	}, "Softmax", te)
}

func TestBlockLayerNorm(te *testing.T) {
	b := newBlock(4, 3, 4, 2, 1, ReLU)
	b.XS0 = testMat([][]float64{
		{1, 0, 0, 0},
		{0, 0, 0, 1},
		{0, 0, 0, 0},
	})
	b.gamma0 = vector{1.1, 1.2, 1.3, 1.4}
	b.beta0 = vector{0.5, 0.6, 0.7, -0.5}
	b.forward()
	assertEq(b.XS1, [][]float64{
		{2.405, -0.093, -0.051, -1.308},
		{-0.135, -0.093, -0.051, 1.925},
		{0.000, 0.000, 0.000, 0.000},
	}, "xs1", te)
}

func TestAddMatV(te *testing.T) {
	A := testMat([][]float64{
		{0, 1, 2, 0},
		{3, 4, 5, 0},
		{6, 7, 8, 9},
	})
	x := vector{1.1, 1.2, 1.3, 1.4}
	addMatV(A, x)
	assertEq(A, [][]float64{
		{1.1, 2.2, 3.3, 1.4},
		{4.1, 5.2, 6.3, 1.4},
		{7.1, 8.2, 9.3, 10.4},
	}, "addMatV", te)
}

func TestBlocksE2E(te *testing.T) {
	m := newModel(4, 3, 2, 2, 2, 2, []rune("abc"))
	m.rand(rand.New(rand.NewSource(7357)))
	m.XS = testMat([][]float64{
		{0.2, -0.34, 1.2, -0.5},
		{-0.6, 0.1, 0.2, 0.6},
		{2, 1, 0, -1},
	})
	m.gamma2 = vector{1.1, 1.1, 1.2, 1.1}
	m.beta2 = vector{0.1, 0.1, 0.2, 0.1}
	m.bias2 = vector{1, 2, 3}
	m.blocks[0].bias0 = vector{0.1, 0.2, -0.3, -0.4, 0.1, 0.2, -0.3, -0.4}
	m.blocks[0].bias1 = vector{-0.1, -0.2, 0.3, 0.4}
	mat34 := makeMat(3, 4)
	mat38 := makeMat(3, 8)
	mat33 := makeMat(3, 3)
	mat32 := makeMat(3, 2)
	m.forward()

	assertEq(m.blocks[0].XS0, m.XS, "0.xs0", te)

	layerNorm(mat34, mat34, m.blocks[0].XS0, m.blocks[0].gamma0, m.blocks[0].beta0)
	assertEq(mat34, m.blocks[0].XS1, "0.xs1", te)

	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].queries[0])
	assertEq(mat32, m.blocks[0].Q[0], "0.Q.0", te)
	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].queries[1])
	assertEq(mat32, m.blocks[0].Q[1], "0.Q.1", te)

	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].keys[0])
	assertEq(mat32, m.blocks[0].K[0], "0.K.0", te)
	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].keys[1])
	assertEq(mat32, m.blocks[0].K[1], "0.K.1", te)

	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].values[0])
	assertEq(mat32, m.blocks[0].V[0], "0.V.0", te)
	mulMat(mat32, m.blocks[0].XS1, m.blocks[0].values[1])
	assertEq(mat32, m.blocks[0].V[1], "0.V.1", te)

	mulMatT(mat33, m.blocks[0].Q[0], m.blocks[0].K[0])
	mulMatK(mat33, 1/math.Sqrt(2))
	assertEq(mat33, m.blocks[0].QK[0], "0.QK.0", te)
	mulMatT(mat33, m.blocks[0].Q[1], m.blocks[0].K[1])
	mulMatK(mat33, 1/math.Sqrt(2))
	assertEq(mat33, m.blocks[0].QK[1], "0.QK.1", te)

	softmax(mat33, m.blocks[0].QK[0], true)
	assertEq(mat33, m.blocks[0].S[0], "0.S.0", te)
	softmax(mat33, m.blocks[0].QK[1], true)
	assertEq(mat33, m.blocks[0].S[1], "0.S.1", te)

	mulMat(mat32, m.blocks[0].S[0], m.blocks[0].V[0])
	assertEq(mat32, m.blocks[0].SV[0], "0.SV.0", te)
	mulMat(mat32, m.blocks[0].S[1], m.blocks[0].V[1])
	assertEq(mat32, m.blocks[0].SV[1], "0.SV.1", te)

	catMat(mat34, m.blocks[0].SV)
	assertEq(mat34, m.blocks[0].CV, "0.CV", te)

	mulMat(mat34, m.blocks[0].CV, m.blocks[0].proj)
	assertEq(mat34, m.blocks[0].P, "0.P", te)

	mat34.Zero()
	addMatM(mat34, m.blocks[0].XS0, m.blocks[0].P)
	assertEq(mat34, m.blocks[0].R0, "0.R1", te)

	mat34.Zero()
	layerNorm(mat34, mat34, m.blocks[0].R0, m.blocks[0].gamma1, m.blocks[0].beta1)
	assertEq(mat34, m.blocks[0].XS2, "0.xs2", te)

	mulMat(mat38, m.blocks[0].XS2, m.blocks[0].input)
	addMatV(mat38, m.blocks[0].bias0)
	assertEq(mat38, m.blocks[0].I, "0.I", te)

	mapMat(mat38, m.blocks[0].I, ReLU)
	assertEq(mat38, m.blocks[0].A, "0.A", te)

	mulMat(mat34, m.blocks[0].A, m.blocks[0].hidden)
	addMatV(mat34, m.blocks[0].bias1)
	assertEq(mat34, m.blocks[0].H, "0.H", te)

	addMatM(mat34, m.blocks[0].H, m.blocks[0].R0)
	assertEq(mat34, m.blocks[0].R1, "0.R2", te)

	assertEq(m.blocks[1].XS0, m.blocks[0].R1, "1.xs0", te)

	layerNorm(mat34, mat34, m.blocks[1].R1, m.gamma2, m.beta2)
	mulMat(mat33, mat34, m.unembed)
	addMatV(mat33, m.bias2)
	assertEq(m.L, mat33, "t.L", te)

	softmax(mat33, m.L, false)
	assertEq(mat33, m.S, "t.S", te)

	d, _, _ := unmat(m.L)
	for _, x := range d {
		if math.Abs(x) < 0.001 {
			te.Fatal("t.L = 0")
		}
	}
}

func TestHeatmaps(te *testing.T) {
	m := newModel(4, 3, 4, 1, 1, 1, []rune("abc"))
	m.unembed = testMat([][]float64{
		{1, 1, 1},
		{1, 1, -1},
		{1, 1, -1},
		{1, 1, 2},
	})
	m.L = testMat([][]float64{
		{-1, -1, -1},
		{-1, -1, 1},
		{-1, -1, -1},
	})
	m.blocks[0].XS0 = testMat([][]float64{
		{1, 0, 0, 1},
		{0, 1, 0, 2},
		{0, 0, 1, 3},
	})
	m.blocks[0].P = testMat([][]float64{
		{1, 0, 0, 1},
		{-1, 1, 1, 0},
		{0, 0, 1, 3},
	})
	m.blocks[0].R0 = testMat([][]float64{
		{1, 0, 0, 1},
		{0, 1, 1, 0},
		{0, 0, 1, 3},
	})
	m.blocks[0].H = testMat([][]float64{
		{1, 0, 0, 1},
		{0, 1, 0, 2},
		{0, 0, 1, 3},
	})
	m.blocks[0].R1 = testMat([][]float64{
		{1, 0, 0, 1},
		{0, 1, 0, 2},
		{0, 0, 1, 3},
	})
	heatmaps := m.calcHeatmap(1, []matrix{
		m.blocks[0].XS0,
		m.blocks[0].P,
		m.blocks[0].R0,
		m.blocks[0].H,
		m.blocks[0].R1,
	})
	m.prompt = []rune("abc")
	m.printHeatmap([]int{1})
	assertEq(testMat(heatmaps), [][]float64{
		{0, -0.2, 0, 0.8},
		{-0.2, -0.2, -0.2, 0},
		{0, -0.2, -0.2, 0},
		{0, -0.2, 0, 0.8},
		{0, -0.2, 0, 0.8},
	}, "heatmaps", te)
}

func TestLossLogits(te *testing.T) {
	m := newModel(4, 5, 4, 1, 1, 1, []rune("abcd"))
	m.L = testMat([][]float64{
		{1, 2, 3, -9},
		{2, 1.6, 1, 0.1},
		{5, -8, -13},
		{-2, 100, -1},
		{-3.14, 7.77, 0},
	})
	softmax(m.S, m.L, false)
	p := func(r, c int) float64 {
		d, _, cols := unmat(m.L)
		sum := 0.0
		for i := range cols {
			sum += math.Exp(d[r*cols+i])
		}
		return -math.Log(math.Exp(float64(d[r*cols+c])) / sum)
	}
	m.ys = []int{1, 0, 2, 0, 2}
	loss := m.loss()
	expected := (p(0, 1) + p(1, 0) + p(2, 2) + p(3, 0) + p(4, 2)) / 5.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
	m.ys = []int{-1, 1, 2, -1, -1}
	loss = m.loss()
	expected = (p(1, 1) + p(2, 2)) / 2.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
}

func TestLossSoftmax(te *testing.T) {
	m := newModel(4, 5, 4, 1, 1, 1, []rune("abcd"))
	m.S = testMat([][]float64{
		{0.25, 0.33, 0.40, 0.02},
		{0.1, 0.8, 0.05, 0.05},
		{0.99, 0.003, 0.003, 0.003},
		{0.5, 0.00001, 0.00001, 0.5},
		{0.00001, 0.2, 0.39, 0.41},
	})
	p := func(r, c int) float64 {
		return -math.Log(m.S.At(r, c))
	}
	m.ys = []int{1, 0, 2, 0, 2}
	loss := m.loss()
	expected := (p(0, 1) + p(1, 0) + p(2, 2) + p(3, 0) + p(4, 2)) / 5.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
	m.ys = []int{-1, 1, 2, -1, -1}
	loss = m.loss()
	expected = (p(1, 1) + p(2, 2)) / 2.0
	if math.Abs(loss-expected) > 0.0001 {
		te.Fatalf("Losses are not equal: %f != %f", loss, expected)
	}
}

func TestLoadXs(te *testing.T) {
	m := newModel(4, 3, 4, 1, 1, 1, []rune("abc"))
	m.tokens = testMat([][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	})
	m.positions = testMat([][]float64{
		{0, 0, 0, -0.5},
		{0, -0.1, 0.1, 0},
		{0, 0, 0, 0.5},
	})
	m.loadXs([]rune("bac"))
	assertEq(m.XS, [][]float64{
		{0, 1, 0, -0.5},
		{1, -0.1, 0.1, 0},
		{0, 0, 1, 0.5},
	}, "LoadXS", te)
	m.loadXs([]rune("cb"))
	assertEq(m.XS, [][]float64{
		{0, 0, 1, -0.5},
		{0, 0.9, 0.1, 0},
		{0, 0, 0, 0},
	}, "LoadXS", te)
}

func TestMatrixInit(te *testing.T) {
	assert := func(X any, p func(float64) bool) {
		switch X := X.(type) {
		case matrix:
			_, rows, cols := unmat(X)
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
		default:
			te.Fatalf("bad input %v", X)
		}
	}
	m := newModel(4, 3, 4, 1, 2, 2, []rune("abc"))
	assert(m.tokens, func(f float64) bool { return f == 0 })
	assert(m.XS, func(f float64) bool { return f == 0 })

	m.rand(rand.New(rand.NewSource(7357)))
	assert(m.tokens, func(f float64) bool { return f != 0 })
	assert(m.positions, func(f float64) bool { return f != 0 })
	for _, b := range m.blocks {
		assert(b.gamma0, func(f float64) bool { return f == 1 })
		assert(b.beta0, func(f float64) bool { return f == 0 })
		for a := range b.attn {
			assert(b.queries[a], func(f float64) bool { return f != 0 })
			assert(b.keys[a], func(f float64) bool { return f != 0 })
			assert(b.values[a], func(f float64) bool { return f != 0 })
		}
		assert(b.proj, func(f float64) bool { return f != 0 })
		assert(b.gamma1, func(f float64) bool { return f == 1 })
		assert(b.beta1, func(f float64) bool { return f == 0 })
		assert(b.input, func(f float64) bool { return f != 0 })
		assert(b.bias0, func(f float64) bool { return f == 0 })
		assert(b.hidden, func(f float64) bool { return f == 0 })
		assert(b.bias1, func(f float64) bool { return f == 0 })
	}
	assert(m.gamma2, func(f float64) bool { return f == 1 })
	assert(m.beta2, func(f float64) bool { return f == 0 })
	assert(m.unembed, func(f float64) bool { return f != 0 })
	assert(m.bias2, func(f float64) bool { return f == 0 })

	theta := make(vector, m.size())
	for i := range theta {
		theta[i] = 7357
	}
	m.apply(theta)
	assert(m.tokens, func(f float64) bool { return f == 7357 })
	assert(m.positions, func(f float64) bool { return f == 7357 })
	for _, b := range m.blocks {
		assert(b.gamma0, func(f float64) bool { return f == 7357 })
		assert(b.beta0, func(f float64) bool { return f == 7357 })
		for a := range b.attn {
			assert(b.queries[a], func(f float64) bool { return f == 7357 })
			assert(b.keys[a], func(f float64) bool { return f == 7357 })
			assert(b.values[a], func(f float64) bool { return f == 7357 })
		}
		assert(b.proj, func(f float64) bool { return f == 7357 })
		assert(b.gamma1, func(f float64) bool { return f == 7357 })
		assert(b.beta1, func(f float64) bool { return f == 7357 })
		assert(b.input, func(f float64) bool { return f == 7357 })
		assert(b.bias0, func(f float64) bool { return f == 7357 })
		assert(b.hidden, func(f float64) bool { return f == 7357 })
		assert(b.bias1, func(f float64) bool { return f == 7357 })
	}
	assert(m.gamma2, func(f float64) bool { return f == 7357 })
	assert(m.beta2, func(f float64) bool { return f == 7357 })
	assert(m.unembed, func(f float64) bool { return f == 7357 })
	assert(m.bias2, func(f float64) bool { return f == 7357 })

	for i := range theta {
		theta[i] = 0
	}
	assert(theta, func(f float64) bool { return f == 0 })
	m.dump(theta)
	assert(theta, func(f float64) bool { return f == 7357 })

	theta1 := make(vector, m.size())
	theta2 := make(vector, m.size())
	m.rand(rand.New(rand.NewSource(7737)))
	m.gamma2 = vector{1.2, 1.4, 1.6, 1.8}
	m.beta2 = vector{0.21, 0.31, 0.41, 0.51}
	m.bias2 = vector{1, 1, 1}
	m.blocks[0].bias0 = vector{2, 2, 2, 2, 2, 2, 2, 2}
	m.blocks[0].bias1 = vector{3, 3, 3, 3}
	m.blocks[0].beta0 = vector{4, 4, 4, 4}
	m.blocks[0].beta1 = vector{5, 5, 5, 5}
	m.blocks[1].bias0 = vector{20, 20, 20, 20, 20, 20, 20, 20}
	m.blocks[1].bias1 = vector{30, 30, 30, 30}
	m.blocks[1].beta0 = vector{40, 40, 40, 40}
	m.blocks[1].beta1 = vector{50, 50, 50, 50}
	m.dump(theta1)
	m.apply(theta1)
	m.dump(theta2)
	for i := range theta1 {
		if theta1[i] != theta2[i] {
			te.Fatalf("apply != dump, %f != %f\n", theta1[i], theta2[i])
		}
	}
}

func TestMatrixCat(te *testing.T) {
	D := makeMat(3, 6)
	A := testMat([][]float64{
		{-1, -2},
		{-3, -4},
		{-5, -6},
	})
	B := testMat([][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	})
	C := testMat([][]float64{
		{0.1, 0.2},
		{0.4, 0.5},
		{0.7, 0.8},
	})
	catMat(D, []matrix{A, C, B})
	assertEq(D, [][]float64{
		{-1, -2, 0.1, 0.2, 1, 2},
		{-3, -4, 0.4, 0.5, 3, 4},
		{-5, -6, 0.7, 0.8, 5, 6},
	}, "catMat", te)
}

func TestBackprop(te *testing.T) {
	const eps = 1e-7
	rng := rand.New(rand.NewSource(7357))
	m := newModel(4, 3, 2, 2, 2, 3, []rune("abcde"))
	fmt.Printf("Backprop test model size: %d\n", m.size())
	m.rand(rng)
	for i := range m.dModel {
		m.gamma2[i] = 0.88 + rng.Float64()
		m.bias2[i] = -0.5 + rng.Float64()
	}
	for _, b := range m.blocks {
		for i := range m.dModel {
			b.bias0[i] = -0.5 + rng.Float64()
			b.bias1[i] = -0.5 + rng.Float64()
			b.gamma0[i] = 0.7 + rng.Float64()
			b.gamma1[i] = 0.8 + rng.Float64()
			b.beta0[i] = -0.5 + rng.Float64()
			b.beta1[i] = -0.5 + rng.Float64()
		}
	}
	expected := make(vector, m.size())
	actual := make(vector, m.size())
	finiteDiff := func(prompt []rune) {
		theta := make(vector, m.size())
		alpha := make(vector, m.size())
		m.dump(theta)
		m.dump(alpha)
		for i := range theta {
			theta[i] += eps
			m.apply(theta)
			m.loadXs(prompt)
			m.forward()
			plus := m.loss()
			theta[i] -= 2 * eps
			m.apply(theta)
			m.loadXs(prompt)
			m.forward()
			minus := m.loss()
			expected[i] = (plus - minus) / (2 * eps)
			theta[i] += eps
		}
		m.apply(alpha)
	}
	backprop := func(prompt []rune) {
		m.loadXs(prompt)
		m.forward()
		m.backward()
		m.grad(actual, 0)
	}
	test := func(proompt []rune, ys []int) {
		copy(m.ys, ys)
		finiteDiff(proompt)
		backprop(proompt)
		for i := range m.size() {
			e := expected[i]
			a := actual[i]
			if math.Abs(e-a) > eps {
				te.Errorf("%d: %.9f != %.9f", i, e, a)
			}
			if math.Abs(e-a)/max(1, math.Abs(e), math.Abs(a)) > eps {
				te.Errorf("%d: %.9f != %.9f", i, e, a)
			}
		}
	}
	test([]rune("a"), []int{1, -1, -1})
	test([]rune("e"), []int{3, -1, -1})
	test([]rune("ab"), []int{1, 2, -1})
	test([]rune("ed"), []int{0, 2, -1})
	test([]rune("cd"), []int{1, 1, -1})
	test([]rune("eee"), []int{3, 1, 1})
	test([]rune("dab"), []int{1, 1, 0})
	test([]rune("bac"), []int{0, 2, 1})
}
