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
	us      vector
	vs      vector
}

func (q *testquad) eval(theta vector) float64 {
	sum := 0.0
	for i := range theta {
		sum += (theta[i] - q.minimum[i]) * (theta[i] - q.minimum[i])
	}
	return sum
}

func (q *testquad) eval2(us, vs []vector, i int) (vector, vector) {
	for i := range us {
		q.us[i] = q.eval(us[i])
		q.vs[i] = q.eval(vs[i])
	}
	return q.us, q.vs
}

func TestSPSA(te *testing.T) {
	const size = 100
	q := testquad{
		minimum: make(vector, size),
		us:      make(vector, 9),
		vs:      make(vector, 9),
	}
	theta := make(vector, size)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	for i := range size {
		q.minimum[i] = rng.Float64() * 10
		theta[i] = -rng.Float64() * 10
	}
	spsa(&q, theta, 4, 2000, 0.005, 0.0001, seed)
	loss := q.eval(theta)
	if loss > 0.001 {
		fmt.Printf("SPSA loss: %f, seed: %d\n", loss, seed)
		te.Fatal("SPSA failed to find minima")
	}
}

func TestIntegration(te *testing.T) {
	var seed int64 = 7359
	rng := rand.New(rand.NewSource(seed))
	m := train(16, 5, 4, 3, 2,
		genCopyDataset([]rune("123"), 2, 30, rng),
		genCopyDataset([]rune("123"), 2, 10, rng),
		4, 1000, 8, 4, 0.01, 0.00001, seed)
	assert := func(expected string) {
		ctx := []rune(fmt.Sprintf("%s|%s", expected, strings.Repeat("?", len(expected))))
		toks, _ := m.predict(ctx)
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

func TestPointLoss(te *testing.T) {
	m := newModel(4, 7, 4, 3, 2, []rune("abc|?"))
	m.rand(rand.New(rand.NewSource(7357)))
	t := newTraining(m)
	p := func(r, c int) float64 {
		d, _, cols, s := unmat(m.L)
		sum := 0.0
		for i := range cols {
			sum += math.Exp(d[r*s+i])
		}
		return -math.Log(math.Exp(float64(d[r*s+c])) / sum)
	}
	actual := t.pointLoss(m, []rune("bc|??=bc"))
	expected := (p(3, 1) + p(4, 2)) / 2.0
	if math.Abs(actual-expected) > 0.000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
	actual = t.pointLoss(m, []rune("cba|???=cba"))
	expected = (p(4, 2) + p(5, 1) + p(6, 0)) / 3.0
	if math.Abs(actual-expected) > 0.000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
	actual = t.pointLoss(m, []rune("a|?=a"))
	expected = (p(2, 0)) / 1.0
	if math.Abs(actual-expected) > 0.000000000001 {
		te.Fatalf("Wrong PointLoss %f != %f\n", actual, expected)
	}
}

func TestLoadYs(te *testing.T) {
	m := newModel(4, 9, 4, 3, 2, []rune("012345|?"))
	m.rand(rand.New(rand.NewSource(7357)))
	t := newTraining(m)
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

func TestBlockLayerNorm(te *testing.T) {
	b := newBlock(4, 3, 4, 1, ReLU)
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
	m := newModel(4, 3, 2, 2, 2, []rune("abc"))
	m.rand(rand.New(rand.NewSource(7357)))
	m.XS = testMat([][]float64{
		{0.2, -0.34, 1.2, -0.5},
		{-0.6, 0.1, 0.2, 0.6},
		{2, 1, 0, -1},
	})
	m.bias2 = vector{1, 2, 3}
	m.blocks[0].bias0 = vector{0.1, 0.2, -0.3, -0.4}
	m.blocks[0].bias1 = vector{-0.1, -0.2, 0.3, 0.4}
	mat34 := makeMat(3, 4)
	mat33 := makeMat(3, 3)
	mat32 := makeMat(3, 2)
	m.forward()

	assertEq(m.blocks[0].XS0, m.XS, "0.xs0", te)

	layerNorm(mat34, m.blocks[0].XS0, m.blocks[0].gamma0, m.blocks[0].beta0)
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

	softmaxT(mat33, m.blocks[0].QK[0])
	assertEq(mat33, m.blocks[0].S[0], "0.S.0", te)
	softmaxT(mat33, m.blocks[0].QK[1])
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
	layerNorm(mat34, m.blocks[0].R0, m.blocks[0].gamma1, m.blocks[0].beta1)
	assertEq(mat34, m.blocks[0].XS2, "0.xs2", te)

	mulMat(mat34, m.blocks[0].XS2, m.blocks[0].input)
	addMatV(mat34, m.blocks[0].bias0)
	assertEq(mat34, m.blocks[0].I, "0.I", te)

	mapMat(mat34, m.blocks[0].I, ReLU)
	assertEq(mat34, m.blocks[0].A, "0.A", te)

	mulMat(mat34, m.blocks[0].A, m.blocks[0].hidden)
	addMatV(mat34, m.blocks[0].bias1)
	assertEq(mat34, m.blocks[0].H, "0.H", te)

	addMatM(mat34, m.blocks[0].H, m.blocks[0].R0)
	assertEq(mat34, m.blocks[0].R1, "0.R2", te)

	assertEq(m.blocks[1].XS0, m.blocks[0].R1, "1.xs0", te)

	mulMat(mat33, m.blocks[1].R1, m.linear)
	addMatV(mat33, m.bias2)
	assertEq(m.L, mat33, "t.L", te)

	d, _, _, _ := unmat(m.L)
	for _, x := range d {
		if math.Abs(x) < 0.001 {
			te.Fatal("t.L = 0")
		}
	}
}

func TestHeatmaps(te *testing.T) {
	m := newModel(4, 3, 4, 1, 1, []rune("abc"))
	m.linear = testMat([][]float64{
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

func TestLoss(te *testing.T) {
	m := newModel(4, 4, 4, 1, 1, []rune("abcd"))
	m.L = testMat([][]float64{
		{1, 2, 3, -9},
		{2, 1.6, 1, 0.1},
		{5, -8, -13},
		{-2, 100, -1},
		{-3.14, 7.77, 0},
	})
	p := func(r, c int) float64 {
		d, _, cols, s := unmat(m.L)
		sum := 0.0
		for i := range cols {
			sum += math.Exp(d[r*s+i])
		}
		return -math.Log(math.Exp(float64(d[r*s+c])) / sum)
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
	m := newModel(4, 3, 4, 1, 1, []rune("abc"))
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
		default:
			te.Fatalf("bad input %v", X)
		}
	}
	m := newModel(4, 3, 4, 1, 2, []rune("abc"))
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
		assert(b.hidden, func(f float64) bool { return f != 0 })
		assert(b.bias1, func(f float64) bool { return f == 0 })
	}
	assert(m.linear, func(f float64) bool { return f != 0 })
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
	assert(m.linear, func(f float64) bool { return f == 7357 })
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
	m.bias2 = vector{1, 1, 1}
	m.blocks[0].bias0 = vector{2, 2, 2, 2}
	m.blocks[0].bias1 = vector{3, 3, 3, 3}
	m.blocks[0].beta0 = vector{4, 4, 4, 4}
	m.blocks[0].beta1 = vector{5, 5, 5, 5}
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
	m := newModel(4, 3, 2, 2, 3, []rune("abcde"))
	m.rand(rand.New(rand.NewSource(7357)))
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
		m.grad(actual)
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
