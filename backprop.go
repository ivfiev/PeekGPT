package main

import (
	"math"
	"slices"
)

func (m *model) backward() {
	m.dLogits()
	m.dLinear()
	mulMatT(m.blocks[len(m.blocks)-1].dR1, m.dL, m.linear)
	for i := len(m.blocks) - 1; i >= 0; i-- {
		m.blocks[i].backward()
		if i > 0 {
			m.blocks[i-1].dR1.Copy(m.blocks[i].dXS0)
		}
	}
	// l(m.L(unembed(...)))
	// l'(m.L(...)) * m.L'(unembed..)
	// dL/dLogits * dLogits/dlinear * dlinear/...
	m.dEmbeds()
}

func (b *block) backward() {
	b.dH.Copy(b.dR1)
	b.dR0.Copy(b.dR1)
	mulTmat(b.dhidden, b.A, b.dH)
	sumCols(b.dbias1, b.dH)
	mulMatT(b.dA, b.dH, b.hidden)
	b.dI.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return b.dA.At(i, j)
		}
		return 0
	}, b.I)
	mulTmat(b.dinput, b.XS2, b.dI)
	sumCols(b.dbias0, b.dI)
	mulMatT(b.dXS2, b.dI, b.input)
	b.hatXS2.Apply(func(i, j int, v float64) float64 {
		return (v - b.beta1[j]) / b.gamma1[j]
	}, b.XS2)
	b.dXS2ThatXS2.Apply(func(i, j int, v float64) float64 {
		return b.dXS2.At(i, j) * b.hatXS2.At(i, j)
	}, b.dXS2ThatXS2)
	b.layerNormBackward(b.dR0, b.R0, b.dXS2, b.hatXS2, b.dhatXS2, b.dXS2ThatXS2, b.gamma1, b.dgamma1, b.dbeta1)

	b.dP.Copy(b.dR0)
	b.dXS0.Copy(b.dR0)
	mulMatT(b.dCV, b.dP, b.proj)
	mulTmat(b.dproj, b.CV, b.dP)
	b.dSVs()
	b.dXS1.Zero()
	for i := range b.attn {
		// d := 1.0 / math.Sqrt(float64(b.attn))
		mulMatT(b.dS[i], b.dSV[i], b.V[i])
		mulTmat(b.dV[i], b.S[i], b.dSV[i])
		b.softmaxBackward(b.dQK[i], b.dS[i], b.S[i])
		mulMat(b.dQ[i], b.dQK[i], b.K[i])
		mulTmat(b.dK[i], b.dQK[i], b.Q[i])
		mulTmat(b.dqueries[i], b.XS1, b.dQ[i])
		mulTmat(b.dkeys[i], b.XS1, b.dK[i])
		mulTmat(b.dvalues[i], b.XS1, b.dV[i])
		mulMatT(b.dXS1q[i], b.dQ[i], b.queries[i])
		mulMatT(b.dXS1k[i], b.dK[i], b.keys[i])
		mulMatT(b.dXS1v[i], b.dV[i], b.values[i])
		addMatM(b.dXS1, b.dXS1, b.dXS1q[i])
		addMatM(b.dXS1, b.dXS1, b.dXS1k[i])
		addMatM(b.dXS1, b.dXS1, b.dXS1v[i])
		// y = f(x)
		// z = g(y) + h(y)
		// dz/dx = g'(f(x))*f'(x) + h'(f(x))*f'(x)
		// dz/dy = g'(y) + h'(y)
	}
	b.hatXS1.Apply(func(i, j int, v float64) float64 {
		return (v - b.beta0[j]) / b.gamma0[j]
	}, b.XS1)
	b.dXS1ThatXS1.Apply(func(i, j int, v float64) float64 {
		return b.dXS1.At(i, j) * b.hatXS1.At(i, j)
	}, b.dXS1ThatXS1)
	b.layerNormBackward(b.dXS0, b.XS0, b.dXS1, b.hatXS1, b.dhatXS1, b.dXS1ThatXS1, b.gamma0, b.dgamma0, b.dbeta0)
}

func (m *model) dLogits() {
	m.dL.Zero()
	count := 0.0
	d, rl, cl, sl := unmat(m.L)
	g, _, _, sg := unmat(m.dL)
	for i := range rl {
		if m.ys[i] == -1 {
			continue
		}
		count++
		row := d[i*sl : i*sl+cl]
		rowMax, _ := rowMax(row)
		sum := rowSum(row, rowMax)
		for c := range row {
			g[i*sg+c] = math.Exp(d[i*sl+c]-rowMax) / sum
			if m.ys[i] == c {
				g[i*sg+c] -= 1.0
			}
		}
	}
	for i := range g {
		g[i] /= count
	}
}

func (b *block) dSVs() {
	_, rows, cols, _ := unmat(b.dCV)
	for r := range rows {
		for c := range cols {
			b.dSV[c/b.dAttn].Set(r, c%b.dAttn, b.dCV.At(r, c))
		}
	}
}

func (m *model) dLinear() {
	m.dlinear.Zero()
	block := m.blocks[len(m.blocks)-1]
	mulTmat(m.dlinear, block.R1, m.dL)
	sumCols(m.dbias2, m.dL)
}

func (m *model) dEmbeds() {
	m.dtokens.Zero()
	m.dpositions.Zero()
	for i, c := range m.prompt {
		v := slices.Index(m.vocab, c)
		for j := range m.dModel {
			m.dtokens.Set(v, j, m.blocks[0].dXS0.At(i, j)+m.dtokens.At(v, j))
			m.dpositions.Set(i, j, m.blocks[0].dXS0.At(i, j))
		}
	}
}

func (b *block) layerNormBackward(dLdXS, XS, dXS, hatXS, dhatXS, dXSThatXS matrix, gamma, dgamma, dbeta vector) {
	sumCols(dgamma, dXSThatXS)
	sumCols(dbeta, dXS)
	// dL/dXS
	const eps = 0.00001
	ctx, dModel := b.context, b.dModel
	for i := range ctx {
		u := 0.0
		for j := range dModel {
			u += XS.At(i, j)
		}
		u /= float64(dModel)
		o2 := 0.0
		for j := range dModel {
			diff := XS.At(i, j) - u
			o2 += diff * diff
		}
		o2 = 1.0 / math.Sqrt(o2/float64(dModel)+eps)
		sumdhat := 0.0
		sumdhat2 := 0.0
		for j := range dModel {
			dhatXS.Set(i, j, dXS.At(i, j)*gamma[j])
			sumdhat += dhatXS.At(i, j)
			sumdhat2 += dhatXS.At(i, j) * hatXS.At(i, j)
		}
		for j := range dModel {
			dx := (dhatXS.At(i, j) - sumdhat/float64(dModel) - hatXS.At(i, j)*sumdhat2/float64(dModel)) * o2
			dLdXS.Set(i, j, dx+dLdXS.At(i, j))
		}
	}
}

func (b *block) softmaxBackward(dQK, dS, S matrix) {
	scale := 1.0 / math.Sqrt(float64(b.dAttn))
	_, rows, cols, _ := unmat(S)
	for i := range rows {
		sum := 0.0
		for j := range cols {
			sum += dS.At(i, j) * S.At(i, j)
		}
		for j := range cols {
			val := S.At(i, j) * (dS.At(i, j) - sum)
			dQK.Set(i, j, val*scale)
		}
	}
}
