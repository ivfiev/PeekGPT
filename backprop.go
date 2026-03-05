package main

import (
	"math"
	"slices"
)

func (m *model) backward() {
	dLogits(m)
	dUnembed(m)
	mulMatT(m.blocks[len(m.blocks)-1].dR1, m.dL, m.unembed)
	for i := len(m.blocks) - 1; i >= 0; i-- {
		m.blocks[i].backward()
		if i > 0 {
			m.blocks[i-1].dR1.Copy(m.blocks[i].dXS0)
		}
	}
	// l(m.L(unembed(...)))
	// l'(m.L(...)) * m.L'(unembed..)
	// dL/dLogits * dLogits/dlinear * dlinear/...
	dEmbed(m)
}

func (b *block) backward() {
	b.dH.Copy(b.dR1)
	b.dR0.Copy(b.dR1)
	mulTmat(b.dhidden, b.A, b.dH)
	sumCols(b.dbias1, b.dH)
	mulMatT(b.dA, b.dH, b.hidden)
	dactivation(b.dI, b.dA, b.I)
	mulTmat(b.dinput, b.XS2, b.dI)
	sumCols(b.dbias0, b.dI)
	mulMatT(b.dXS2, b.dI, b.input)
	hats(b.dXS2ThatXS2, b.hatXS2, b.dXS2)
	layerNormBackward(b, b.dR0, b.R0, b.dXS2, b.hatXS2, b.dhatXS2, b.dXS2ThatXS2, b.gamma1, b.dgamma1, b.dbeta1)

	b.dP.Copy(b.dR0)
	b.dXS0.Copy(b.dR0)
	mulMatT(b.dCV, b.dP, b.proj)
	mulTmat(b.dproj, b.CV, b.dP)
	dSVs(b)
	b.dXS1.Zero()
	for i := range b.attn {
		// d := 1.0 / math.Sqrt(float64(b.attn))
		mulMatT(b.dS[i], b.dSV[i], b.V[i])
		mulTmat(b.dV[i], b.S[i], b.dSV[i])
		softmaxBackward(b, b.dQK[i], b.dS[i], b.S[i])
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
	hats(b.dXS1ThatXS1, b.hatXS1, b.dXS1)
	layerNormBackward(b, b.dXS0, b.XS0, b.dXS1, b.hatXS1, b.dhatXS1, b.dXS1ThatXS1, b.gamma0, b.dgamma0, b.dbeta0)
}

func dactivation(dI, dA, I matrix) {
	ddi, _, _ := unmat(dI)
	dda, _, _ := unmat(dA)
	di, _, _ := unmat(I)
	for i := range ddi {
		if di[i] > 0 {
			ddi[i] = dda[i]
		} else {
			ddi[i] = 0
		}
	}
}

func hats(dXSThatXS, hatXS, dXS matrix) {
	dxshatxs, _, _ := unmat(dXSThatXS)
	dxs, _, _ := unmat(dXS)
	dhatxs, _, _ := unmat(hatXS)
	for i := range dxshatxs {
		dxshatxs[i] = dxs[i] * dhatxs[i]
	}
}

func dLogits(m *model) {
	m.dL.Zero()
	count := 0.0
	d, rl, cl := unmat(m.L)
	g, _, cg := unmat(m.dL)
	for i := range rl {
		if m.ys[i] == -1 {
			continue
		}
		count++
		row := d[i*cl : i*cl+cl]
		rowMax, _ := rowMax(row)
		sum := rowSum(row, rowMax)
		for c := range cl {
			g[i*cg+c] = math.Exp(d[i*cl+c]-rowMax) / sum
			if m.ys[i] == c {
				g[i*cg+c] -= 1.0
			}
		}
	}
	for i := range g {
		g[i] /= count
	}
}

func dSVs(b *block) {
	ddcv, rows, cols := unmat(b.dCV)
	for r := range rows {
		for c := range cols {
			b.dSV[c/b.dAttn].Set(r, c%b.dAttn, ddcv[r*cols+c])
		}
	}
}

func dUnembed(m *model) {
	block := m.blocks[len(m.blocks)-1]
	mulTmat(m.dunembed, block.R1, m.dL)
	sumCols(m.dbias2, m.dL)
}

func dEmbed(m *model) {
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

func layerNormBackward(b *block, dLdXS, XS, dXS, hatXS, dhatXS, dXSThatXS matrix, gamma, dgamma, dbeta vector) {
	sumCols(dgamma, dXSThatXS)
	sumCols(dbeta, dXS)
	xs, _, _ := unmat(XS)
	dxs, _, _ := unmat(dXS)
	hatxs, _, _ := unmat(hatXS)
	dhatxs, _, _ := unmat(dhatXS)
	dldxs, _, _ := unmat(dLdXS)
	// dL/dXS
	const eps = 0.00001
	ctx, dModel := b.context, b.dModel
	for i := range ctx {
		u := 0.0
		for j := range dModel {
			u += xs[i*dModel+j]
		}
		u /= float64(dModel)
		o2 := 0.0
		for j := range dModel {
			diff := xs[i*dModel+j] - u
			o2 += diff * diff
		}
		o2 = 1.0 / math.Sqrt(o2/float64(dModel)+eps)
		sumdhat := 0.0
		sumdhat2 := 0.0
		for j := range dModel {
			ij := i*dModel + j
			dhatxs[ij] = dxs[ij] * gamma[j]
			sumdhat += dhatxs[ij]
			sumdhat2 += dhatxs[ij] * hatxs[ij]
		}
		for j := range dModel {
			ij := i*dModel + j
			dx := (dhatxs[ij] - sumdhat/float64(dModel) - hatxs[ij]*sumdhat2/float64(dModel)) * o2
			dldxs[ij] += dx
		}
	}
}

func softmaxBackward(b *block, dQK, dS, S matrix) {
	scale := 1.0 / math.Sqrt(float64(b.dAttn))
	s, rows, cols := unmat(S)
	ds, _, _ := unmat(dS)
	dqk, _, _ := unmat(dQK)
	for i := range rows {
		sum := 0.0
		for j := range cols {
			ij := i*cols + j
			sum += ds[ij] * s[ij]
		}
		for j := range cols {
			ij := i*cols + j
			val := s[ij] * (ds[ij] - sum)
			dqk[ij] = val * scale
		}
	}
}
