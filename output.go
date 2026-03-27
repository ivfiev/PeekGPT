package main

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

func (m *model) printAttention() {
	for bi, b := range m.blocks {
		fmt.Printf("\nBlock %d\n", 1+bi)
		for i := range m.prompt {
			for a := range b.attn {
				fmt.Printf("%c ", m.prompt[i])
				for j := range m.prompt {
					fg, bg := 0, 0
					fg = int(255 * b.S[a].At(i, j))
					fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm███\x1b[0m", fg, fg, fg, bg, bg, bg)
				}
				fmt.Printf("     ")
			}
			println()
		}
		fmt.Printf("   ")
		for range b.attn {
			for _, c := range m.prompt {
				fmt.Printf("%c  ", c)
			}
			fmt.Printf("       ")
		}
		println()
		if bi < len(m.blocks)-1 {
			println()
		}
	}
}

func (m *model) calcHeatmap(x int, As []matrix) []vector {
	rgbs := make([]vector, 0, len(As))
	maxVal := math.Inf(-1)
	minVal := math.Inf(1)
	for _, A := range As {
		d, _, c := unmat(A) // matRow or sth, srs...
		X := d[x*c : x*c+c]
		for i := range X {
			maxVal = max(maxVal, X[i])
			minVal = min(minVal, X[i])
		}
	}
	for _, A := range As {
		rgb := make(vector, m.dModel)
		d, _, c := unmat(A)
		x := d[x*c : x*c+c]
		for i := range rgb {
			rgb[i] = x[i] / (maxVal - minVal)
		}
		rgbs = append(rgbs, rgb)
	}
	return rgbs
}

func (m *model) printHeatmap(x int) {
	heatmaps := [][]vector{}
	As := make([]matrix, 5*len(m.blocks))
	for i, b := range m.blocks {
		As[0+i*5] = b.XS0
		As[1+i*5] = b.P
		As[2+i*5] = b.R0
		As[3+i*5] = b.H
		As[4+i*5] = b.R1
	}
	heatmaps = append(heatmaps, m.calcHeatmap(x, As))
	width := 2
	if m.dModel > 64 {
		width = 1
	}
	strip := strings.Repeat("█", width)
	for _, heatmap := range heatmaps {
		println()
		for label, rgb := range heatmap {
			for i, x := range rgb {
				red, blue, bg := 0, 0, 0
				if x < 0 {
					blue = int(-rgb[i] * 255)
				} else {
					red = int(rgb[i] * 255)
				}
				fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm%s\x1b[0m", red, 0, blue, bg, bg, bg, strip)
			}
			blockIx := label / 5
			switch label % 5 {
			case 0:
				fmt.Printf("  Block #%d, \"%s\" - ", 1+blockIx, tokenHighlight(m.prompt, x))
				printStats(m.blocks[blockIx].XS0, x)
			case 1:
				fmt.Printf("  Attention Δ")
			case 2:
				fmt.Printf("  Post-attention - ")
				printStats(m.blocks[blockIx].R0, x)
			case 3:
				fmt.Printf("  MLP Δ")
			case 4:
				fmt.Printf("  Post-MLP")
				if blockIx == len(m.blocks)-1 {
					fmt.Printf(", final output")
				}
				fmt.Printf(" - ")
				printStats(m.blocks[blockIx].R1, x)
			}
			println()
		}
		if 1+x != len(heatmaps) {
			println()
		}
	}
}

func (m *model) printNextTokenProbs(i int) {
	type pair struct {
		token rune
		prob  float64
	}
	pairs := make([]pair, len(m.vocab))
	s, _, c := unmat(m.S)
	for i, p := range s[i*c : i*c+c] {
		pairs[i].token = m.vocab[i]
		pairs[i].prob = p
	}
	slices.SortFunc(pairs, func(p1, p2 pair) int {
		if p1.prob < p2.prob {
			return 1
		}
		return -1
	})
	fmt.Printf("Next token probabilities:\n")
	for i, pair := range pairs {
		fmt.Printf("%q -> %.6f,  ", pair.token, pair.prob)
		if (i+1)%4 == 0 || (i+1) == len(pairs) {
			println()
		}
	}
	println()
}

func printStats(A matrix, x int) {
	a, _, c := unmat(A)
	v := a[x*c : x*c+c]
	u, o2 := meanStd(v)
	fmt.Printf("(%.3f, %.3f)\n", u, o2)
}

func tokenHighlight(prompt []rune, i int) string {
	hl := append([]rune(nil), prompt[:i]...)
	hl = append(hl, '[')
	hl = append(hl, prompt[i])
	hl = append(hl, ']')
	hl = append(hl, prompt[i+1:]...)
	return string(hl)
}
