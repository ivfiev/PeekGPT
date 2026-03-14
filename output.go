package main

import (
	"fmt"
	"log"
	"math"
)

func (m *model) printAttention() {
	for bi, b := range m.blocks {
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
	target := make(vector, m.dModel)
	d, _, c := unmat(m.L)
	_, rmix := rowMax(d[x*c : x*c+c])
	for i := range m.dModel {
		target[i] = m.unembed.At(i, rmix)
	}
	maxProd := math.Inf(-1)
	minProd := math.Inf(1)
	for _, A := range As {
		d, _, c := unmat(A) // matRow or sth, srs...
		x := d[x*c : x*c+c]
		if len(x) != len(target) {
			log.Panicf("Incompatible heatmaps, %d != %d", len(x), len(target))
		}
		for i := range x {
			maxProd = max(maxProd, x[i]*target[i])
			minProd = min(minProd, x[i]*target[i])
		}
	}
	rgbs := make([]vector, 0, len(As))
	for _, A := range As {
		rgb := make(vector, m.dModel)
		d, _, c := unmat(A)
		x := d[x*c : x*c+c]
		for i := range rgb {
			rgb[i] = (x[i] * target[i]) / (maxProd - minProd)
		}
		rgbs = append(rgbs, rgb)
	}
	return rgbs
}

func (m *model) printHeatmap(xs []int) {
	heatmaps := [][]vector{}
	As := make([]matrix, 5*len(m.blocks))
	for _, x := range xs {
		for i, b := range m.blocks {
			As[0+i*5] = b.XS0
			As[1+i*5] = b.P
			As[2+i*5] = b.R0
			As[3+i*5] = b.H
			As[4+i*5] = b.R1
		}
		heatmaps = append(heatmaps, m.calcHeatmap(x, As))
	}
	for x, heatmap := range heatmaps {
		println()
		for label, rgb := range heatmap {
			for i, x := range rgb {
				red, blue, bg := 0, 0, 0
				if x < 0 {
					blue = int(-rgb[i] * 255)
				} else {
					red = int(rgb[i] * 255)
				}
				fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm█\x1b[0m", red, 0, blue, bg, bg, bg)
			}
			blockIx := label / 5
			switch label % 5 {
			case 0:
				fmt.Printf("  Block #%d, \"%s\"\n", 1+blockIx, tokenHighlight(m.prompt, xs[x]))
			case 1:
				fmt.Printf("  Attention Δ")
			case 2:
				fmt.Printf("  Post-attention\n")
			case 3:
				fmt.Printf("  MLP Δ")
			case 4:
				final := ""
				if blockIx == len(m.blocks)-1 {
					final = ", final output"
				}
				fmt.Printf("  Post-MLP%s\n", final)
			}
			println()
		}
		println()
	}
}

func tokenHighlight(prompt []rune, i int) string {
	hl := append([]rune(nil), prompt[:i]...)
	hl = append(hl, '[')
	hl = append(hl, prompt[i])
	hl = append(hl, ']')
	hl = append(hl, prompt[i+1:]...)
	return string(hl)
}
