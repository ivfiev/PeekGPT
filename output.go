package main

import (
	"fmt"
	"log"
	"math"
)

func (t *transformer) printAttention() {
	for bi, b := range t.blocks {
		for i := range t.prompt {
			fmt.Printf("%c ", t.prompt[i])
			for j := range t.prompt {
				fg, bg := 0, 0
				fg = int(255 * b.S.At(i, j))
				fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm███\x1b[0m", fg, fg, fg, bg, bg, bg)
			}
			println()
		}
		fmt.Printf("   ")
		for _, c := range t.prompt {
			fmt.Printf("%c  ", c)
		}
		println()
		if bi < len(t.blocks)-1 {
			println()
		}
	}
}

func (t *transformer) calcHeatmap(x int, As []matrix) []vector {
	target := make(vector, t.dModel)
	d, _, c, s := unmat(t.L)
	_, rmix := rowMax(d[x*s : x*s+c])
	for i := range t.dModel {
		target[i] = t.linear.At(i, rmix)
	}
	maxProd := math.Inf(-1)
	minProd := math.Inf(1)
	for _, A := range As {
		d, _, c, s := unmat(A) // matRow or sth, srs...
		x := d[x*s : x*s+c]
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
		rgb := make(vector, t.dModel)
		d, _, c, s := unmat(A)
		x := d[x*s : x*s+c]
		for i := range rgb {
			rgb[i] = (x[i] * target[i]) / (maxProd - minProd)
		}
		rgbs = append(rgbs, rgb)
	}
	return rgbs
}

func (t *transformer) printHeatmap(xs []int) {
	heatmaps := [][]vector{}
	As := make([]matrix, 5*len(t.blocks))
	for _, x := range xs {
		for i, b := range t.blocks {
			As[0+i*5] = b.XS0
			As[1+i*5] = b.P
			As[2+i*5] = b.R0
			As[3+i*5] = b.H
			As[4+i*5] = b.R1
		}
		heatmaps = append(heatmaps, t.calcHeatmap(x, As))
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
				fmt.Printf("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm██\x1b[0m", red, 0, blue, bg, bg, bg)
			}
			blockIx := label / 5
			switch label % 5 {
			case 0:
				fmt.Printf("  Block #%d, \"%s\"\n", 1+blockIx, tokenHighlight(t.prompt, xs[x]))
			case 1:
				fmt.Printf("  Attention Δ")
			case 2:
				fmt.Printf("  Post-attention\n")
			case 3:
				fmt.Printf("  MLP Δ")
			case 4:
				final := ""
				if blockIx == len(t.blocks)-1 {
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
