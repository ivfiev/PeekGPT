package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func main() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateCopyTask([]rune("0123456789"), 4, 1000, rng)
	validationSet := generateCopyTask([]rune("0123456789"), 4, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 9, 2,
		trainingSet, validationSet,
		50000, 32, 16, 0.0001, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
	// fix graphics/peek. triple-check printAttention, is S not inverted?
	// more info during training?
	// maybe spsa K
	// scale & seqs
}
