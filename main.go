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
	trainingSet := generateCopyTask([]rune("abc"), 3, 1000, rng)
	validationSet := generateCopyTask([]rune("abc"), 3, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 5, 7,
		trainingSet, validationSet,
		30000, 32, 16, 0.0001, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		// t.peek([]rune(strings.TrimRight(input, "\n\r")))
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
	// multi-blocks
	// fix graphics/peek
	// seqs
}
