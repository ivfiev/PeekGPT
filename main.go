package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func main() {
	trainingSet := generateCopyTask([]rune("abc"), 3, 1000)
	validationSet := generateCopyTask([]rune("abc"), 4, 100)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	seed := time.Now().UnixNano()
	t := train(64, 5, 7,
		trainingSet, validationSet,
		10000, 16, 16, 0.0001, 0.00001, seed)
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
	// validation + training set, use that shuffle func
	// fix graphics/peek
	// try to make compatible with seqs
}
