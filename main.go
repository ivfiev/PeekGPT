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
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	seed := time.Now().UnixNano()
	t := train(64, 4, 32,
		[]rune("|aa|bb|cc|aa|bb|cc|aa|bb|cc|aa|bb|cc|aa|bb|cc|"),
		100000, 4, 16, 0.00025, 0.0001, seed)
	t.generate([]rune("|aa|bb|c"), 192)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
	// multi-blocks, task modes
}
