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
	t := train(27, 32,
		[]rune("the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog "),
		200000, 12, 16, 0.0005, 0.001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
	// learnable embeds
}
