package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	seed := time.Now().UnixNano()
	t := newT(11, 3, 8, ReLU)
	t.train([]rune("aa|bb|aa|bb|aa|bb|"), seed, 5000, 0.01, 0.0001)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
	// microbatch, avx
}
