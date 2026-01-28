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
	ctx := 8
	seed := time.Now().UnixNano()
	t := trainModel(ctx, []rune("aa|bb|aa|bb|aa|bb|"), 4, seed, 10000, 0.01, 0.0001)
	for {
		fmt.Printf("Enter context, up to %d chars: ", ctx)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
}
