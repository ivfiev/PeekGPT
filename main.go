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
	ctx := 16
	seed := time.Now().UnixNano()
	t := trainModel(ctx, []rune("the more you buy the more you save the more you buy the more you save "), 4, seed, 5000, 0.01, 0.0001)
	for {
		fmt.Printf("Enter context, up to %d chars: ", ctx)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
}
