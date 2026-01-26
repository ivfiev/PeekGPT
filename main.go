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
	ctx := 4
	seed := time.Now().UnixNano()
	t := trainModel(ctx, []rune("112112112"), 5, seed, 35000, 0.05, 0.0001)
	for {
		fmt.Printf("Enter context, up to %d chars: ", ctx)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
}
