package main

import (
	"fmt"
	"time"
)

func main() {
	// Sum of squares from 1 to 10 million
	start := time.Now()
	sum := 0
	for i := 1; i <= 10_000_000; i++ {
		sum += i * i
	}
	elapsed := time.Since(start)
	fmt.Printf("Go  -> Sum: %d | Time: %v\n", sum, elapsed)
}
