import time

# Sum of squares from 1 to 10 million
start = time.time()
total = 0
for i in range(1, 10_000_001):
    total += i * i
elapsed = time.time() - start
print(f"Python -> Sum: {total} | Time: {elapsed:.6f}s")
