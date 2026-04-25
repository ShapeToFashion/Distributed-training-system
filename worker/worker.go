// worker.go
//
// PURPOSE: This is the worker node — the "muscle" of the distributed system.
//
// COMPLETE WORKER FLOW:
// ─────────────────────
// 1. Connect to master via gRPC
// 2. Register (get assigned a data shard)
// 3. Start heartbeat in background goroutine
// 4. TRAINING LOOP:
//    a) Call GetTask() → receive shard path + batch size + learning rate
//    b) Call GetWeights() → receive latest model weights from master
//    c) Save weights to temp file → run Python train.py → read gradients from stdout
//    d) Call SendGradients() → send gradients to master
//    e) Wait for next step (master aggregates and updates weights)
//    f) Repeat
//
// HOW TO RUN:
// ───────────
// Terminal 1: go run ./master/
// Terminal 2: go run ./worker/ -id=worker1
// Terminal 3: go run ./worker/ -id=worker2

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	pb "distributed_llm/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	_ "google.golang.org/grpc/encoding/gzip"
)

// masterDefault is the hosted master address.
// After running "fly deploy", replace this with your Fly.io TCP address.
const masterDefault = "localhost:50051"

func pythonExecutable() string {
	candidates := []string{
		".venv/Scripts/python", // Windows venv
		".venv/bin/python",     // Linux/Mac venv
		"python3",
		"python",
		"py",
	}
	for _, c := range candidates {
		if p, err := exec.LookPath(c); err == nil {
			return p
		}
	}
	return "python"
}

// TrainResult holds the JSON output from the Python training script.
type TrainResult struct {
	Gradients     []float32 `json:"gradients"`
	Loss          float64   `json:"loss"`
	NumParameters int       `json:"num_parameters"`
	Error         string    `json:"error"`
}

type MetricsTracker struct {
	mu               sync.Mutex
	throughput       float32
	lastBatchSeconds float32
	lastCPUIdle      uint64
	lastCPUTotal     uint64
	hasCPUSample     bool
}

func (m *MetricsTracker) Snapshot() *pb.WorkerMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	cpuPercent := m.readCPUPercentLocked()
	gpuUtil, gpuMemFree := readGPUStats()
	return &pb.WorkerMetrics{
		CpuPercent:     cpuPercent,
		RamFreeGb:      readMemAvailableGB(),
		GpuUtilPercent: gpuUtil,
		GpuMemFreeGb:   gpuMemFree,
		Throughput:     m.throughput,
	}
}

func (m *MetricsTracker) UpdateThroughput(samples int, elapsed time.Duration) {
	if elapsed <= 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	current := float32(float64(samples) / elapsed.Seconds())
	if m.throughput == 0 {
		m.throughput = current
	} else {
		// EWMA smoothing to reduce oscillation.
		m.throughput = 0.7*m.throughput + 0.3*current
	}
	m.lastBatchSeconds = float32(elapsed.Seconds())
}

func readMemAvailableGB() float32 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0
		}
		return float32(kb / (1024 * 1024))
	}
	return 0
}

func (m *MetricsTracker) readCPUPercentLocked() float32 {
	idle, total, ok := readCPUStat()
	if !ok || total == 0 {
		return 0
	}
	if !m.hasCPUSample {
		m.lastCPUIdle = idle
		m.lastCPUTotal = total
		m.hasCPUSample = true
		return 0
	}
	deltaIdle := idle - m.lastCPUIdle
	deltaTotal := total - m.lastCPUTotal
	m.lastCPUIdle = idle
	m.lastCPUTotal = total
	if deltaTotal == 0 {
		return 0
	}
	util := (1.0 - float64(deltaIdle)/float64(deltaTotal)) * 100.0
	if util < 0 {
		util = 0
	}
	if util > 100 {
		util = 100
	}
	return float32(util)
}

func readCPUStat() (idle uint64, total uint64, ok bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) == 0 {
		return 0, 0, false
	}
	fields := strings.Fields(lines[0])
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0, 0, false
	}
	var values []uint64
	for _, f := range fields[1:] {
		v, err := strconv.ParseUint(f, 10, 64)
		if err != nil {
			return 0, 0, false
		}
		values = append(values, v)
		total += v
	}
	// idle + iowait
	idle = values[3]
	if len(values) > 4 {
		idle += values[4]
	}
	return idle, total, true
}

func readGPUStats() (gpuUtilPercent float32, gpuMemFreeGB float32) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return 0, 0
	}
	line := strings.TrimSpace(string(output))
	if line == "" {
		return 0, 0
	}
	first := strings.Split(line, "\n")[0]
	parts := strings.Split(first, ",")
	if len(parts) < 2 {
		return 0, 0
	}
	util, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 32)
	if err != nil {
		return 0, 0
	}
	memMB, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return 0, 0
	}
	return float32(util), float32(memMB / 1024.0)
}

func generateWorkerID() string {
	host, err := os.Hostname()
	if err != nil || host == "" {
		host = "worker"
	}
	var b strings.Builder
	for _, c := range strings.ToLower(host) {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			b.WriteRune(c)
		}
	}
	safe := b.String()
	if safe == "" {
		safe = "worker"
	}
	if len(safe) > 12 {
		safe = safe[:12]
	}
	return fmt.Sprintf("%s-%04d", safe, rand.Intn(10000))
}

func main() {
	// ── Parse command-line flags ──
	workerID := flag.String("id", generateWorkerID(), "Unique worker ID")
	masterAddr := flag.String("master", masterDefault, "Master server address")
	testOnly := flag.Bool("test", false, "Only test connection, then exit")
	flag.Parse()

	fmt.Printf("[WORKER %s] Starting up...\n", *workerID)

	// ── Step 1: Connect to master ──
	conn, err := grpc.NewClient(
		*masterAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(1024*1024*200),
			grpc.MaxCallSendMsgSize(1024*1024*200),
			grpc.UseCompressor("gzip"),
		),
	)
	if err != nil {
		log.Fatalf("[WORKER %s] Failed to connect: %v", *workerID, err)
	}
	defer conn.Close()

	client := pb.NewTrainerServiceClient(conn)
	metrics := &MetricsTracker{}

	// ── Step 2: Register with master ──
	resp, err := client.RegisterWorker(context.Background(), &pb.RegisterRequest{
		WorkerId: *workerID,
		Address:  "localhost",
	})
	if err != nil {
		log.Fatalf("[WORKER %s] Registration failed: %v", *workerID, err)
	}
	fmt.Printf("[WORKER %s] Registration: %s\n", *workerID, resp.Status)

	if *testOnly {
		fmt.Printf("[WORKER %s] Connection test PASSED — connected to master at %s\n", *workerID, *masterAddr)
		return
	}

	// ── Step 3: Start heartbeat in background ──
	go StartHeartbeat(client, *workerID, metrics.Snapshot)

	// ── Step 4: Training loop ──
	fmt.Printf("[WORKER %s] Entering training loop...\n", *workerID)

	for {
		// 4a) Ask master for a task
		task, err := client.GetTask(context.Background(), &pb.TaskRequest{
			WorkerId: *workerID,
			Metrics:  metrics.Snapshot(),
		})
		if err != nil {
			fmt.Printf("[WORKER %s] GetTask error: %v (retrying...)\n", *workerID, err)
			time.Sleep(3 * time.Second)
			continue
		}

		if !task.HasTask {
			// The master can temporarily withhold a task while stabilizing a round.
			// Keep polling instead of exiting to avoid dropping a worker.
			fmt.Printf("[WORKER %s] No task assigned right now — waiting for next scheduling round...\n", *workerID)
			time.Sleep(2 * time.Second)
			continue
		}

		fmt.Printf("[WORKER %s] Got task: shard=%s, size=%s, batch=%d, score=%.3f, part=%d/%d, lr=%.4f, epoch=%d\n",
			*workerID, task.ShardPath, task.TaskSize, task.BatchSize, task.CapacityScore,
			task.PartitionIndex, task.PartitionCount, task.LearningRate, task.Epoch)

		// 4b) Get latest weights from master
		weightsResp, err := client.GetWeights(context.Background(), &pb.WeightsRequest{})
		if err != nil {
			fmt.Printf("[WORKER %s] GetWeights error: %v\n", *workerID, err)
			time.Sleep(2 * time.Second)
			continue
		}

		// 4c) Save weights to temp file for Python to read
		weightsFile := filepath.Join(os.TempDir(), fmt.Sprintf("weights_%s.json", *workerID))
		weightsJSON, err := json.Marshal(pb.UnpackF16(weightsResp.Weights))
		if err != nil {
			fmt.Printf("[WORKER %s] Failed to marshal weights: %v\n", *workerID, err)
			continue
		}
		if err := os.WriteFile(weightsFile, weightsJSON, 0600); err != nil {
			fmt.Printf("[WORKER %s] Failed to write weights file: %v\n", *workerID, err)
			continue
		}

		// 4d) Run Python training script
		fmt.Printf("[WORKER %s] Launching Python trainer...\n", *workerID)
		stepStart := time.Now()
		gradients, loss, err := runPythonTrainer(
			task.ShardPath,
			int(task.BatchSize),
			task.LearningRate,
			weightsFile,
			int(task.PartitionIndex),
			int(task.PartitionCount),
		)
		if err != nil {
			fmt.Printf("[WORKER %s] Python training error: %v\n", *workerID, err)
			time.Sleep(2 * time.Second)
			continue
		}
		metrics.UpdateThroughput(int(math.Max(1, float64(task.BatchSize))), time.Since(stepStart))

		fmt.Printf("[WORKER %s] Training done — loss: %.6f, gradients: %d params\n",
			*workerID, loss, len(gradients))

		// 4e) Send gradients to master
		_, err = client.SendGradients(context.Background(), &pb.GradientRequest{
			WorkerId:  *workerID,
			Gradients: pb.PackInt8(gradients),
			Loss:      loss,
		})
		if err != nil {
			fmt.Printf("[WORKER %s] SendGradients error: %v\n", *workerID, err)
		}

		// 4f) Small delay before next iteration (let master aggregate)
		time.Sleep(2 * time.Second)
	}

	fmt.Printf("[WORKER %s] Shutting down.\n", *workerID)
}

// runPythonTrainer executes the Python training script and parses its output.
//
// The Go worker calls Python because:
//   - Python has the best ML ecosystem (NumPy, PyTorch, etc.)
//   - Go is great for networking/concurrency (master + worker management)
//   - This is how real systems work: Go/C++ for infra, Python for ML
//
// Communication: Go writes weights to a JSON file, Python reads them,
// trains, and writes gradients as JSON to stdout.
func runPythonTrainer(shardPath string, batchSize int, lr float32, weightsFile string, partitionIndex int, partitionCount int) ([]float32, float64, error) {
	cmd := exec.Command(
		pythonExecutable(),
		"training/train.py",
		fmt.Sprintf("--data=%s", shardPath),
		fmt.Sprintf("--batch_size=%d", batchSize),
		fmt.Sprintf("--lr=%f", lr),
		fmt.Sprintf("--weights=%s", weightsFile),
		fmt.Sprintf("--partition_index=%d", partitionIndex),
		fmt.Sprintf("--partition_count=%d", partitionCount),
	)
	cmd.Stderr = os.Stderr

	// Capture stdout (JSON result) and stderr (logs)
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, 0, fmt.Errorf("python error: %s", string(exitErr.Stderr))
		}
		return nil, 0, fmt.Errorf("exec error: %w", err)
	}

	// Parse JSON output from Python
	var result TrainResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, 0, fmt.Errorf("JSON parse error: %w (output: %s)", err, string(output))
	}

	if result.Error != "" {
		return nil, 0, fmt.Errorf("training error: %s", result.Error)
	}

	return result.Gradients, result.Loss, nil
}
