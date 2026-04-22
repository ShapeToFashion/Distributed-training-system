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
	"os"
	"os/exec"
	"path/filepath"
	"time"

	pb "distributed_llm/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func pythonExecutable() string {
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
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

func main() {
	// ── Parse command-line flags ──
	workerID := flag.String("id", "worker1", "Unique worker ID")
	masterAddr := flag.String("master", "localhost:50051", "Master server address")
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
		),
	)
	if err != nil {
		log.Fatalf("[WORKER %s] Failed to connect: %v", *workerID, err)
	}
	defer conn.Close()

	client := pb.NewTrainerServiceClient(conn)

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
	go StartHeartbeat(client, *workerID)

	// ── Step 4: Training loop ──
	fmt.Printf("[WORKER %s] Entering training loop...\n", *workerID)

	for {
		// 4a) Ask master for a task
		task, err := client.GetTask(context.Background(), &pb.TaskRequest{
			WorkerId: *workerID,
		})
		if err != nil {
			fmt.Printf("[WORKER %s] GetTask error: %v (retrying...)\n", *workerID, err)
			time.Sleep(3 * time.Second)
			continue
		}

		if !task.HasTask {
			fmt.Printf("[WORKER %s] No more tasks — training complete!\n", *workerID)
			break
		}

		fmt.Printf("[WORKER %s] Got task: shard=%s, batch=%d, lr=%.4f, epoch=%d\n",
			*workerID, task.ShardPath, task.BatchSize, task.LearningRate, task.Epoch)

		// 4b) Get latest weights from master
		weightsResp, err := client.GetWeights(context.Background(), &pb.WeightsRequest{})
		if err != nil {
			fmt.Printf("[WORKER %s] GetWeights error: %v\n", *workerID, err)
			time.Sleep(2 * time.Second)
			continue
		}

		// 4c) Save weights to temp file for Python to read
		weightsFile := filepath.Join(os.TempDir(), fmt.Sprintf("weights_%s.json", *workerID))
		weightsJSON, err := json.Marshal(weightsResp.Weights)
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
		gradients, loss, err := runPythonTrainer(task.ShardPath, int(task.BatchSize), task.LearningRate, weightsFile)
		if err != nil {
			fmt.Printf("[WORKER %s] Python training error: %v\n", *workerID, err)
			time.Sleep(2 * time.Second)
			continue
		}

		fmt.Printf("[WORKER %s] Training done — loss: %.6f, gradients: %d params\n",
			*workerID, loss, len(gradients))

		// 4e) Send gradients to master
		_, err = client.SendGradients(context.Background(), &pb.GradientRequest{
			WorkerId:  *workerID,
			Gradients: gradients,
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
func runPythonTrainer(shardPath string, batchSize int, lr float32, weightsFile string) ([]float32, float64, error) {
	cmd := exec.Command(
		pythonExecutable(),
		"training/train.py",
		fmt.Sprintf("--data=%s", shardPath),
		fmt.Sprintf("--batch_size=%d", batchSize),
		fmt.Sprintf("--lr=%f", lr),
		fmt.Sprintf("--weights=%s", weightsFile),
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
