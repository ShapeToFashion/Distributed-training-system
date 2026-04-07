// gradient_server.go
//
// PURPOSE: Implements the gRPC server that handles ALL communication
//          between master and workers.
//
// COMPLETE TRAINING FLOW:
// ───────────────────────
//   1. Workers register → cluster forms
//   2. Master assigns data shards to workers (via GetTask RPC)
//   3. Workers train on their shard → compute gradients
//   4. Workers send gradients to master (via SendGradients RPC)
//   5. Master AGGREGATES gradients: g_avg = (g1 + g2 + ... + gN) / N
//   6. Master UPDATES weights: W = W - lr * g_avg
//   7. Workers fetch new weights (via GetWeights RPC)
//   8. Repeat for next epoch
//   9. Master saves checkpoints every N steps

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"sync"
	"time"

	"distributed_llm/storage"

	pb "distributed_llm/proto"

	"google.golang.org/grpc"
)

// ──────────────────────────────────────────────────────────
// WORKER INFO — tracks each worker in the cluster
// ──────────────────────────────────────────────────────────

type WorkerInfo struct {
	ID            string
	Address       string
	LastHeartbeat time.Time
	IsAlive       bool
	AssignedShard string // Which data shard this worker is training on
}

// ──────────────────────────────────────────────────────────
// TRAINING CONFIG — controls the distributed training loop
// ──────────────────────────────────────────────────────────

type TrainingConfig struct {
	NumEpochs          int      // How many passes over the full dataset
	BatchSize          int      // Samples per training batch
	LearningRate       float32  // Step size for gradient descent
	CheckpointInterval int      // Save model every N steps
	ShardPaths         []string // Paths to dataset shards
}

// ──────────────────────────────────────────────────────────
// MASTER SERVER — the brain of the distributed system
// ──────────────────────────────────────────────────────────

type MasterServer struct {
	pb.UnimplementedTrainerServiceServer

	mu      sync.Mutex
	workers map[string]*WorkerInfo
	weights []float32

	// ── Gradient aggregation state ──
	// The master collects gradients from all workers for each step.
	// Once ALL alive workers have sent gradients, the master averages
	// them and updates the weights.
	gradientBuffer map[string][]float32 // worker_id → gradients
	lossBuffer     map[string]float64   // worker_id → batch loss (cleared each aggregate step)
	gradientCount  int                  // How many workers have sent gradients this step

	// ── Training state ──
	config       TrainingConfig
	currentEpoch int
	currentStep  int
	totalLoss    float64
	lossCount    int
	trainingDone bool

	// ── Task assignment ──
	shardAssignments map[string]string // worker_id → shard_path
}

func NewMasterServer(config TrainingConfig) *MasterServer {
	// Initialize model weights (2994 parameters for our simple model)
	// vocab_size=50, embed_dim=16, hidden_dim=32
	numParams := 50*16 + 16*32 + 32 + 32*50 + 50 // = 2994
	weights := make([]float32, numParams)
	for i := range weights {
		// Same initialization as the Python model (must match!)
		weights[i] = float32(math.Sin(float64(i)*0.01) * 0.1)
	}

	return &MasterServer{
		workers:          make(map[string]*WorkerInfo),
		weights:          weights,
		gradientBuffer:   make(map[string][]float32),
		lossBuffer:       make(map[string]float64),
		config:           config,
		shardAssignments: make(map[string]string),
	}
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: RegisterWorker
// ──────────────────────────────────────────────────────────

func (m *MasterServer) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.workers[req.WorkerId] = &WorkerInfo{
		ID:            req.WorkerId,
		Address:       req.Address,
		LastHeartbeat: time.Now(),
		IsAlive:       true,
	}

	// Assign a shard to this worker (round-robin)
	workerIndex := len(m.workers) - 1
	if len(m.config.ShardPaths) > 0 {
		shardIndex := workerIndex % len(m.config.ShardPaths)
		shard := m.config.ShardPaths[shardIndex]
		m.shardAssignments[req.WorkerId] = shard
		m.workers[req.WorkerId].AssignedShard = shard
		fmt.Printf("[MASTER] Assigned shard %s → %s\n", shard, req.WorkerId)
	}

	fmt.Printf("[MASTER] Worker registered: %s (address: %s)\n", req.WorkerId, req.Address)
	fmt.Printf("[MASTER] Total workers in cluster: %d\n", len(m.workers))

	return &pb.RegisterResponse{Status: "registered"}, nil
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: SendHeartbeat
// ──────────────────────────────────────────────────────────

func (m *MasterServer) SendHeartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if w, exists := m.workers[req.WorkerId]; exists {
		w.LastHeartbeat = time.Now()
		w.IsAlive = true
	}

	return &pb.HeartbeatResponse{Status: "alive"}, nil
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: GetTask
// ──────────────────────────────────────────────────────────
// Worker polls this to get its training task (which shard, batch size, etc.)
// This is how the master DISTRIBUTES work to the cluster.

func (m *MasterServer) GetTask(ctx context.Context, req *pb.TaskRequest) (*pb.TaskResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.trainingDone {
		return &pb.TaskResponse{HasTask: false}, nil
	}

	shard, exists := m.shardAssignments[req.WorkerId]
	if !exists {
		return &pb.TaskResponse{HasTask: false}, nil
	}

	return &pb.TaskResponse{
		ShardPath:    shard,
		BatchSize:    int32(m.config.BatchSize),
		LearningRate: m.config.LearningRate,
		Epoch:        int32(m.currentEpoch),
		HasTask:      true,
	}, nil
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: SendGradients  ★ THE MOST IMPORTANT PART ★
// ──────────────────────────────────────────────────────────
//
// GRADIENT AGGREGATION — this is how distributed training works:
//
//   1. Worker1 trains on shard1 → computes gradients g1
//   2. Worker2 trains on shard2 → computes gradients g2
//   3. Worker3 trains on shard3 → computes gradients g3
//   4. Worker4 trains on shard4 → computes gradients g4
//
//   5. Master receives all gradients
//   6. Master computes AVERAGE: g_avg = (g1 + g2 + g3 + g4) / 4
//   7. Master updates weights:  W_new = W_old - learning_rate * g_avg
//
// This is called SYNCHRONOUS SGD (Stochastic Gradient Descent).
// It's the same algorithm used by PyTorch DistributedDataParallel.

func (m *MasterServer) SendGradients(ctx context.Context, req *pb.GradientRequest) (*pb.GradientResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	fmt.Printf("[MASTER] Received gradients from %s (size: %d)\n", req.WorkerId, len(req.Gradients))

	// Store this worker's gradients and loss
	m.gradientBuffer[req.WorkerId] = req.Gradients
	m.lossBuffer[req.WorkerId] = req.GetLoss()
	m.gradientCount++

	// Count how many alive workers we expect gradients from
	aliveCount := 0
	for _, w := range m.workers {
		if w.IsAlive {
			aliveCount++
		}
	}

	// ── Check: have ALL workers sent their gradients? ──
	if m.gradientCount >= aliveCount && aliveCount > 0 {
		fmt.Printf("[MASTER] ★ All %d workers reported. Aggregating gradients...\n", aliveCount)
		m.aggregateAndUpdate()
	}

	return &pb.GradientResponse{Status: "received"}, nil
}

// aggregateAndUpdate averages all worker gradients and updates model weights.
//
// MATH:
//
//	g_avg[i] = (g1[i] + g2[i] + ... + gN[i]) / N    for each parameter i
//	W[i] = W[i] - learning_rate * g_avg[i]            gradient descent step
//
// This is the CORE of distributed training.
func (m *MasterServer) aggregateAndUpdate() {
	numWorkers := len(m.gradientBuffer)
	if numWorkers == 0 {
		return
	}

	numParams := len(m.weights)

	// Step 1: Sum all gradients element-wise
	avgGrad := make([]float32, numParams)
	for _, grads := range m.gradientBuffer {
		for i := 0; i < numParams && i < len(grads); i++ {
			avgGrad[i] += grads[i]
		}
	}

	// Step 2: Divide by number of workers to get average
	for i := range avgGrad {
		avgGrad[i] /= float32(numWorkers)
	}

	// Step 3: Update weights using gradient descent
	// W = W - lr * g_avg
	lr := m.config.LearningRate
	for i := range m.weights {
		m.weights[i] -= lr * avgGrad[i]
	}

	if len(m.lossBuffer) > 0 {
		var lossSum float64
		for _, v := range m.lossBuffer {
			lossSum += v
		}
		m.totalLoss += lossSum / float64(len(m.lossBuffer))
		m.lossCount++
	}

	m.currentStep++
	fmt.Printf("[MASTER] ✓ Step %d complete. Weights updated. (epoch %d)\n", m.currentStep, m.currentEpoch)

	// Step 4: Checkpoint if needed
	if m.config.CheckpointInterval > 0 && m.currentStep%m.config.CheckpointInterval == 0 {
		avgLoss := float64(0)
		if m.lossCount > 0 {
			avgLoss = m.totalLoss / float64(m.lossCount)
		}
		_, err := storage.SaveCheckpoint("storage", m.currentStep, m.currentEpoch, m.weights, avgLoss)
		if err != nil {
			fmt.Printf("[MASTER] Checkpoint error: %v\n", err)
		}
	}

	// Step 5: Check if epoch is complete
	if m.currentStep > 0 && m.currentStep%len(m.config.ShardPaths) == 0 {
		m.currentEpoch++
		fmt.Printf("[MASTER] ═══ Epoch %d complete ═══\n", m.currentEpoch-1)

		if m.currentEpoch >= m.config.NumEpochs {
			m.trainingDone = true
			fmt.Println("[MASTER] ★★★ TRAINING COMPLETE ★★★")

			// Save final checkpoint
			avgLoss := float64(0)
			if m.lossCount > 0 {
				avgLoss = m.totalLoss / float64(m.lossCount)
			}
			storage.SaveCheckpoint("storage", m.currentStep, m.currentEpoch, m.weights, avgLoss)
		}
	}

	// Reset gradient buffer for next step
	m.gradientBuffer = make(map[string][]float32)
	m.lossBuffer = make(map[string]float64)
	m.gradientCount = 0
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: GetWeights
// ──────────────────────────────────────────────────────────
// Workers call this after each step to get the updated model weights.
// This ensures all workers train with the SAME weights each step.

func (m *MasterServer) GetWeights(ctx context.Context, req *pb.WeightsRequest) (*pb.WeightsResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return &pb.WeightsResponse{Weights: m.weights}, nil
}

// ──────────────────────────────────────────────────────────
// gRPC METHOD: SaveCheckpoint
// ──────────────────────────────────────────────────────────

func (m *MasterServer) SaveCheckpoint(ctx context.Context, req *pb.CheckpointRequest) (*pb.CheckpointResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	avgLoss := float64(0)
	if m.lossCount > 0 {
		avgLoss = m.totalLoss / float64(m.lossCount)
	}

	path, err := storage.SaveCheckpoint("storage", int(req.Step), m.currentEpoch, m.weights, avgLoss)
	if err != nil {
		return &pb.CheckpointResponse{Status: "error: " + err.Error()}, nil
	}

	return &pb.CheckpointResponse{Status: "saved", Path: path}, nil
}

// ──────────────────────────────────────────────────────────
// HEALTH CHECKER — background goroutine
// ──────────────────────────────────────────────────────────

func (m *MasterServer) StartHealthChecker() {
	go func() {
		for {
			time.Sleep(10 * time.Second)

			m.mu.Lock()
			for id, w := range m.workers {
				if time.Since(w.LastHeartbeat) > 15*time.Second && w.IsAlive {
					w.IsAlive = false
					fmt.Printf("[MASTER] ⚠ Worker %s is DEAD (no heartbeat)\n", id)
				}
			}
			m.reassignDeadWorkerShardsLocked()
			m.mu.Unlock()
		}
	}()
}

// SaveWeightsToFile writes the current weights to a JSON file that
// the Python training script can read.
func (m *MasterServer) SaveWeightsToFile(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, err := json.Marshal(m.weights)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

// ──────────────────────────────────────────────────────────
// MAIN — starts the master server
// ──────────────────────────────────────────────────────────

func main() {
	// Training configuration
	config := TrainingConfig{
		NumEpochs:          20,    // 20 passes over the dataset
		BatchSize:          4,    // 4 lines per training batch
		LearningRate:       0.01, // Step size for weight updates
		CheckpointInterval: 5,    // Save every 5 steps
		ShardPaths: []string{
			"dataset/shard1.txt",
			"dataset/shard2.txt",
			"dataset/shard3.txt",
			"dataset/shard4.txt",
		},
	}

	// Try to resume from checkpoint
	ckpt, err := storage.LoadLatestCheckpoint("storage")
	var master *MasterServer
	if err == nil {
		fmt.Printf("[MASTER] Resuming from checkpoint step %d\n", ckpt.Step)
		master = NewMasterServer(config)
		master.weights = ckpt.Weights
		master.currentStep = ckpt.Step
		master.currentEpoch = ckpt.Epoch
	} else {
		master = NewMasterServer(config)
		fmt.Println("[MASTER] Starting fresh (no checkpoint found)")
	}

	// Listen on TCP port 50051
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("[MASTER] Failed to listen: %v", err)
	}

	server := grpc.NewServer()
	pb.RegisterTrainerServiceServer(server, master)
	master.StartHealthChecker()

	fmt.Println("═══════════════════════════════════════════════════")
	fmt.Println("  DISTRIBUTED LLM TRAINING — MASTER NODE")
	fmt.Printf("  Config: %d epochs, batch=%d, lr=%.4f\n",
		config.NumEpochs, config.BatchSize, config.LearningRate)
	fmt.Printf("  Model: %d parameters\n", len(master.weights))
	fmt.Printf("  Shards: %d\n", len(config.ShardPaths))
	fmt.Println("  Listening on port 50051")
	fmt.Println("  Waiting for workers to connect...")
	fmt.Println("═══════════════════════════════════════════════════")

	if err := server.Serve(lis); err != nil {
		log.Fatalf("[MASTER] Failed to serve: %v", err)
	}
}
