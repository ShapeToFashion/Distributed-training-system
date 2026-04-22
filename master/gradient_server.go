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

// gradient_server.go (UPDATED FOR CNN)

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"os"
	"sort"
	"sync"
	"time"

	pb "distributed_llm/proto"

	"google.golang.org/grpc"
)

// ───────────── Worker Info ─────────────
type WorkerInfo struct {
	ID            string
	Address       string
	LastHeartbeat time.Time
	IsAlive       bool
	AssignedShard string
	Metrics       pb.WorkerMetrics
}

// ───────────── Config ─────────────
type TrainingConfig struct {
	NumEpochs          int
	BatchSize          int
	LearningRate       float32
	CheckpointInterval int
	ShardPaths         []string
}

// ───────────── Master ─────────────
type MasterServer struct {
	pb.UnimplementedTrainerServiceServer

	mu      sync.Mutex
	workers map[string]*WorkerInfo
	weights []float32

	gradientBuffer map[string][]float32
	lossBuffer     map[string]float64
	gradientCount  int

	config           TrainingConfig
	currentEpoch     int
	currentStep      int
	trainingDone     bool
	shardAssignments map[string]string
	nextShardIndex   int
	roundPartitions  map[string]int
	roundPartCount   int
}

// ───────────── Load weights ─────────────
func loadWeights(path string) []float32 {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}

	var weights []float32
	json.Unmarshal(data, &weights)
	return weights
}

func NewMasterServer(config TrainingConfig, weights []float32) *MasterServer {
	return &MasterServer{
		workers:          make(map[string]*WorkerInfo),
		weights:          weights,
		gradientBuffer:   make(map[string][]float32),
		lossBuffer:       make(map[string]float64),
		config:           config,
		shardAssignments: make(map[string]string),
		roundPartitions:  make(map[string]int),
	}
}

// ───────────── Register Worker ─────────────
func (m *MasterServer) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.workers[req.WorkerId] = &WorkerInfo{
		ID:            req.WorkerId,
		Address:       req.Address,
		LastHeartbeat: time.Now(),
		IsAlive:       true,
	}
	fmt.Printf("[MASTER] Registered worker: %s\n", req.WorkerId)
	return &pb.RegisterResponse{Status: "registered"}, nil
}

func (m *MasterServer) SendHeartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if w, ok := m.workers[req.WorkerId]; ok {
		w.LastHeartbeat = time.Now()
		w.IsAlive = true
		if req.Metrics != nil {
			w.Metrics = *req.Metrics
		}
	}
	return &pb.HeartbeatResponse{Status: "alive"}, nil
}

func (m *MasterServer) GetTask(ctx context.Context, req *pb.TaskRequest) (*pb.TaskResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.trainingDone {
		return &pb.TaskResponse{HasTask: false}, nil
	}
	worker, ok := m.workers[req.WorkerId]
	if !ok || !worker.IsAlive {
		return &pb.TaskResponse{HasTask: false}, nil
	}
	if req.Metrics != nil {
		worker.Metrics = *req.Metrics
	}

	shard := m.pickNextShardLocked()
	if shard == "" {
		return &pb.TaskResponse{HasTask: false}, nil
	}
	worker.AssignedShard = shard

	capacityScore := m.computeCapacityScoreLocked(req.WorkerId)
	taskSize, batchSize := m.pickTaskSize(capacityScore)
	partitionIndex, partitionCount, ok := m.workerPartitionLocked(req.WorkerId)
	if !ok {
		// New workers wait until next aggregation round so partitioning remains stable.
		return &pb.TaskResponse{HasTask: false}, nil
	}

	fmt.Printf("[MASTER] Task for %s -> shard=%s size=%s batch=%d score=%.3f\n",
		req.WorkerId, shard, taskSize, batchSize, capacityScore)

	return &pb.TaskResponse{
		ShardPath:    shard,
		BatchSize:    int32(batchSize),
		LearningRate: m.config.LearningRate,
		Epoch:        int32(m.currentEpoch),
		HasTask:      true,
		TaskSize:     taskSize,
		CapacityScore: float32(capacityScore),
		PartitionIndex: int32(partitionIndex),
		PartitionCount: int32(partitionCount),
	}, nil
}

// ───────────── Send Gradients ─────────────
func (m *MasterServer) SendGradients(ctx context.Context, req *pb.GradientRequest) (*pb.GradientResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.roundPartitions) > 0 {
		if _, ok := m.roundPartitions[req.WorkerId]; !ok {
			// Worker belongs to next round; ignore stale/out-of-round gradients.
			return &pb.GradientResponse{Status: "waiting_next_round"}, nil
		}
	}

	if _, exists := m.gradientBuffer[req.WorkerId]; !exists {
		m.gradientCount++
	}
	m.gradientBuffer[req.WorkerId] = req.Gradients
	m.lossBuffer[req.WorkerId] = req.Loss
	fmt.Printf("[MASTER] Received gradients from %s: grad_len=%d loss=%.6f\n", req.WorkerId, len(req.Gradients), req.Loss)

	if m.gradientCount >= m.expectedWorkersForRoundLocked() {
		m.aggregateAndUpdate()
	}

	return &pb.GradientResponse{Status: "ok"}, nil
}

// ───────────── Aggregate ─────────────
func (m *MasterServer) aggregateAndUpdate() {
	numParams := len(m.weights)
	avgGrad := make([]float32, numParams)
	fmt.Printf("[MASTER] Aggregating gradients: workers=%d weight_len=%d\n", len(m.gradientBuffer), len(m.weights))

	for workerID, grads := range m.gradientBuffer {
		if len(grads) != numParams {
			fmt.Printf("[MASTER] WARNING: gradient length mismatch for %s (got=%d expected=%d)\n", workerID, len(grads), numParams)
		}
		for i := 0; i < numParams && i < len(grads); i++ {
			avgGrad[i] += grads[i]
		}
	}

	for i := range avgGrad {
		avgGrad[i] /= float32(len(m.gradientBuffer))
	}

	for i := range m.weights {
		if i < len(avgGrad) {
			m.weights[i] -= m.config.LearningRate * avgGrad[i]
		}
	}

	// SAVE UPDATED WEIGHTS
	m.SaveWeightsToFile("weights.json")

	m.gradientBuffer = make(map[string][]float32)
	m.lossBuffer = make(map[string]float64)
	m.gradientCount = 0
	m.roundPartitions = make(map[string]int)
	m.roundPartCount = 0

	fmt.Println("[MASTER] Weights updated & saved")
}

func (m *MasterServer) aliveWorkerCountLocked() int {
	count := 0
	now := time.Now()
	for _, w := range m.workers {
		if now.Sub(w.LastHeartbeat) <= 15*time.Second {
			w.IsAlive = true
			count++
			continue
		}
		w.IsAlive = false
	}
	if count == 0 {
		return 1
	}
	return count
}

func (m *MasterServer) expectedWorkersForRoundLocked() int {
	if m.roundPartCount > 0 {
		return m.roundPartCount
	}
	return m.aliveWorkerCountLocked()
}

func (m *MasterServer) pickNextShardLocked() string {
	if len(m.config.ShardPaths) == 0 {
		return ""
	}
	shard := m.config.ShardPaths[m.nextShardIndex%len(m.config.ShardPaths)]
	m.nextShardIndex++
	return shard
}

func (m *MasterServer) computeCapacityScoreLocked(workerID string) float64 {
	worker, ok := m.workers[workerID]
	if !ok {
		return 0
	}
	var maxThroughput float64 = 1
	var maxMemory float64 = 1
	var maxGPUHeadroom float64 = 1

	for _, w := range m.workers {
		if !w.IsAlive {
			continue
		}
		maxThroughput = math.Max(maxThroughput, float64(w.Metrics.Throughput))
		maxMemory = math.Max(maxMemory, float64(w.Metrics.RamFreeGb+w.Metrics.GpuMemFreeGb))
		maxGPUHeadroom = math.Max(maxGPUHeadroom, float64(100.0-w.Metrics.GpuUtilPercent))
	}

	throughputNorm := float64(worker.Metrics.Throughput) / maxThroughput
	memoryNorm := float64(worker.Metrics.RamFreeGb+worker.Metrics.GpuMemFreeGb) / maxMemory
	gpuHeadroomNorm := float64(100.0-worker.Metrics.GpuUtilPercent) / maxGPUHeadroom

	throughputNorm = clamp01(throughputNorm)
	memoryNorm = clamp01(memoryNorm)
	gpuHeadroomNorm = clamp01(gpuHeadroomNorm)

	return 0.5*throughputNorm + 0.3*memoryNorm + 0.2*gpuHeadroomNorm
}

func (m *MasterServer) pickTaskSize(score float64) (string, int) {
	baseBatch := m.config.BatchSize
	if baseBatch <= 0 {
		baseBatch = 1
	}
	switch {
	case score >= 0.66:
		return "large", baseBatch * 4
	case score >= 0.33:
		return "medium", baseBatch * 2
	default:
		return "small", baseBatch
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func (m *MasterServer) workerPartitionLocked(workerID string) (int, int, bool) {
	if len(m.roundPartitions) == 0 {
		var alive []string
		for id, w := range m.workers {
			if w.IsAlive {
				alive = append(alive, id)
			}
		}
		if len(alive) == 0 {
			return 0, 1, false
		}
		sort.Strings(alive)
		m.roundPartCount = len(alive)
		for idx, id := range alive {
			m.roundPartitions[id] = idx
		}
	}
	idx, ok := m.roundPartitions[workerID]
	return idx, m.roundPartCount, ok
}

// ───────────── Get Weights ─────────────
func (m *MasterServer) GetWeights(ctx context.Context, req *pb.WeightsRequest) (*pb.WeightsResponse, error) {
	return &pb.WeightsResponse{Weights: m.weights}, nil
}

// ───────────── Save Weights ─────────────
func (m *MasterServer) SaveWeightsToFile(path string) {
	data, _ := json.Marshal(m.weights)
	os.WriteFile(path, data, 0644)
}

// ───────────── MAIN ─────────────
func main() {

	config := TrainingConfig{
		NumEpochs:    10,
		BatchSize:    4,
		LearningRate: 0.001,
		ShardPaths: []string{
			"dataset/train",
		},
	}

	weights := loadWeights("weights.json")

	master := NewMasterServer(config, weights)

	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		panic(err)
	}

	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(1024*1024*200),
		grpc.MaxSendMsgSize(1024*1024*200),
	)

	pb.RegisterTrainerServiceServer(server, master)

	fmt.Println("MASTER RUNNING ON :50051")
	server.Serve(lis)
}