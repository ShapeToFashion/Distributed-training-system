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
	"net"
	"os"
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

	shardIndex := len(m.workers) % len(m.config.ShardPaths)
	shard := m.config.ShardPaths[shardIndex]
	m.shardAssignments[req.WorkerId] = shard

	fmt.Printf("[MASTER] Assigned shard %s → %s\n", shard, req.WorkerId)
	return &pb.RegisterResponse{Status: "registered"}, nil
}

func (m *MasterServer) SendHeartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if w, ok := m.workers[req.WorkerId]; ok {
		w.LastHeartbeat = time.Now()
		w.IsAlive = true
	}
	return &pb.HeartbeatResponse{Status: "alive"}, nil
}

func (m *MasterServer) GetTask(ctx context.Context, req *pb.TaskRequest) (*pb.TaskResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.trainingDone {
		return &pb.TaskResponse{HasTask: false}, nil
	}
	shard := m.shardAssignments[req.WorkerId]
	if shard == "" {
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

// ───────────── Send Gradients ─────────────
func (m *MasterServer) SendGradients(ctx context.Context, req *pb.GradientRequest) (*pb.GradientResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.gradientBuffer[req.WorkerId] = req.Gradients
	m.lossBuffer[req.WorkerId] = req.Loss
	m.gradientCount++
	fmt.Printf("[MASTER] Received gradients from %s: grad_len=%d loss=%.6f\n", req.WorkerId, len(req.Gradients), req.Loss)

	if m.gradientCount >= len(m.workers) {
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

	fmt.Println("[MASTER] Weights updated & saved")
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