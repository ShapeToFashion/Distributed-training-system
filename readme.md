# Distributed LLM Training System

A distributed system for training Large Language Models across multiple machines
using data parallelism, gRPC communication, and synchronous gradient aggregation.

## Architecture

```
                 ┌───────────────────┐
                 │      MASTER       │
                 │    (Go Server)    │
                 │                   │
                 │ Gradient Server   │
                 │ Worker Manager    │
                 │ Scheduler         │
                 └─────────┬─────────┘
                           │
                     gRPC (port 50051)
                           │
        ┌──────────┬───────┴───┬──────────┐
     Worker1    Worker2    Worker3    Worker4
      (Go)       (Go)       (Go)       (Go)
```

## Project Structure

```
proto/              gRPC protocol definition
  trainer.proto     Service + message definitions
  trainer.pb.go     Auto-generated message code
  trainer_grpc.pb.go Auto-generated service code

master/             Master node (runs on one machine)
  gradient_server.go Main entry point + gRPC handlers
  worker_manager.go  Worker health monitoring helpers
  schedule.go        Task scheduling logic

worker/             Worker node (runs on each machine)
  worker.go          Main entry point + registration
  heartbeat.go       Heartbeat/keep-alive system

training/           Python training scripts (Step 7)
dataset/            Training data shards
storage/            Model checkpoints (Step 9)
```

## Quick Start

**Terminal 1 — Start Master:**

```
go run master/gradient_server.go master/worker_manager.go master/schedule.go
```

**Terminal 2 — Start Worker 1:**

```
go run worker/worker.go worker/heartbeat.go -id=worker1
```

**Terminal 3 — Start Worker 2:**

```
go run worker/worker.go worker/heartbeat.go -id=worker2
```

## Development Progress

- [x] Step 1: gRPC protocol definition
- [x] Step 2: Master server
- [x] Step 3: Worker node
- [x] Step 4: Worker registration
- [x] Step 5: Heartbeat system
- [ ] Step 6: Dataset distribution
- [ ] Step 7: Training execution
- [ ] Step 8: Gradient aggregation
- [ ] Step 9: Checkpointing
