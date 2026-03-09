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

## Prerequisites

### 1. Install Go

Download and install Go (1.25+): https://go.dev/dl/

Verify:

```bash
go version
```

### 2. Install Protocol Buffers Compiler (protoc)

**Windows:**

```bash
# Using Chocolatey
choco install protobuf

# OR download manually from:
# https://github.com/protocolbuffers/protobuf/releases
# Add protoc.exe to your PATH
```

**macOS:**

```bash
brew install protobuf
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt install -y protobuf-compiler
```

Verify:

```bash
protoc --version
```

### 3. Install Go gRPC Plugins

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

Make sure `$GOPATH/bin` (or `%GOPATH%\bin` on Windows) is in your PATH:

```bash
# Linux/macOS — add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:$(go env GOPATH)/bin"

# Windows (PowerShell)
$env:PATH += ";$(go env GOPATH)\bin"
```

### 4. Install Python (for training scripts)

Download Python 3.10+: https://www.python.org/downloads/

---

## Setup After Cloning

```bash
git clone https://github.com/ShapeToFashion/Distributed-training-system.git
cd Distributed-training-system
```

### Install Go Dependencies

```bash
go mod tidy
```

### (Re)generate gRPC Code from Proto

If you modify `proto/trainer.proto`, regenerate the Go bindings:

```bash
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto/trainer.proto
```

This generates:

- `proto/trainer.pb.go` — message serialization code
- `proto/trainer_grpc.pb.go` — gRPC client/server stubs

---

## Quick Start (Same Machine)

**Terminal 1 — Start Master:**

```bash
go run master/gradient_server.go master/worker_manager.go master/schedule.go
```

Master listens on **port 50051**.

**Terminal 2 — Start Worker 1:**

```bash
go run worker/worker.go worker/heartbeat.go -id=worker1
```

**Terminal 3 — Start Worker 2:**

```bash
go run worker/worker.go worker/heartbeat.go -id=worker2
```

---

## Connecting Across Two Machines (LAN / Remote)

### Machine A (Master)

1. Master IP (Kartik's machine):

   ```
   IPv4 Address: 10.56.89.116
   Port: 50051
   ```

2. Start the master:

   ```bash
   go run master/gradient_server.go master/worker_manager.go master/schedule.go
   ```

   Master will listen on `0.0.0.0:50051` (all interfaces).

3. **Open firewall for port 50051:**

   ```bash
   # Windows (run as Administrator)
   netsh advfirewall firewall add rule name="gRPC Master" dir=in action=allow protocol=TCP localport=50051

   # Linux
   sudo ufw allow 50051/tcp
   ```

### Machine B (Worker)

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/ShapeToFashion/Distributed-training-system.git
   cd Distributed-training-system
   go mod tidy
   ```

2. Connect worker to master:
   ```bash
   go run worker/worker.go worker/heartbeat.go -id=worker1 -master=10.56.89.116:50051
   ```

### Test Connection Only (without training)

```bash
go run worker/worker.go worker/heartbeat.go -id=test-worker -master=10.56.89.116:50051 -test
```

This registers with the master and exits — useful to verify the gRPC link works.

---

## gRPC Service Reference

Defined in `proto/trainer.proto`:

| RPC Method       | Direction       | Purpose                                  |
| ---------------- | --------------- | ---------------------------------------- |
| `RegisterWorker` | Worker → Master | Worker joins the cluster                 |
| `SendHeartbeat`  | Worker → Master | Keep-alive ping (every 5s)               |
| `GetTask`        | Worker → Master | Worker polls for a training task (shard) |
| `GetWeights`     | Worker → Master | Worker fetches latest model weights      |
| `SendGradients`  | Worker → Master | Worker sends computed gradients          |
| `SaveCheckpoint` | Internal        | Master saves model checkpoint to disk    |

**Default port:** `50051`

---

## Troubleshooting

| Problem                            | Fix                                                                  |
| ---------------------------------- | -------------------------------------------------------------------- |
| `connection refused`               | Check master is running and IP/port are correct                      |
| `context deadline exceeded`        | Firewall may be blocking port 50051                                  |
| `protoc-gen-go: program not found` | Run `go install` commands above and add GOPATH/bin to PATH           |
| `module not found` errors          | Run `go mod tidy` in the project root                                |
| Workers disconnect                 | Check heartbeat logs; master marks workers dead after 15s of silence |

---

## Development Progress

- [x] Step 1: gRPC protocol definition
- [x] Step 2: Master server
- [x] Step 3: Worker node
- [x] Step 4: Worker registration
- [x] Step 5: Heartbeat system
- [ ] Step 6: Dataset distribution
- [ ] Step 7: Training execution
- [ ] Step 8: Gradient aggregation
- [ ] Step 9: Checkpointingeck master is running and IP/port are correct |
| `context deadline exceeded` | Firewall may be blocking port 50051 |
| `protoc-gen-go: program not found` | Run `go install` commands above and add GOPATH/bin to PATH |
| `module not found` errors | Run `go mod tidy` in the project root |
| Workers disconnect | Check heartbeat logs; master marks workers dead after 15s of silence |

---

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
