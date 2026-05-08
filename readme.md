# Distributed AI Training System

This project implements a distributed system for training an AI model. It consists of a central master server that coordinates tasks and a set of worker nodes that perform the actual training. The communication between the master and workers is handled via gRPC.

## Prerequisites

- Go
- Python 3
- PyTorch
- `bore` (for running workers on different networks)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Go dependencies:**
    ```bash
    go mod tidy
    ```

3.  **Create a Python virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    # Activate the environment
    # On Windows (PowerShell):
    .venv\Scripts\Activate.ps1
    # On macOS/Linux:
    # source .venv/bin/activate

    pip install -r training/requirements.txt
    ```

## Running the System

### 1. Start the Master Server

Open a terminal in the project root and run:

```bash
go run master/gradient_server.go master/worker_manager.go master/schedule.go
```

The master server will start and listen for worker connections on `localhost:50051`.

### 2. Running a Local Worker

To run a worker on the same machine as the master, open a **new** terminal (and activate the Python virtual environment) and run:

```bash
go run worker/worker.go worker/heartbeat.go -id=local-worker-1
```

You can run multiple local workers by providing a different `-id` for each one.

### 3. Running a Worker on a Different Network

To connect workers from external networks, you need to expose your master server to the internet. We use a tool called `bore` for this.

**A. Expose the Master Server:**

1.  Download and extract `bore`.
2.  Open a **new** terminal and run the following command to create a public tunnel to your local master server on port `50051`:
    ```bash
    # Replace <path-to-bore> with the actual path to the executable
    <path-to-bore>\bore.exe local 50051 --to bore.pub
    ```
3.  `bore` will output a public address, for example: `listening at bore.pub:12345`. Note this address.

**B. Start the Remote Worker:**

1.  On the other computer, make sure you have the project code and all dependencies installed.
2.  Open a terminal and run the worker, pointing it to the public address from the previous step:
    ```bash
    # Replace bore.pub:12345 with the actual address from your bore terminal
    go run worker/worker.go worker/heartbeat.go -id=internet-worker -master=bore.pub:12345
    ```

**Important:** For the remote worker to function, you must keep the terminals for the **master server** and the **`bore` tunnel** running on your main computer.

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

### Python Environment (required for workers)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision pillow
```

### Prepare local training paths

The current master config uses `dataset/train`. If your dataset is class folders under `dataset/`, create a train view:

```bash
mkdir -p dataset/train
ln -sfn ../apple dataset/train/apple
ln -sfn ../hourglass dataset/train/hourglass
ln -sfn ../inverted_triangle dataset/train/inverted_triangle
ln -sfn ../pear dataset/train/pear
ln -sfn ../rectangle dataset/train/rectangle
```

### Generate compatible `weights.json`

```bash
PATH="$(pwd)/.venv/bin:$PATH" python -c "import json, torch.nn as nn; from torchvision import models; m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,5); flat=[]; [flat.extend(t.detach().cpu().view(-1).tolist()) for _,t in m.state_dict().items()]; json.dump(flat, open('weights.json','w')); print('weights_len', len(flat))"
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
PATH="$(pwd)/.venv/bin:$PATH" go run worker/worker.go worker/heartbeat.go -id=worker1
```

**Terminal 3 — Start Worker 2:**

```bash
PATH="$(pwd)/.venv/bin:$PATH" go run worker/worker.go worker/heartbeat.go -id=worker2
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
   PATH="$(pwd)/.venv/bin:$PATH" go run worker/worker.go worker/heartbeat.go -id=worker1 -master=10.56.89.116:50051
   ```

### Test Connection Only (without training)

```bash
PATH="$(pwd)/.venv/bin:$PATH" go run worker/worker.go worker/heartbeat.go -id=test-worker -master=10.56.89.116:50051 -test
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
- [ ] Step 9: Checkpointing
