// checkpoint.go
//
// PURPOSE: Saves and loads model weights to/from disk.
//
// WHY CHECKPOINTING?
// ──────────────────
// Training can take hours. If the system crashes, we lose all progress.
// Checkpointing saves the model weights periodically so we can:
//   - Resume training after a crash
//   - Keep snapshots of the model at different stages
//   - Compare model quality over time
//
// HOW IT WORKS:
// ─────────────
// Every N training steps, the master:
//   1. Takes the current model weights
//   2. Saves them to a JSON file: storage/checkpoint_step_500.json
//   3. Logs the save
//
// To resume, the master loads the latest checkpoint on startup.

package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Checkpoint holds the model state at a given training step.
type Checkpoint struct {
	Step    int       `json:"step"`
	Epoch   int       `json:"epoch"`
	Weights []float32 `json:"weights"`
	Loss    float64   `json:"loss"`
}

// SaveCheckpoint writes model weights to disk.
//
// File format: storage/checkpoint_step_500.json
// Content: JSON with step number, weights array, and loss value.
func SaveCheckpoint(dir string, step int, epoch int, weights []float32, loss float64) (string, error) {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(dir, 0750); err != nil {
		return "", fmt.Errorf("failed to create checkpoint dir: %w", err)
	}

	filename := fmt.Sprintf("checkpoint_step_%d.json", step)
	path := filepath.Join(dir, filename)

	ckpt := Checkpoint{
		Step:    step,
		Epoch:   epoch,
		Weights: weights,
		Loss:    loss,
	}

	data, err := json.Marshal(ckpt)
	if err != nil {
		return "", fmt.Errorf("failed to marshal checkpoint: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return "", fmt.Errorf("failed to write checkpoint: %w", err)
	}

	fmt.Printf("[CHECKPOINT] Saved step %d to %s (loss: %.6f)\n", step, path, loss)
	return path, nil
}

// LoadLatestCheckpoint finds and loads the most recent checkpoint.
//
// It scans the directory for checkpoint_step_*.json files,
// finds the one with the highest step number, and loads it.
func LoadLatestCheckpoint(dir string) (*Checkpoint, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("no checkpoint directory: %w", err)
	}

	// Find checkpoint files and pick the highest step (numeric, not lexicographic).
	var latestFile string
	var latestStep int
	found := false
	prefix := "checkpoint_step_"
	suffix := ".json"
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffix) {
			continue
		}
		mid := strings.TrimSuffix(strings.TrimPrefix(name, prefix), suffix)
		step, err := strconv.Atoi(mid)
		if err != nil {
			continue
		}
		if !found || step > latestStep {
			found = true
			latestStep = step
			latestFile = name
		}
	}

	if !found {
		return nil, fmt.Errorf("no checkpoints found in %s", dir)
	}
	path := filepath.Join(dir, latestFile)

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint: %w", err)
	}

	var ckpt Checkpoint
	if err := json.Unmarshal(data, &ckpt); err != nil {
		return nil, fmt.Errorf("failed to parse checkpoint: %w", err)
	}

	fmt.Printf("[CHECKPOINT] Loaded step %d from %s (loss: %.6f)\n", ckpt.Step, path, ckpt.Loss)
	return &ckpt, nil
}
