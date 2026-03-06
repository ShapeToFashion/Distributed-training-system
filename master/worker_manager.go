// worker_manager.go
//
// PURPOSE: Utility functions for worker management and cluster monitoring.

package main

import "fmt"

// PrintClusterStatus prints a human-readable summary of the cluster.
func (m *MasterServer) PrintClusterStatus() {
	m.mu.Lock()
	defer m.mu.Unlock()

	fmt.Println("\n═══ Cluster Status ═══")
	fmt.Printf("Total workers: %d\n", len(m.workers))
	fmt.Printf("Training step: %d | Epoch: %d/%d\n", m.currentStep, m.currentEpoch, m.config.NumEpochs)

	for id, w := range m.workers {
		status := "ALIVE"
		if !w.IsAlive {
			status = "DEAD"
		}
		shard := w.AssignedShard
		if shard == "" {
			shard = "none"
		}
		fmt.Printf("  %s [%s] — %s — shard: %s — heartbeat: %s\n",
			id, w.Address, status, shard, w.LastHeartbeat.Format("15:04:05"))
	}
	fmt.Println("══════════════════════")
}

// GetAliveWorkerIDs returns the IDs of all workers currently alive.
func (m *MasterServer) GetAliveWorkerIDs() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	var ids []string
	for id, w := range m.workers {
		if w.IsAlive {
			ids = append(ids, id)
		}
	}
	return ids
}

// GetDeadWorkerIDs returns the IDs of all workers that have stopped responding.
func (m *MasterServer) GetDeadWorkerIDs() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	var ids []string
	for id, w := range m.workers {
		if !w.IsAlive {
			ids = append(ids, id)
		}
	}
	return ids
}
