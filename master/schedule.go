// schedule.go
//
// PURPOSE: Handles task scheduling — deciding which worker trains on which data.
//
// HOW SCHEDULING WORKS:
// ─────────────────────
// 1. When a worker registers, the master assigns it a shard (round-robin)
// 2. When a worker calls GetTask(), it gets its assigned shard + training config
// 3. If a worker dies, its shard is reassigned to an alive worker
//
// The actual shard assignment now happens in RegisterWorker (gradient_server.go)
// and the reassignment happens in the health checker.
// This file holds additional scheduling utilities.

package main

import "fmt"

// ShardAssignment represents which worker is training on which data shard.
type ShardAssignment struct {
	WorkerID  string
	ShardPath string
}

// GetShardAssignments returns a snapshot of all current shard assignments.
// Used for status reporting and debugging.
func (m *MasterServer) GetShardAssignments() []ShardAssignment {
	m.mu.Lock()
	defer m.mu.Unlock()

	var assignments []ShardAssignment
	for workerID, shard := range m.shardAssignments {
		assignments = append(assignments, ShardAssignment{
			WorkerID:  workerID,
			ShardPath: shard,
		})
	}
	return assignments
}

// reassignDeadWorkerShardsLocked gives each dead worker's shard to the first alive worker.
// Caller must hold m.mu.
func (m *MasterServer) reassignDeadWorkerShardsLocked() {
	for id, w := range m.workers {
		if w.IsAlive {
			continue
		}
		deadShard, ok := m.shardAssignments[id]
		if !ok || deadShard == "" {
			continue
		}
		for otherID, otherW := range m.workers {
			if otherW.IsAlive && otherID != id {
				m.shardAssignments[otherID] = deadShard
				fmt.Printf("[MASTER] Reassigned shard %s → %s (from dead worker %s)\n", deadShard, otherID, id)
				break
			}
		}
	}
}

// ReassignDeadWorkerShards takes shards from dead workers and gives them
// to alive workers. Called during failure recovery.
func (m *MasterServer) ReassignDeadWorkerShards() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reassignDeadWorkerShardsLocked()
}
