// heartbeat.go
//
// PURPOSE: Implements the heartbeat system — how workers tell the master
//          "I'm still alive".
//
// WHY HEARTBEATS?
// ───────────────
// In a distributed system, any machine can crash at any time.
// The master needs to know which workers are alive so it can:
//   - Only send tasks to live workers
//   - Detect failures and reassign work
//
// HOW IT WORKS:
// ─────────────
//   Worker sends heartbeat ──every 5 seconds──► Master
//   Master checks: "When was the last heartbeat from this worker?"
//   If > 15 seconds ago → worker is DEAD
//
// WHY 5 SECONDS?
//   Too fast (1s) = lots of network traffic for nothing
//   Too slow (60s) = takes too long to detect failures
//   5 seconds is a good balance
//
// WHY 15 SECOND TIMEOUT?
//   15 seconds = 3 missed heartbeats
//   This accounts for temporary network delays without false alarms

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "distributed_llm/proto"
)

// StartHeartbeat runs an infinite loop that sends heartbeats to the master.
// This function BLOCKS (runs forever), so it should be the last thing
// called in main().
//
// If the master is unreachable, the worker logs the error and keeps trying.
// This is important because the master might restart.
func StartHeartbeat(client pb.TrainerServiceClient, workerID string, metricsFn func() *pb.WorkerMetrics) {
	for {
		metrics := metricsFn()
		if metrics == nil {
			metrics = &pb.WorkerMetrics{}
		}
		// Send heartbeat to master
		_, err := client.SendHeartbeat(context.Background(), &pb.HeartbeatRequest{
			WorkerId: workerID,
			Metrics:  metrics,
		})

		if err != nil {
			// Don't crash — the master might come back
			fmt.Printf("[WORKER %s] Heartbeat failed: %v (will retry)\n", workerID, err)
		} else {
			fmt.Printf("[WORKER %s] Heartbeat sent at %s\n",
				workerID, time.Now().Format("15:04:05"))
		}

		// Wait 5 seconds before next heartbeat
		time.Sleep(5 * time.Second)
	}
}

// SendOneHeartbeat sends a single heartbeat (used for testing).
func SendOneHeartbeat(client pb.TrainerServiceClient, workerID string) error {
	_, err := client.SendHeartbeat(context.Background(), &pb.HeartbeatRequest{
		WorkerId: workerID,
	})
	if err != nil {
		log.Printf("[WORKER %s] Heartbeat error: %v", workerID, err)
	}
	return err
}
