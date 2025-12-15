package policy

import (
	"log"
	"time"

	"github.com/marcorentap/slrun/internal/types"
)

type ColdOnIdle struct {
	Funcs          []*types.Function
	StartFunc      func(*types.Function) error
	StopFunc       func(*types.Function) error
	idleThreshold  time.Duration
	lastExecTime   map[*types.Function]time.Time // Remember function last execution times
	activeRequests map[*types.Function]int       // Track number of active requests per function
}

func (p *ColdOnIdle) OnRuntimeStart() error {
	p.idleThreshold = 5 * time.Second
	p.lastExecTime = make(map[*types.Function]time.Time)
	p.activeRequests = make(map[*types.Function]int)
	return nil
}

func (p *ColdOnIdle) PreFunctionCall(f *types.Function) (*types.ContainerInstance, error) {
	if !f.IsRunning {
		err := p.StartFunc(f)
		if err != nil {
			return nil, err
		}
		log.Printf("ColdOnIdle: Started function %v\n", f.Name)
	}

	// Increment active request counter
	p.activeRequests[f]++
	
	// Log last execution time
	p.lastExecTime[f] = time.Now()
	
	// Return container instance using shared container
	instance := &types.ContainerInstance{
		ContainerId: f.ContainerId,
		Port:        f.Port,
		Function:    f,
	}
	return instance, nil
}

func (p *ColdOnIdle) PostFunctionCall(instance *types.ContainerInstance) error {
	f := instance.Function
	
	// Decrement active request counter
	p.activeRequests[f]--
	
	// Update last execution time when request completes
	p.lastExecTime[f] = time.Now()
	return nil
}

func (p *ColdOnIdle) OnTick() error {
	// Stop idled functions
	for _, f := range p.Funcs {
		if !f.IsRunning {
			continue
		}
		
		// Don't stop if there are active requests
		if p.activeRequests[f] > 0 {
			continue
		}
		
		lastExec, exists := p.lastExecTime[f]
		if !exists {
			continue
		}

		idleTime := time.Since(lastExec)
		if idleTime > p.idleThreshold {
			log.Printf("ColdOnIdle: Function %v idled for %v ms, stopping...", f.Name, idleTime.Milliseconds())
			err := p.StopFunc(f)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
