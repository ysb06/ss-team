package policy

import (
	"log"
	"time"

	"github.com/marcorentap/slrun/internal/types"
)

type ColdOnIdle struct {
	Funcs         []*types.Function
	StartFunc     func(*types.Function) error
	StopFunc      func(*types.Function) error
	idleThreshold time.Duration
	lastExecTime  map[*types.Function]time.Time // Remember function last execution times
}

func (p *ColdOnIdle) OnRuntimeStart() error {
	p.idleThreshold = 5 * time.Second
	p.lastExecTime = make(map[*types.Function]time.Time)
	return nil
}

func (p *ColdOnIdle) PreFunctionCall(f *types.Function) error {
	if !f.IsRunning {
		err := p.StartFunc(f)
		if err != nil {
			return err
		}
		log.Printf("ColdOnIdle: Started function %v\n", f.Name)
	}

	// Log last execution time
	p.lastExecTime[f] = time.Now()
	return nil
}

func (p *ColdOnIdle) PostFunctionCall(f *types.Function) error {
	return nil
}

func (p *ColdOnIdle) OnTick() error {
	// Stop idled functions
	for _, f := range p.Funcs {
		if !f.IsRunning {
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
