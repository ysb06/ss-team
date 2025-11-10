package policy

import (
	"log"

	"github.com/marcorentap/slrun/internal/types"
)

type AlwaysCold struct {
	Funcs     []*types.Function
	StartFunc func(*types.Function) error
	StopFunc  func(*types.Function) error
}

func (p *AlwaysCold) OnRuntimeStart() error {
	// Do nothing, functions started on demand
	return nil
}

func (p *AlwaysCold) PreFunctionCall(f *types.Function) error {
	err := p.StartFunc(f)
	if err != nil {
		return err
	}
	log.Printf("AlwaysCold: Started function %v\n", f.Name)
	return nil
}

func (p *AlwaysCold) PostFunctionCall(f *types.Function) error {
	err := p.StopFunc(f)
	if err != nil {
		return err
	}
	log.Printf("AlwaysCold: Stopped function %v\n", f.Name)
	return nil
}
