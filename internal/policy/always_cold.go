package policy

import (
	"log"

	"github.com/marcorentap/slrun/internal/types"
)

type AlwaysCold struct {
	Funcs             []*types.Function
	StartFuncInstance func(*types.Function) (*types.ContainerInstance, error)
	StopFuncInstance  func(*types.ContainerInstance) error
}

func (p *AlwaysCold) OnRuntimeStart() error {
	// Do nothing, functions started on demand
	return nil
}

func (p *AlwaysCold) PreFunctionCall(f *types.Function) (*types.ContainerInstance, error) {
	instance, err := p.StartFuncInstance(f)
	if err != nil {
		return nil, err
	}
	log.Printf("AlwaysCold: Started function %v (container %v)\n", f.Name, instance.ContainerId[:12])
	return instance, nil
}

func (p *AlwaysCold) PostFunctionCall(instance *types.ContainerInstance) error {
	err := p.StopFuncInstance(instance)
	if err != nil {
		return err
	}
	log.Printf("AlwaysCold: Stopped function %v (container %v)\n", instance.Function.Name, instance.ContainerId[:12])
	return nil
}

func (p *AlwaysCold) OnTick() error {
	return nil
}
