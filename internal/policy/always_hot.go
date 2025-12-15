package policy

import (
	"log"

	"github.com/marcorentap/slrun/internal/types"
)

type AlwaysHot struct {
	Funcs     []*types.Function
	StartFunc func(*types.Function) error
	StopFunc  func(*types.Function) error
}

func (p *AlwaysHot) OnRuntimeStart() error {
	for _, f := range p.Funcs {
		err := p.StartFunc(f)
		if err != nil {
			return err
		}
		log.Printf("AlwaysHot: started function %v\n", f.Name)
	}
	return nil
}

func (p *AlwaysHot) PreFunctionCall(f *types.Function) (*types.ContainerInstance, error) {
	// Return container instance using shared container
	instance := &types.ContainerInstance{
		ContainerId: f.ContainerId,
		Port:        f.Port,
		Function:    f,
	}
	return instance, nil
}

func (p *AlwaysHot) PostFunctionCall(instance *types.ContainerInstance) error {
	// Do nothing, functions are always hot
	return nil
}

func (p *AlwaysHot) OnTick() error {
	return nil
}
