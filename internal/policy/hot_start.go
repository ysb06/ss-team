package policy

import (
	"log"
	"sync"
	"time"

	"github.com/marcorentap/slrun/internal/types"
)

// HotStart policy maintains a pool of warm containers that can be reused across requests
// This provides fast startup times while being more resource-efficient than AlwaysHot
type HotStart struct {
	Funcs             []*types.Function
	StartFuncInstance func(*types.Function) (*types.ContainerInstance, error)
	StopFuncInstance  func(*types.ContainerInstance) error

	// Container pool: available instances per function
	instancePools map[string][]*types.ContainerInstance // key: function name
	poolMutex     sync.Mutex

	// Pool configuration
	MaxPoolSize int // Maximum pool size per function (default: 5)
	MinPoolSize int // Minimum instances to keep warm per function (default: 1)
}

func (p *HotStart) OnRuntimeStart() error {
	// Initialize instance pools
	p.instancePools = make(map[string][]*types.ContainerInstance)

	// Set default pool sizes if not configured
	if p.MaxPoolSize == 0 {
		p.MaxPoolSize = 150
	}
	if p.MinPoolSize == 0 {
		p.MinPoolSize = 1
	}

	// Pre-warm containers (optional: warm up MinPoolSize containers per function)
	for _, f := range p.Funcs {
		p.instancePools[f.Name] = make([]*types.ContainerInstance, 0, p.MaxPoolSize)

		// Create initial warm containers
		for i := 0; i < p.MinPoolSize; i++ {
			instance, err := p.StartFuncInstance(f)
			if err != nil {
				log.Printf("HotStart: Failed to pre-warm function %v: %v", f.Name, err)
				continue
			}
			instance.LastUsedAt = time.Now()
			instance.InUse = false
			p.instancePools[f.Name] = append(p.instancePools[f.Name], instance)
			log.Printf("HotStart: Pre-warmed function %v (container %v)", f.Name, instance.ContainerId[:12])
		}
	}

	log.Printf("HotStart: Runtime started with pool sizes [min=%d, max=%d]", p.MinPoolSize, p.MaxPoolSize)
	return nil
}

func (p *HotStart) PreFunctionCall(f *types.Function) (*types.ContainerInstance, error) {
	p.poolMutex.Lock()

	// Check if there's an available instance in the pool
	pool, exists := p.instancePools[f.Name]
	if exists && len(pool) > 0 {
		// Reuse an instance from the pool
		instance := pool[len(pool)-1]
		p.instancePools[f.Name] = pool[:len(pool)-1] // Pop from pool

		instance.InUse = true
		instance.LastUsedAt = time.Now()

		p.poolMutex.Unlock()

		log.Printf("HotStart: Reused container for function %v (container %v, pool size: %d)",
			f.Name, instance.ContainerId[:12], len(pool)-1)
		return instance, nil
	}

	p.poolMutex.Unlock()

	// No available instance, create a new one
	instance, err := p.StartFuncInstance(f)
	if err != nil {
		return nil, err
	}

	instance.InUse = true
	instance.LastUsedAt = time.Now()

	log.Printf("HotStart: Created new container for function %v (container %v, pool was empty)",
		f.Name, instance.ContainerId[:12])
	return instance, nil
}

func (p *HotStart) PostFunctionCall(instance *types.ContainerInstance) error {
	p.poolMutex.Lock()
	defer p.poolMutex.Unlock()

	instance.InUse = false
	instance.LastUsedAt = time.Now()

	// Check current pool size
	pool := p.instancePools[instance.Function.Name]
	currentSize := len(pool)

	if currentSize < p.MaxPoolSize {
		// Return instance to pool for reuse
		p.instancePools[instance.Function.Name] = append(pool, instance)
		log.Printf("HotStart: Returned container to pool for function %v (container %v, pool size: %d -> %d)",
			instance.Function.Name, instance.ContainerId[:12], currentSize, currentSize+1)
		return nil
	}

	// Pool is full, stop and remove the excess container
	// Note: We release the lock before stopping the container to avoid blocking
	p.poolMutex.Unlock()
	err := p.StopFuncInstance(instance)
	p.poolMutex.Lock()

	if err != nil {
		log.Printf("HotStart: Failed to stop excess container for function %v: %v",
			instance.Function.Name, err)
		return err
	}

	log.Printf("HotStart: Pool full, stopped excess container for function %v (container %v)",
		instance.Function.Name, instance.ContainerId[:12])
	return nil
}

func (p *HotStart) OnTick() error {
	// Optional: Implement idle timeout and pool maintenance
	// For now, we'll keep it simple and just maintain the pools as-is
	// Future enhancement: Remove containers that haven't been used for X minutes

	p.poolMutex.Lock()
	defer p.poolMutex.Unlock()

	// Ensure minimum pool size is maintained
	for _, f := range p.Funcs {
		pool := p.instancePools[f.Name]
		currentSize := len(pool)

		if currentSize < p.MinPoolSize {
			// Need to create more instances
			needed := p.MinPoolSize - currentSize
			for i := 0; i < needed; i++ {
				p.poolMutex.Unlock()
				instance, err := p.StartFuncInstance(f)
				p.poolMutex.Lock()

				if err != nil {
					log.Printf("HotStart: Failed to maintain min pool size for function %v: %v", f.Name, err)
					continue
				}

				instance.LastUsedAt = time.Now()
				instance.InUse = false
				p.instancePools[f.Name] = append(p.instancePools[f.Name], instance)
				log.Printf("HotStart: Added instance to maintain min pool size for function %v (container %v)",
					f.Name, instance.ContainerId[:12])
			}
		}
	}

	return nil
}
