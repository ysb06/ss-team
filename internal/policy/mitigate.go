package policy

import (
	"log"
	"math/rand"
	"sync"
	"time"

	"github.com/marcorentap/slrun/internal/types"
)

// Mitigate policy is identical to HotStart but adds random delay before function calls
// The delay follows a normal distribution to mitigate timing attacks
type Mitigate struct {
	Funcs             []*types.Function
	StartFuncInstance func(*types.Function) (*types.ContainerInstance, error)
	StopFuncInstance  func(*types.ContainerInstance) error

	// Container pool: available instances per function
	instancePools map[string][]*types.ContainerInstance // key: function name
	poolMutex     sync.Mutex

	// Pool configuration
	MaxPoolSize int // Maximum pool size per function (default: 150)
	MinPoolSize int // Minimum instances to keep warm per function (default: 1)

	// Delay configuration (hardcoded)
	DelayMean   float64 // Mean delay in seconds (default: 3.65)
	DelayStdDev float64 // Standard deviation in seconds (default: 1.97)

	// Random number generator
	rng *rand.Rand
}

func (p *Mitigate) OnRuntimeStart() error {
	// Initialize instance pools
	p.instancePools = make(map[string][]*types.ContainerInstance)

	// Initialize random number generator with current time as seed
	p.rng = rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set default pool sizes if not configured
	if p.MaxPoolSize == 0 {
		p.MaxPoolSize = 150
	}
	if p.MinPoolSize == 0 {
		p.MinPoolSize = 1
	}

	// Set default delay parameters if not configured
	if p.DelayMean == 0 {
		p.DelayMean = 0.9125
	}
	if p.DelayStdDev == 0 {
		p.DelayStdDev = 0.4925
	}

	// Pre-warm containers (optional: warm up MinPoolSize containers per function)
	for _, f := range p.Funcs {
		p.instancePools[f.Name] = make([]*types.ContainerInstance, 0, p.MaxPoolSize)

		// Create initial warm containers
		for i := 0; i < p.MinPoolSize; i++ {
			instance, err := p.StartFuncInstance(f)
			if err != nil {
				log.Printf("Mitigate: Failed to pre-warm function %v: %v", f.Name, err)
				continue
			}
			instance.LastUsedAt = time.Now()
			instance.InUse = false
			p.instancePools[f.Name] = append(p.instancePools[f.Name], instance)
			log.Printf("Mitigate: Pre-warmed function %v (container %v)", f.Name, instance.ContainerId[:12])
		}
	}

	log.Printf("Mitigate: Runtime started with pool sizes [min=%d, max=%d], delay [mean=%.2fs, stddev=%.2fs]",
		p.MinPoolSize, p.MaxPoolSize, p.DelayMean, p.DelayStdDev)
	return nil
}

func (p *Mitigate) PreFunctionCall(f *types.Function) (*types.ContainerInstance, error) {
	// Apply random delay before processing the request
	// Generate a normally distributed random delay
	delay := p.rng.NormFloat64()*p.DelayStdDev + p.DelayMean
	
	// Ensure delay is non-negative
	if delay < 0 {
		delay = 0
	}

	delayDuration := time.Duration(delay * float64(time.Second))
	log.Printf("Mitigate: Applying random delay of %.3fs for function %v", delay, f.Name)
	time.Sleep(delayDuration)

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

		log.Printf("Mitigate: Reused container for function %v (container %v, pool size: %d)",
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

	log.Printf("Mitigate: Created new container for function %v (container %v, pool was empty)",
		f.Name, instance.ContainerId[:12])
	return instance, nil
}

func (p *Mitigate) PostFunctionCall(instance *types.ContainerInstance) error {
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
		log.Printf("Mitigate: Returned container to pool for function %v (container %v, pool size: %d -> %d)",
			instance.Function.Name, instance.ContainerId[:12], currentSize, currentSize+1)
		return nil
	}

	// Pool is full, stop and remove the excess container
	// Note: We release the lock before stopping the container to avoid blocking
	p.poolMutex.Unlock()
	err := p.StopFuncInstance(instance)
	p.poolMutex.Lock()

	if err != nil {
		log.Printf("Mitigate: Failed to stop excess container for function %v: %v",
			instance.Function.Name, err)
		return err
	}

	log.Printf("Mitigate: Pool full, stopped excess container for function %v (container %v)",
		instance.Function.Name, instance.ContainerId[:12])
	return nil
}

func (p *Mitigate) OnTick() error {
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
					log.Printf("Mitigate: Failed to maintain min pool size for function %v: %v", f.Name, err)
					continue
				}

				instance.LastUsedAt = time.Now()
				instance.InUse = false
				p.instancePools[f.Name] = append(p.instancePools[f.Name], instance)
				log.Printf("Mitigate: Added instance to maintain min pool size for function %v (container %v)",
					f.Name, instance.ContainerId[:12])
			}
		}
	}

	return nil
}
