package types

import "time"

type Function struct {
	Name     string `json:"name"`
	BuildDir string `json:"build_dir"`

	ImageName string
	
	// For ColdOnIdle and AlwaysHot policies that reuse containers
	ContainerId string
	IsRunning   bool
	Port        int // 127.0.0.1:X->80/tcp
}

// ContainerInstance represents a single container instance for a request
type ContainerInstance struct {
	ContainerId string
	Port        int
	Function    *Function // Reference to function template
	LastUsedAt  time.Time // Last time this instance was used (for idle timeout)
	InUse       bool      // Whether this instance is currently processing a request
}

type Config struct {
	ConfigFile string
	Functions  []*Function `json:"functions"`
	Policy     PolicyID
}

type PolicyID string

const (
	AlwaysHotPolicy  = "always_hot"
	AlwaysColdPolicy = "always_cold"
	ColdOnIdlePolicy = "cold_on_idle"
	HotStartPolicy   = "hot_start"
)

type Policy interface {
	OnRuntimeStart() error
	PreFunctionCall(f *Function) (*ContainerInstance, error)
	PostFunctionCall(instance *ContainerInstance) error
	OnTick() error
}
