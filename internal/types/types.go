package types

type Function struct {
	Name     string `json:"name"`
	BuildDir string `json:"build_dir"`

	ImageName   string
	ContainerId string
	IsRunning   bool
	Port        int // 127.0.0.1:X->80/tcp
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
)

type Policy interface {
	OnRuntimeStart() error
	PreFunctionCall(f *Function) error
	PostFunctionCall(f *Function) error
	OnTick() error
}
