package slrun

type Function struct {
	Name     string `json:"name"`
	BuildDir string `json:"build_dir"`

	ImageName string
}

type Config struct {
	ConfigFile string
	Functions  []Function `json:"functions"`
}
