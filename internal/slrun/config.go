package slrun

import (
	"encoding/json"
	"fmt"
	"os"
)

func validateConfig(config *Config) error {
	// Function names should be unique
	for _, f := range config.Functions {
		for _, f2 := range config.Functions {
			if f != f2 && f.Name == f2.Name {
				return fmt.Errorf("config has duplicate function name: %s", f.Name)
			}
		}
	}

	return nil
}

func ReadConfigFile(path string) (*Config, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	config := Config{ConfigFile: path}

	err = json.Unmarshal(bytes, &config)
	if err != nil {
		return nil, err
	}

	err = validateConfig(&config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}
