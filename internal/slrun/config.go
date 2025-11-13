package slrun

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"slices"

	"github.com/marcorentap/slrun/internal/types"
)

func validateConfig(config *types.Config) error {
	// Function names should be unique
	for _, f := range config.Functions {
		for _, f2 := range config.Functions {
			if f != f2 && f.Name == f2.Name {
				return fmt.Errorf("config has duplicate function name: %s", f.Name)
			}
		}
	}

	validPolicies := []types.PolicyID{types.AlwaysHotPolicy, types.AlwaysColdPolicy, types.ColdOnIdlePolicy}
	if !slices.Contains(validPolicies, config.Policy) {
		return fmt.Errorf("invalid policy: %s", config.Policy)
	}

	return nil
}

func ReadConfigFile(path string) (*types.Config, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	config := types.Config{ConfigFile: path}

	err = json.Unmarshal(bytes, &config)
	if err != nil {
		return nil, err
	}

	err = validateConfig(&config)
	if err != nil {
		return nil, err
	}

	log.Printf("Policy: %v\n", config.Policy)

	return &config, nil
}
