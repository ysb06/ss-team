package slrun

import (
	"context"
	"log"
	"os"
	"strings"

	"archive/tar"
	"bytes"
	"io"
	"path/filepath"

	"github.com/docker/docker/api/types/build"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
)

var config *Config
var dockerCli *client.Client
var dockerCtx context.Context

// createTarContext creates a tar archive of the directory at dirPath.
func createTarContext(dirPath string) (io.Reader, error) {
	buf := new(bytes.Buffer)
	tw := tar.NewWriter(buf)

	err := filepath.Walk(dirPath, func(file string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		header, err := tar.FileInfoHeader(fi, fi.Name())
		if err != nil {
			return err
		}

		// Use relative path so the archive structure matches the relative paths in the context directory
		relPath, err := filepath.Rel(dirPath, file)
		if err != nil {
			return err
		}
		header.Name = relPath

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if fi.Mode().IsRegular() {
			f, err := os.Open(file)
			if err != nil {
				return err
			}
			defer f.Close()

			if _, err := io.Copy(tw, f); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	if err := tw.Close(); err != nil {
		return nil, err
	}

	return buf, nil
}

func BuildFunctionImage(function Function) error {
	buildCtx, err := createTarContext(function.BuildDir)
	if err != nil {
		return err
	}

	// Remove then rebuild image
	imageName := "slrun-" + function.Name
	_, err = dockerCli.ImageRemove(dockerCtx, imageName, image.RemoveOptions{
		Force:         true,
		PruneChildren: true,
	})

	if err != nil {
		// If image doesn't exist, it's ok
		if !strings.Contains(err.Error(), "No such image: slrun-") {
			return err
		}
	}

	buildResp, err := dockerCli.ImageBuild(dockerCtx, buildCtx, build.ImageBuildOptions{
		Tags: []string{imageName},
	})
	if err != nil {
		return err
	}
	defer buildResp.Body.Close()

	// We have to read from the response, else it won't build
	io.Copy(io.Discard, buildResp.Body)
	return nil
}

func Start(cfgFile string) error {
	// Init
	config, err := ReadConfigFile(cfgFile)
	if err != nil {
		return err
	}
	dockerCli, err = client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return err
	}
	dockerCtx = context.Background()

	// Build function images
	for _, function := range config.Functions {
		log.Printf("Building function image: %v => %v\n", function.Name, function.BuildDir)
		err := BuildFunctionImage(function)
		if err != nil {
			return err
		}
		log.Printf("Built function image: %v\n", function.ImageName)
	}

	return nil
}
