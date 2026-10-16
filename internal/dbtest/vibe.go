package dbtest

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// WriteVibeUnifiedGeneration publishes a minimal session with aggregate usage.
func WriteVibeUnifiedGeneration(t *testing.T, dir, generation, parent, model string, mtime time.Time) string {
	t.Helper()
	genDir := filepath.Join(dir, "generations", generation)
	anchor := filepath.Join(dir, "CURRENT")
	for path, content := range map[string]string{
		anchor:                                 `{"generation":"` + generation + `"}`,
		filepath.Join(genDir, "manifest.json"): `{}`,
		filepath.Join(genDir, "runtime-state.json"):    `{"runtime_state_version":3,"identity":{"kind":"subagent","parent_session_id":"` + parent + `"},"session_metadata":{"active_model":"` + model + `"}}`,
		filepath.Join(genDir, "projection-state.json"): `{"snapshot":{"session":{"tokenUsage":{"inputTokens":10,"outputTokens":5}},"history":{"entries":[{"type":"message","role":"assistant","content":[{"text":"child answer"}]}]}}}`,
	} {
		WriteTestFile(t, path, []byte(content))
		if !mtime.IsZero() {
			require.NoError(t, os.Chtimes(path, mtime, mtime))
		}
	}
	return anchor
}
