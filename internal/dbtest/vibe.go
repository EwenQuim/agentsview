package dbtest

import (
	"crypto/sha256"
	"fmt"
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
	manifest := fmt.Sprintf(`{"manifest_version":1,"session_id":%q,"generation":%q,"snapshot_sequence":1}`, filepath.Base(dir), generation)
	pointer := fmt.Sprintf(`{"store_format":"mistral.vibe.unified-session-store/v1","store_format_minor":8,"session_id":%q,"generation":%q,"snapshot_sequence":1,"manifest_sha256":"%x"}`, filepath.Base(dir), generation, sha256.Sum256([]byte(manifest)))
	for path, content := range map[string]string{
		filepath.Join(genDir, "manifest.json"):         manifest,
		filepath.Join(genDir, "runtime-state.json"):    `{"runtime_state_version":3,"identity":{"kind":"subagent","parent_session_id":"` + parent + `"},"session_metadata":{"active_model":"` + model + `"}}`,
		filepath.Join(genDir, "projection-state.json"): `{"snapshot":{"session":{"tokenUsage":{"inputTokens":10,"outputTokens":5}},"history":{"entries":[{"type":"message","role":"assistant","content":[{"text":"child answer"}]}]}}}`,
	} {
		WriteTestFile(t, path, []byte(content))
		if !mtime.IsZero() {
			require.NoError(t, os.Chtimes(path, mtime, mtime))
		}
	}
	WriteTestFile(t, anchor, []byte(pointer))
	if !mtime.IsZero() {
		require.NoError(t, os.Chtimes(anchor, mtime, mtime))
	}
	return anchor
}
