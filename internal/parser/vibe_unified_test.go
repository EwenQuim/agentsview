package parser

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const vibeUnifiedSessionID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"

func writeVibeUnifiedFile(t *testing.T, path, content string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	writeSourceFile(t, path, content)
}

func vibeUnifiedChunkFixture() string {
	return `[
  {"content":[{"text":"unified question","type":"text"}],"createdAt":1790601805237,"id":"e1","role":"user","source":"turn_start","turnId":"turn-1","type":"message"},
  {"createdAt":1790601807938,"id":"reasoning-1","text":"thinking it through","turnId":"turn-1","type":"reasoning"},
  {"content":[{"text":"unified answer","type":"text"}],"createdAt":1790601807938,"id":"assistant-1","role":"assistant","source":"harness","turnId":"turn-1","type":"message"},
  {"createdAt":1790601808027,"detail":{"kind":"tool","toolName":"file_system.bash","input":{"command":"ls"}},"id":"effect-1","state":{"output":{"content":[{"text":"file-a\nfile-b","type":"text"}],"type":"success"},"status":"completed"},"turnId":"turn-1","type":"effect"},
  {"createdAt":1790601808100,"id":"notice-1","level":"info","message":"Running hooks","type":"notice"}
]`
}

func writeVibeUnifiedSession(t *testing.T, root, sessionID string) string {
	t.Helper()
	sessionDir := filepath.Join(root, "unified", sessionID)
	gen := "0000000000000001"
	writeVibeUnifiedFile(t, filepath.Join(sessionDir, "chunks", "cafe0001.json"), vibeUnifiedChunkFixture())
	writeVibeUnifiedFile(t, filepath.Join(sessionDir, "CURRENT"),
		`{"generation":"`+gen+`","session_id":"`+sessionID+`","snapshot_sequence":1,`+
			`"store_format":"mistral.vibe.unified-session-store/v1"}`)
	writeVibeUnifiedFile(t, filepath.Join(sessionDir, "meta.json"),
		`{"session_id":"`+sessionID+`","parent_session_id":null,`+
			`"start_time":"2026-09-28T13:23:23.600000+00:00",`+
			`"end_time":"2026-09-28T13:32:53.039000+00:00",`+
			`"git_commit":null,"git_branch":null,"title":null,`+
			`"environment":{"working_directory":"/Users/dev/work/my-repo"}}`)
	genDir := filepath.Join(sessionDir, "generations", gen)
	writeVibeUnifiedFile(t, filepath.Join(genDir, "manifest.json"),
		`{"generation":"`+gen+`","session_id":"`+sessionID+`",`+
			`"projection_state":{"chunks":["cafe0001"]},`+
			`"checkpoint":{"chunks":[]},"snapshot_sequence":1}`)
	writeVibeUnifiedFile(t, filepath.Join(genDir, "projection-state.json"),
		`{"projection_state_version":1,"session_id":"`+sessionID+`","snapshot":{`+
			`"format":"harness.public-session-state/v1",`+
			`"session":{"createdAt":1790601803600,"id":"`+sessionID+`",`+
			`"parentSessionId":null,"title":null,"updatedAt":1790602373039,`+
			`"tokenUsage":{"cachedInputTokens":500,"inputTokens":1500,`+
			`"outputTokens":200,"totalTokens":1700},`+
			`"contextUsage":{"cachedInputTokens":400,"inputTokens":1400,`+
			`"outputTokens":50,"totalTokens":1450}}}}`)
	writeVibeUnifiedFile(t, filepath.Join(genDir, "runtime-state.json"),
		`{"identity":{"depth":0,"kind":"root","parent_session_id":null,`+
			`"root_session_id":"`+sessionID+`","session_id":"`+sessionID+`"},`+
			`"session_metadata":{"active_model":null,`+
			`"cwd":"/Users/dev/work/my-repo"}}`)
	return sessionDir
}

func TestVibeUnifiedProviderSourceMethods(t *testing.T) {
	root := t.TempDir()
	sessionDir := writeVibeUnifiedSession(t, root, vibeUnifiedSessionID)
	anchor := filepath.Join(sessionDir, "CURRENT")

	provider, ok := NewProvider(AgentVibe, ProviderConfig{Roots: []string{root}})
	require.True(t, ok)

	discovered, err := provider.Discover(t.Context())
	require.NoError(t, err)
	require.Len(t, discovered, 1)
	assert.Equal(t, anchor, discovered[0].DisplayPath)
	assert.Equal(t, vibeUnifiedSessionID, discovered[0].ProjectHint)

	found, ok, err := provider.FindSource(t.Context(), FindSourceRequest{
		RawSessionID: vibeUnifiedSessionID,
	})
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, anchor, found.DisplayPath)

	fingerprint, err := provider.Fingerprint(t.Context(), found)
	require.NoError(t, err)
	assert.Equal(t, anchor, fingerprint.Key)
	assert.NotEmpty(t, fingerprint.Hash)
	assert.Positive(t, fingerprint.Size)

	for _, changedPath := range []string{
		anchor,
		filepath.Join(sessionDir, "chunks", "cafe0001.json"),
		filepath.Join(sessionDir, "generations", "0000000000000001", "manifest.json"),
	} {
		changed, err := provider.SourcesForChangedPath(t.Context(), ChangedPathRequest{
			Path: changedPath, EventKind: "write", WatchRoot: root,
		})
		require.NoError(t, err)
		require.Len(t, changed, 1)
		assert.Equal(t, anchor, changed[0].DisplayPath)
	}

	require.NoError(t, os.RemoveAll(sessionDir))
	changed, err := provider.SourcesForChangedPath(t.Context(), ChangedPathRequest{
		Path:      filepath.Join(sessionDir, "chunks", "cafe0001.json"),
		EventKind: "remove",
		WatchRoot: root,
	})
	require.NoError(t, err)
	require.Len(t, changed, 1)
	assert.Equal(t, anchor, changed[0].DisplayPath)
}

func TestVibeUnifiedProviderParse(t *testing.T) {
	root := t.TempDir()
	writeVibeUnifiedSession(t, root, vibeUnifiedSessionID)

	provider, ok := NewProvider(AgentVibe, ProviderConfig{
		Roots:   []string{root},
		Machine: "devbox",
	})
	require.True(t, ok)
	sources, err := provider.Discover(t.Context())
	require.NoError(t, err)
	require.Len(t, sources, 1)
	fingerprint, err := provider.Fingerprint(t.Context(), sources[0])
	require.NoError(t, err)

	outcome, err := provider.Parse(t.Context(), ParseRequest{
		Source:      sources[0],
		Fingerprint: fingerprint,
	})
	require.NoError(t, err)
	require.True(t, outcome.ResultSetComplete)
	require.Len(t, outcome.Results, 1)
	result := outcome.Results[0]

	assert.Equal(t, "vibe:"+vibeUnifiedSessionID, result.Result.Session.ID)
	assert.Equal(t, AgentVibe, result.Result.Session.Agent)
	assert.Equal(t, "devbox", result.Result.Session.Machine)
	assert.Equal(t, "my_repo", result.Result.Session.Project)
	assert.Equal(t, "/Users/dev/work/my-repo", result.Result.Session.Cwd)
	assert.Equal(t, "unified question", result.Result.Session.FirstMessage)
	assert.Equal(t, 200, result.Result.Session.TotalOutputTokens)
	assert.True(t, result.Result.Session.HasTotalOutputTokens)
	assert.Equal(t, 1450, result.Result.Session.PeakContextTokens)
	assert.True(t, result.Result.Session.HasPeakContextTokens)
	assert.Equal(
		t, "2026-09-28T13:23:23Z", result.Result.Session.StartedAt.UTC().Format("2006-01-02T15:04:05Z07:00"),
	)

	messages := result.Result.Messages
	require.Len(t, messages, 4)
	assert.Equal(t, RoleUser, messages[0].Role)
	assert.Equal(t, "unified question", messages[0].Content)
	assert.Equal(t, RoleAssistant, messages[1].Role)
	assert.Equal(t, "unified answer", messages[1].Content)
	assert.True(t, messages[1].HasThinking)
	assert.Equal(t, "thinking it through", messages[1].ThinkingText)
	assert.Equal(t, RoleAssistant, messages[2].Role)
	require.Len(t, messages[2].ToolCalls, 1)
	assert.Equal(t, "effect-1", messages[2].ToolCalls[0].ToolUseID)
	assert.Equal(t, "file_system.bash", messages[2].ToolCalls[0].ToolName)
	assert.Equal(t, RoleUser, messages[3].Role)
	require.Len(t, messages[3].ToolResults, 1)
	assert.Equal(t, "effect-1", messages[3].ToolResults[0].ToolUseID)
	assert.Contains(t, messages[3].ToolResults[0].ContentRaw, "file-a")
}

func TestVibeUnifiedProviderParseEmitsUsageEventsFromRuntimeModel(t *testing.T) {
	root := t.TempDir()
	sessionDir := writeVibeUnifiedSession(t, root, vibeUnifiedSessionID)
	genDir := filepath.Join(sessionDir, "generations", "0000000000000001")
	writeVibeUnifiedFile(t, filepath.Join(genDir, "runtime-state.json"),
		`{"identity":{"depth":0,"kind":"root","parent_session_id":null,`+
			`"root_session_id":"`+vibeUnifiedSessionID+`","session_id":"`+vibeUnifiedSessionID+`"},`+
			`"session_metadata":{"active_model":"mistral-large-2411",`+
			`"cwd":"/Users/dev/work/my-repo"}}`)

	provider, ok := NewProvider(AgentVibe, ProviderConfig{Roots: []string{root}})
	require.True(t, ok)
	sources, err := provider.Discover(t.Context())
	require.NoError(t, err)
	require.Len(t, sources, 1)
	fingerprint, err := provider.Fingerprint(t.Context(), sources[0])
	require.NoError(t, err)
	outcome, err := provider.Parse(t.Context(), ParseRequest{
		Source:      sources[0],
		Fingerprint: fingerprint,
	})
	require.NoError(t, err)
	require.Len(t, outcome.Results, 1)

	usageEvents := outcome.Results[0].Result.UsageEvents
	require.Len(t, usageEvents, 1)
	assert.Equal(t, "vibe:"+vibeUnifiedSessionID, usageEvents[0].SessionID)
	assert.Equal(t, "mistral-large-2411", usageEvents[0].Model)
	assert.Equal(t, 1000, usageEvents[0].InputTokens)
	assert.Equal(t, 200, usageEvents[0].OutputTokens)
	assert.Equal(t, 500, usageEvents[0].CacheReadInputTokens)
	assert.Equal(t, "session:vibe:"+vibeUnifiedSessionID, usageEvents[0].DedupKey)
}

// The unified session store records the per-session model only when the user
// overrode it for that session; sessions on the config default must fall back
// to the CLI's active_model so their usage is still attributable.
func TestVibeUnifiedProviderParseFallsBackToConfigDefaultModel(t *testing.T) {
	home := t.TempDir()
	root := filepath.Join(home, "logs", "session")
	writeVibeUnifiedSession(t, root, vibeUnifiedSessionID)
	writeVibeUnifiedFile(t, filepath.Join(home, "config.toml"),
		"active_model = \"glm-5-3\"\n")

	provider, ok := NewProvider(AgentVibe, ProviderConfig{Roots: []string{root}})
	require.True(t, ok)
	sources, err := provider.Discover(t.Context())
	require.NoError(t, err)
	require.Len(t, sources, 1)
	fingerprint, err := provider.Fingerprint(t.Context(), sources[0])
	require.NoError(t, err)
	outcome, err := provider.Parse(t.Context(), ParseRequest{
		Source:      sources[0],
		Fingerprint: fingerprint,
	})
	require.NoError(t, err)
	require.Len(t, outcome.Results, 1)

	usageEvents := outcome.Results[0].Result.UsageEvents
	require.Len(t, usageEvents, 1)
	assert.Equal(t, "glm-5-3", usageEvents[0].Model)
}

// Subagent sessions have no meta.json; identity comes from runtime-state and
// the projection snapshot, and the directory prefix marks the relationship.
func TestVibeUnifiedProviderParseSubagentSession(t *testing.T) {
	root := t.TempDir()
	childID := "child-6f3c8f5f244fec85b81fd2a2"
	parentID := "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
	sessionDir := writeVibeUnifiedSession(t, root, childID)
	require.NoError(t, os.Remove(filepath.Join(sessionDir, "meta.json")))
	genDir := filepath.Join(sessionDir, "generations", "0000000000000001")
	writeVibeUnifiedFile(t, filepath.Join(genDir, "runtime-state.json"),
		`{"identity":{"depth":1,"kind":"subagent",`+
			`"parent_session_id":"`+parentID+`",`+
			`"root_session_id":"`+parentID+`","session_id":"`+childID+`"},`+
			`"session_metadata":{"active_model":"mistral-large-2411",`+
			`"cwd":"/Users/dev/work/my-repo"}}`)

	provider, ok := NewProvider(AgentVibe, ProviderConfig{Roots: []string{root}})
	require.True(t, ok)
	sources, err := provider.Discover(t.Context())
	require.NoError(t, err)
	require.Len(t, sources, 1)
	fingerprint, err := provider.Fingerprint(t.Context(), sources[0])
	require.NoError(t, err)
	outcome, err := provider.Parse(t.Context(), ParseRequest{
		Source:      sources[0],
		Fingerprint: fingerprint,
	})
	require.NoError(t, err)
	require.Len(t, outcome.Results, 1)
	session := outcome.Results[0].Result.Session

	assert.Equal(t, "vibe:"+childID, session.ID)
	assert.Equal(t, "vibe:"+parentID, session.ParentSessionID)
	assert.Equal(t, RelSubagent, session.RelationshipType)
	assert.Equal(t, "my_repo", session.Project)
}
