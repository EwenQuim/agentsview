package sync_test

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.kenn.io/agentsview/internal/db"
	"go.kenn.io/agentsview/internal/dbtest"
	"go.kenn.io/agentsview/internal/parser"
	"go.kenn.io/agentsview/internal/sync"
)

// writeVibeSyncFixture writes a Vibe session directory with a messages.jsonl
// transcript and a sibling meta.json. It returns the two file paths.
func writeVibeSyncFixture(
	t *testing.T, root, dirName, sessionID, title string,
) (messagesPath, metaPath string) {
	t.Helper()

	sessionDir := filepath.Join(root, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")

	messagesPath = filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")

	metaPath = filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"`+title+`",`+
			`"model":"mistral-medium-3.5",`+
			`"stats":{"session_prompt_tokens":100,"session_completion_tokens":50}}`+"\n"),
		0o644,
	), "write meta.json")
	return messagesPath, metaPath
}

func TestSyncAllSinceVibeMetaUpdateTriggersResync(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	sessionID := "abc123def-0000-0000-0000-000000000000"
	messagesPath, metaPath := writeVibeSyncFixture(
		t, vibeDir, "session_20260616_083518_abc123", sessionID, "Before rename",
	)

	engine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, "vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Before rename", *sess.DisplayName)
	})

	// meta.json-only update: the transcript is untouched (and left below the
	// cutoff), but the title in meta.json changes and its mtime moves ahead.
	transcriptTime := time.Unix(1_781_475_210, 0)
	metaTime := transcriptTime.Add(time.Second)
	require.NoError(t, os.Chtimes(messagesPath, transcriptTime, transcriptTime))
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"After rename",`+
			`"model":"mistral-medium-3.5",`+
			`"stats":{"session_prompt_tokens":100,"session_completion_tokens":50}}`+"\n"),
		0o644,
	), "rewrite meta.json")
	require.NoError(t, os.Chtimes(metaPath, metaTime, metaTime))

	cutoff := transcriptTime.Add(500 * time.Millisecond)
	stats := engine.SyncAllSince(t.Context(), cutoff, nil)
	require.Equal(t, 1, stats.Synced, "synced = %d, want 1", stats.Synced)

	assertSessionState(t, testDB, "vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "After rename", *sess.DisplayName)
	})
}

func TestSyncAllSinceVibeUnifiedGenerationTriggersResync(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	root := t.TempDir()
	database := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), database, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{parser.AgentVibe: {root}},
		Machine:   "local",
	})
	dir := filepath.Join(root, "unified", "session-a")
	anchor := filepath.Join(dir, "CURRENT")
	baseTime := time.Unix(1_791_000_000, 0)
	childDir := filepath.Join(root, "unified", "session-child")
	childAnchor := dbtest.WriteVibeUnifiedGeneration(t, childDir, "1", "session-a", "", baseTime)
	for i, model := range []string{"mistral-medium-3.5", "mistral-small-latest"} {
		generation := strconv.Itoa(i + 1)
		dbtest.WriteVibeUnifiedGeneration(t, dir, generation, "", model, baseTime.Add(time.Duration(i+1)*time.Second))
		if i == 0 {
			engine.SyncPaths([]string{anchor, childAnchor})
		} else {
			stats := engine.SyncAllSince(t.Context(), baseTime.Add(1500*time.Millisecond), nil)
			assert.Equal(t, 2, stats.Synced)
		}
	}
	stats := engine.SyncAllSince(t.Context(), baseTime.Add(1500*time.Millisecond), nil)
	assert.Zero(t, stats.Synced)
}

func TestSyncAllSinceVibeMissingParentRefreshesInheritedModel(t *testing.T) {
	root := t.TempDir()
	database := dbtest.OpenTestDB(t)
	config := sync.EngineConfig{AgentDirs: map[parser.AgentType][]string{parser.AgentVibe: {root}}, Machine: "local"}
	baseTime := time.Unix(1_791_000_000, 0)
	parent := dbtest.WriteVibeUnifiedGeneration(t, filepath.Join(root, "unified", "parent"), "1", "", "model-a", baseTime.Add(time.Second))
	child := dbtest.WriteVibeUnifiedGeneration(t, filepath.Join(root, "unified", "child"), "1", "parent", "", baseTime)
	engine := sync.NewEngine(t.Context(), database, config)
	engine.SyncPaths([]string{parent, child})
	engine.Close()
	usage, err := database.GetUsageEvents(t.Context(), "vibe:child")
	require.NoError(t, err)
	require.Len(t, usage, 1)
	require.Equal(t, "model-a", usage[0].Model)

	// No watcher observes the deletion while the engine is stopped.
	require.NoError(t, os.Remove(parent))
	engine = sync.NewEngine(t.Context(), database, config)
	t.Cleanup(engine.Close)
	stats := engine.SyncAllSince(t.Context(), baseTime.Add(2*time.Second), nil)
	assert.Equal(t, 1, stats.Synced)
	usage, err = database.GetUsageEvents(t.Context(), "vibe:child")
	require.NoError(t, err)
	require.Len(t, usage, 1)
	assert.Empty(t, usage[0].Model)
	stats = engine.SyncAllSince(t.Context(), baseTime.Add(2*time.Second), nil)
	assert.Zero(t, stats.Synced)
}

func TestSyncVibeParentOnlyModelChanges(t *testing.T) {
	for _, name := range []string{"SyncPaths", "metadata event", "metadata removal", "watcher plan"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			database := dbtest.OpenTestDB(t)
			engine := sync.NewEngine(t.Context(), database, sync.EngineConfig{
				AgentDirs: map[parser.AgentType][]string{parser.AgentVibe: {root}},
				Machine:   "local",
			})
			t.Cleanup(engine.Close)
			baseTime := time.Unix(1_791_000_000, 0)
			parentDir := filepath.Join(root, "unified", "parent")
			parentAnchor := filepath.Join(parentDir, "CURRENT")
			childDir := filepath.Join(root, "unified", "child")
			childAnchor := dbtest.WriteVibeUnifiedGeneration(t, childDir, "1", "parent", "", baseTime.Add(time.Hour))
			var previousToken int64
			var previousRevision string
			var previousUsageID int64
			for i, model := range []string{"", "vibe-test-model-a", "vibe-test-model-b", "vibe-test-model-b"} {
				generation := strconv.Itoa(i + 1)
				genDir := filepath.Join(parentDir, "generations", generation)
				require.NoError(t, os.MkdirAll(genDir, 0o755))
				for path, content := range map[string]string{
					parentAnchor:                                   `{"generation":"` + generation + `"}`,
					filepath.Join(genDir, "manifest.json"):         `{}`,
					filepath.Join(genDir, "runtime-state.json"):    `{"runtime_state_version":3,"session_metadata":{"active_model":"` + model + `"}}`,
					filepath.Join(genDir, "projection-state.json"): `{"snapshot":{"history":{"entries":[{"type":"message","role":"user","content":[{"text":"parent question"}]}]}}}`,
				} {
					require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
					mtime := baseTime.Add(time.Duration(i) * time.Second)
					if i == 3 {
						mtime = baseTime.Add(2 * time.Hour)
					}
					require.NoError(t, os.Chtimes(path, mtime, mtime))
				}
				if i == 0 {
					engine.SyncPaths([]string{parentAnchor, childAnchor})
				} else if name == "watcher plan" {
					plan, err := engine.PlanChangedPathsContext(t.Context(), []string{parentAnchor})
					require.NoError(t, err)
					paths := make([]string, 0, len(plan.Files))
					for _, file := range plan.Files {
						paths = append(paths, file.Path)
					}
					assert.ElementsMatch(t, []string{parentAnchor, childAnchor}, paths)
					_, err = engine.SyncChangedPathPlanContext(t.Context(), plan, nil)
					require.NoError(t, err)
				} else if name == "metadata event" || name == "metadata removal" {
					metaPath := filepath.Join(parentDir, "meta.json")
					require.NoError(t, os.WriteFile(metaPath, []byte(`{"title":"updated parent title"}`), 0o644))
					if name == "metadata removal" {
						require.NoError(t, os.Remove(metaPath))
					}
					engine.SyncPaths([]string{metaPath})
				} else {
					engine.SyncPaths([]string{parentAnchor})
				}
				usage, err := database.GetUsageEvents(t.Context(), "vibe:child")
				require.NoError(t, err)
				require.Len(t, usage, 1)
				assert.Equal(t, model, usage[0].Model)
				token := engine.SourceMtime(t.Context(), "vibe:child")
				assert.NotZero(t, token)
				revision, err := database.TranscriptRevision(t.Context(), "vibe:child")
				require.NoError(t, err)
				if i == 3 {
					assert.Equal(t, previousToken, token)
					assert.Equal(t, previousRevision, revision)
					assert.Equal(t, previousUsageID, usage[0].ID)
				}
				if i > 0 && i < 3 {
					assert.NotEqual(t, previousToken, token, "parent model must change the polling token despite the later child mtime")
				}
				previousToken = token
				previousRevision = revision
				previousUsageID = usage[0].ID
			}
		})
	}
}

func TestSourceMtimeVibeIncludesMetaMtime(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	sessionID := "abc123def-0000-0000-0000-000000000000"
	messagesPath, metaPath := writeVibeSyncFixture(
		t, vibeDir, "session_20260616_083518_abc123", sessionID, "Source mtime",
	)

	transcriptTime := time.Unix(1_781_475_210, 0)
	metaTime := transcriptTime.Add(time.Second)
	require.NoError(t, os.Chtimes(messagesPath, transcriptTime, transcriptTime))
	require.NoError(t, os.Chtimes(metaPath, metaTime, metaTime))

	engine.SyncPaths([]string{messagesPath})
	assert.Equal(t, metaTime.UnixNano(), engine.SourceMtime(t.Context(), "vibe:"+sessionID))
}

// TestSyncVibeCorruptMetaRetriesAfterMetaFixed verifies that a parse error
// caused by a corrupt meta.json is not permanently skip-cached against the
// transcript mtime. The skip-cache key uses the Vibe effective mtime (max of
// messages.jsonl and meta.json), so fixing meta.json (which advances only its
// mtime) invalidates the cached skip and the next sync reparses the session.
func TestSyncVibeCorruptMetaRetriesAfterMetaFixed(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	sessionDir := filepath.Join(vibeDir, "session_20260616_083518_abc123")
	require.NoError(t, os.MkdirAll(sessionDir, 0o755))
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	))
	// A truncated/partial write: not even minimally valid JSON, so the parse
	// fails and the file is skip-cached.
	metaPath := filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(metaPath, []byte(`{"session_id":"abc`), 0o644))

	baseTime := time.Unix(1_781_475_210, 0)
	require.NoError(t, os.Chtimes(messagesPath, baseTime, baseTime))
	require.NoError(t, os.Chtimes(metaPath, baseTime, baseTime))

	sessionID := "abc123def-0000-0000-0000-000000000000"
	canonicalID := "vibe:" + sessionID

	// First sync fails to parse and caches a skip at the effective mtime.
	engine.SyncPaths([]string{messagesPath})
	got, err := testDB.GetSession(t.Context(), canonicalID)
	require.NoError(t, err)
	assert.Nil(t, got, "corrupt meta.json must not produce a session")

	// Fix meta.json and advance only its mtime; the transcript mtime is
	// unchanged, so a skip cache keyed on the transcript alone would wrongly
	// skip this file forever.
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"Recovered"}`+"\n"),
		0o644,
	))
	metaTime := baseTime.Add(time.Hour)
	require.NoError(t, os.Chtimes(metaPath, metaTime, metaTime))

	engine.SyncPaths([]string{messagesPath})
	got, err = testDB.GetSession(t.Context(), canonicalID)
	require.NoError(t, err)
	require.NotNil(t, got, "fixed meta.json must reparse instead of staying skipped")
	assert.Equal(t, canonicalID, got.ID)
}

// TestSyncVibeMetaPromotionRemovesFallbackID verifies that when a session is
// first synced without meta.json (stored under the directory-name fallback ID)
// and meta.json later appears, the session is re-stored under the meta
// session_id and the stale fallback row is removed rather than left behind to
// double-count messages and usage.
func TestSyncVibeMetaPromotionRemovesFallbackID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	dirName := "session_20260616_083518_abc123"
	sessionDir := filepath.Join(vibeDir, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")

	// First sync without meta.json: stored under the directory-name fallback.
	engine.SyncPaths([]string{messagesPath})
	fallbackID := "vibe:" + dirName
	assertSessionState(t, testDB, fallbackID, nil)

	// meta.json appears with a distinct session_id (a uuid).
	sessionID := "abc123def-0000-0000-0000-000000000000"
	metaPath := filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"Promoted"}`+"\n"),
		0o644,
	), "write meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(metaPath, future, future))

	engine.SyncPaths([]string{messagesPath})

	// The promoted ID exists, the stale fallback row is gone.
	assertSessionState(t, testDB, "vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Promoted", *sess.DisplayName)
	})
	gone, err := testDB.GetSession(t.Context(), fallbackID)
	require.NoError(t, err)
	assert.Nil(t, gone, "stale fallback session row must be deleted")
}

// TestSyncSingleSessionVibeMetaPromotionRemovesFallbackID verifies that the
// single-session resync path (used by manual re-sync and the session watcher)
// also removes the stale directory-name fallback row when meta.json promotes
// the session to its canonical ID. Without consuming the parser-excluded IDs on
// this path, both the fallback and the canonical row would linger and
// double-count messages and usage.
func TestSyncSingleSessionVibeMetaPromotionRemovesFallbackID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	dirName := "session_20260616_083518_abc123"
	sessionDir := filepath.Join(vibeDir, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")

	// First sync without meta.json: stored under the directory-name fallback.
	engine.SyncPaths([]string{messagesPath})
	fallbackID := "vibe:" + dirName
	assertSessionState(t, testDB, fallbackID, nil)

	// meta.json appears with a distinct session_id (a uuid).
	sessionID := "abc123def-0000-0000-0000-000000000000"
	metaPath := filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"Promoted"}`+"\n"),
		0o644,
	), "write meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(metaPath, future, future))

	// Resync via the single-session path keyed by the fallback ID, as the
	// session watcher would after the fallback row already exists.
	require.NoError(t, engine.SyncSingleSession(fallbackID))

	// The promoted ID exists, the stale fallback row is gone.
	assertSessionState(t, testDB, "vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Promoted", *sess.DisplayName)
	})
	gone, err := testDB.GetSession(t.Context(), fallbackID)
	require.NoError(t, err)
	assert.Nil(t, gone, "stale fallback session row must be deleted")
}

func TestSyncVibeMetaPromotionHonorsHiddenFallbackIDs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	tests := []struct {
		name      string
		dirName   string
		sessionID string
		hide      func(*testing.T, string)
		wantMsg   string
	}{
		{
			name:      "deleted fallback",
			dirName:   "session_20260616_083518_deleted",
			sessionID: "abc123def-0000-0000-0000-000000000001",
			hide: func(t *testing.T, fallbackID string) {
				t.Helper()
				require.NoError(t, testDB.DeleteSession(t.Context(), fallbackID),
					"delete fallback")
				assert.True(t, testDB.IsSessionExcluded(t.Context(), fallbackID),
					"fallback ID should be permanently excluded")
			},
			wantMsg: "promoted canonical ID must not resurrect deleted fallback session",
		},
		{
			name:      "trashed fallback",
			dirName:   "session_20260616_083518_trashed",
			sessionID: "abc123def-0000-0000-0000-000000000002",
			hide: func(t *testing.T, fallbackID string) {
				t.Helper()

				require.NoError(t, testDB.SoftDeleteSession(t.Context(), fallbackID),
					"trash fallback")
				trashed, err := testDB.GetSessionFull(
					t.Context(), fallbackID,
				)
				require.NoError(t, err)
				require.NotNil(t, trashed, "trashed fallback row")
				require.NotNil(t, trashed.DeletedAt,
					"fallback should be trashed")
			},
			wantMsg: "promoted canonical ID must not resurrect trashed fallback session",
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sessionDir := filepath.Join(vibeDir, tt.dirName)
			require.NoError(t, os.MkdirAll(sessionDir, 0o755),
				"mkdir vibe session dir")
			messagesPath := filepath.Join(sessionDir, "messages.jsonl")
			require.NoError(t, os.WriteFile(
				messagesPath,
				[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
				0o644,
			), "write messages.jsonl")

			fallbackID := "vibe:" + tt.dirName
			engine.SyncPaths([]string{messagesPath})
			assertSessionState(t, testDB, fallbackID, nil)
			tt.hide(t, fallbackID)

			metaPath := filepath.Join(sessionDir, "meta.json")
			require.NoError(t, os.WriteFile(
				metaPath,
				[]byte(`{"session_id":"`+tt.sessionID+
					`","title":"Promoted"}`+"\n"),
				0o644,
			), "write meta.json")
			future := time.Now().Add(time.Duration(i+1) * time.Hour)
			require.NoError(t, os.Chtimes(metaPath, future, future))

			engine.SyncPaths([]string{messagesPath})

			canonical, err := testDB.GetSession(
				t.Context(), "vibe:"+tt.sessionID,
			)
			require.NoError(t, err)
			assert.Nil(t, canonical, tt.wantMsg)
		})
	}
}

func TestSyncVibeRemotePromotionIgnoresDeletedLocalFallbackID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	localEngine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})
	remoteEngine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine:  "remote-host",
		IDPrefix: "host~",
		PathRewriter: func(path string) string {
			return "host:" + path
		},
	})

	dirName := "session_20260616_083518_abc123"
	sessionDir := filepath.Join(vibeDir, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")

	localFallbackID := "vibe:" + dirName
	localEngine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, localFallbackID, nil)
	require.NoError(t, testDB.DeleteSession(t.Context(), localFallbackID), "delete local fallback")
	assert.True(t, testDB.IsSessionExcluded(t.Context(), localFallbackID),
		"local fallback ID should be permanently excluded")

	sessionID := "abc123def-0000-0000-0000-000000000000"
	metaPath := filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"Remote"}`+"\n"),
		0o644,
	), "write meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(metaPath, future, future))

	remoteEngine.SyncPaths([]string{messagesPath})

	assertSessionState(t, testDB, "host~vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Remote", *sess.DisplayName)
	})
}

func TestSyncVibeRemotePromotionIgnoresTrashedLocalFallbackID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	localEngine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})
	remoteEngine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine:  "remote-host",
		IDPrefix: "host~",
		PathRewriter: func(path string) string {
			return "host:" + path
		},
	})

	dirName := "session_20260616_083518_abc123"
	sessionDir := filepath.Join(vibeDir, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")

	localFallbackID := "vibe:" + dirName
	localEngine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, localFallbackID, nil)
	require.NoError(t, testDB.SoftDeleteSession(t.Context(), localFallbackID), "trash local fallback")
	trashed, err := testDB.GetSessionFull(t.Context(), localFallbackID)
	require.NoError(t, err)
	require.NotNil(t, trashed, "trashed local fallback row")
	require.NotNil(t, trashed.DeletedAt, "local fallback should be trashed")

	sessionID := "abc123def-0000-0000-0000-000000000000"
	metaPath := filepath.Join(sessionDir, "meta.json")
	require.NoError(t, os.WriteFile(
		metaPath,
		[]byte(`{"session_id":"`+sessionID+`","title":"Remote"}`+"\n"),
		0o644,
	), "write meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(metaPath, future, future))

	remoteEngine.SyncPaths([]string{messagesPath})

	assertSessionState(t, testDB, "host~vibe:"+sessionID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Remote", *sess.DisplayName)
	})
}

func TestSyncVibeMissingMetaHonorsHiddenCanonicalIDs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	tests := []struct {
		name      string
		dirName   string
		sessionID string
		hide      func(*testing.T, string)
		wantMsg   string
	}{
		{
			name:      "deleted canonical",
			dirName:   "session_20260616_083518_deleted_canonical",
			sessionID: "abc123def-0000-0000-0000-000000000101",
			hide: func(t *testing.T, canonicalID string) {
				t.Helper()
				require.NoError(t, testDB.DeleteSession(t.Context(), canonicalID),
					"delete canonical")
				assert.True(t, testDB.IsSessionExcluded(t.Context(), canonicalID),
					"canonical ID should be permanently excluded")
			},
			wantMsg: "fallback ID must not resurrect deleted canonical session",
		},
		{
			name:      "trashed canonical",
			dirName:   "session_20260616_083518_trashed_canonical",
			sessionID: "abc123def-0000-0000-0000-000000000102",
			hide: func(t *testing.T, canonicalID string) {
				t.Helper()

				require.NoError(t, testDB.SoftDeleteSession(t.Context(), canonicalID),
					"trash canonical")
				trashed, err := testDB.GetSessionFull(
					t.Context(), canonicalID,
				)
				require.NoError(t, err)
				require.NotNil(t, trashed, "trashed canonical row")
				require.NotNil(t, trashed.DeletedAt,
					"canonical should be trashed")
			},
			wantMsg: "fallback ID must not resurrect trashed canonical session",
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messagesPath, metaPath := writeVibeSyncFixture(
				t, vibeDir, tt.dirName, tt.sessionID, "Canonical",
			)
			canonicalID := "vibe:" + tt.sessionID
			fallbackID := "vibe:" + tt.dirName

			engine.SyncPaths([]string{messagesPath})
			assertSessionState(t, testDB, canonicalID, nil)
			tt.hide(t, canonicalID)

			require.NoError(t, os.Remove(metaPath), "remove meta.json")
			future := time.Now().Add(time.Duration(i+1) * time.Hour)
			require.NoError(t, os.Chtimes(messagesPath, future, future))

			engine.SyncPaths([]string{messagesPath})

			fallback, err := testDB.GetSession(
				t.Context(), fallbackID,
			)
			require.NoError(t, err)
			assert.Nil(t, fallback, tt.wantMsg)
		})
	}
}

func TestSyncVibeMissingMetaKeepsTrashedFallbackID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	dirName := "session_20260616_083518_abc123"
	sessionDir := filepath.Join(vibeDir, dirName)
	require.NoError(t, os.MkdirAll(sessionDir, 0o755), "mkdir vibe session dir")
	messagesPath := filepath.Join(sessionDir, "messages.jsonl")
	require.NoError(t, os.WriteFile(
		messagesPath,
		[]byte(`{"role":"user","content":"hello vibe"}`+"\n"),
		0o644,
	), "write messages.jsonl")
	fallbackID := "vibe:" + dirName

	engine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, fallbackID, nil)
	require.NoError(t, testDB.SoftDeleteSession(t.Context(), fallbackID), "trash fallback")

	for i := range 2 {
		require.NoError(t, os.WriteFile(
			messagesPath,
			[]byte(`{"role":"user","content":"hello vibe `+
				strconv.Itoa(i)+`"}`+"\n"),
			0o644,
		), "rewrite messages.jsonl")
		future := time.Now().Add(time.Duration(i+1) * time.Hour)
		require.NoError(t, os.Chtimes(messagesPath, future, future))

		engine.SyncPaths([]string{messagesPath})

		visible, err := testDB.GetSession(t.Context(), fallbackID)
		require.NoError(t, err)
		assert.Nil(t, visible, "trashed fallback must stay hidden")

		trashed, err := testDB.GetSessionFull(t.Context(), fallbackID)
		require.NoError(t, err)
		require.NotNil(t, trashed, "trashed fallback row")
		require.NotNil(t, trashed.DeletedAt, "fallback should remain trashed")
		assert.False(t, testDB.IsSessionExcluded(t.Context(), fallbackID),
			"parser cleanup must not permanently exclude the trashed fallback")
	}
}

func TestSyncVibeMissingMetaRemovesCanonicalID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	dirName := "session_20260616_083518_abc123"
	sessionID := "abc123def-0000-0000-0000-000000000000"
	messagesPath, metaPath := writeVibeSyncFixture(
		t, vibeDir, dirName, sessionID, "Canonical",
	)
	canonicalID := "vibe:" + sessionID
	fallbackID := "vibe:" + dirName

	engine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, canonicalID, func(sess *db.Session) {
		require.NotNil(t, sess.DisplayName)
		assert.Equal(t, "Canonical", *sess.DisplayName)
	})

	require.NoError(t, os.Remove(metaPath), "remove meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(messagesPath, future, future))

	engine.SyncPaths([]string{messagesPath})

	assertSessionState(t, testDB, fallbackID, nil)
	gone, err := testDB.GetSession(t.Context(), canonicalID)
	require.NoError(t, err)
	assert.Nil(t, gone, "stale canonical session row must be deleted")

	var rowsForPath int
	require.NoError(t, testDB.Reader().QueryRow(t.Context(),
		`SELECT COUNT(*) FROM sessions WHERE file_path = ?`,
		messagesPath,
	).Scan(&rowsForPath), "count rows for vibe file path")
	assert.Equal(t, 1, rowsForPath,
		"vibe file path must not leave duplicate session rows")
}

func TestSyncPathsVibeDeletedMetaPathRemovesCanonicalID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine: "local",
	})

	dirName := "session_20260616_083518_abc123"
	sessionID := "abc123def-0000-0000-0000-000000000000"
	messagesPath, metaPath := writeVibeSyncFixture(
		t, vibeDir, dirName, sessionID, "Canonical",
	)
	canonicalID := "vibe:" + sessionID
	fallbackID := "vibe:" + dirName

	engine.SyncPaths([]string{messagesPath})
	assertSessionState(t, testDB, canonicalID, nil)

	require.NoError(t, os.Remove(metaPath), "remove meta.json")

	engine.SyncPaths([]string{metaPath})

	assertSessionState(t, testDB, fallbackID, nil)
	gone, err := testDB.GetSession(t.Context(), canonicalID)
	require.NoError(t, err)
	assert.Nil(t, gone, "deleted meta.json event must remove stale canonical row")
}

func TestSyncVibeMissingMetaRemotePathRemovesCanonicalID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	vibeDir := t.TempDir()
	testDB := dbtest.OpenTestDB(t)
	rewriter := func(path string) string {
		return "host:" + path
	}
	engine := sync.NewEngine(t.Context(), testDB, sync.EngineConfig{
		AgentDirs: map[parser.AgentType][]string{
			parser.AgentVibe: {vibeDir},
		},
		Machine:      "remote-host",
		IDPrefix:     "host~",
		PathRewriter: rewriter,
	})

	dirName := "session_20260616_083518_abc123"
	sessionID := "abc123def-0000-0000-0000-000000000000"
	messagesPath, metaPath := writeVibeSyncFixture(
		t, vibeDir, dirName, sessionID, "Canonical",
	)
	canonicalID := "host~vibe:" + sessionID
	fallbackID := "host~vibe:" + dirName

	engine.SyncPaths([]string{messagesPath})
	canonical, err := testDB.GetSessionFull(t.Context(), canonicalID)
	require.NoError(t, err)
	require.NotNil(t, canonical, "canonical remote session")
	require.NotNil(t, canonical.FilePath)
	assert.Equal(t, rewriter(messagesPath), *canonical.FilePath)

	require.NoError(t, os.Remove(metaPath), "remove meta.json")
	future := time.Now().Add(time.Hour)
	require.NoError(t, os.Chtimes(messagesPath, future, future))

	engine.SyncPaths([]string{messagesPath})

	assertSessionState(t, testDB, fallbackID, nil)
	gone, err := testDB.GetSession(t.Context(), canonicalID)
	require.NoError(t, err)
	assert.Nil(t, gone, "stale remote canonical session row must be deleted")

	var rowsForPath int
	require.NoError(t, testDB.Reader().QueryRow(t.Context(),
		`SELECT COUNT(*) FROM sessions WHERE file_path = ?`,
		rewriter(messagesPath),
	).Scan(&rowsForPath), "count rows for rewritten vibe file path")
	assert.Equal(t, 1, rowsForPath,
		"rewritten vibe file path must not leave duplicate remote rows")
}

func TestSyncVibeUnsupportedVersionsPreserveArchive(t *testing.T) {
	root := t.TempDir()
	database := dbtest.OpenTestDB(t)
	engine := sync.NewEngine(t.Context(), database, sync.EngineConfig{AgentDirs: map[parser.AgentType][]string{parser.AgentVibe: {root}}, Machine: "local"})
	t.Cleanup(engine.Close)
	mtime := time.Unix(1_791_000_000, 0)
	childDir := filepath.Join(root, "unified", "child")
	child := dbtest.WriteVibeUnifiedGeneration(t, childDir, "1", "", "", mtime)
	engine.SyncPaths([]string{child})
	revision, err := database.TranscriptRevision(t.Context(), "vibe:child")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(child, []byte(`{"generation":"1","store_format_minor":9}`), 0o644))
	// Move CURRENT so fingerprint and failure caches see this new generation.
	require.NoError(t, os.Chtimes(child, mtime.Add(time.Second), mtime.Add(time.Second)))
	engine.SyncPaths([]string{child})
	after, err := database.TranscriptRevision(t.Context(), "vibe:child")
	require.NoError(t, err)
	assert.Equal(t, revision, after)
	messages, err := database.GetMessages(t.Context(), "vibe:child", 0, 10, true)
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "child answer", messages[0].Content)
}
