package ingest_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kenn.io/agentsview/internal/db"
	"go.kenn.io/agentsview/internal/dbtest"
	"go.kenn.io/agentsview/internal/ingest"
	"go.kenn.io/agentsview/internal/parser"
	"go.kenn.io/agentsview/internal/signals"
)

func TestProviderToolResultFailureMetadata(t *testing.T) {
	for _, tt := range []struct {
		name, result, content, status string
		agent                         parser.AgentType
	}{
		{"claude_error", `{"type":"tool_result","tool_use_id":"call-1","content":"File does not exist.","is_error":true}`, "File does not exist.", "errored", parser.AgentClaude},
		{"claude_unknown", `{"type":"tool_result","tool_use_id":"call-1","content":"file contents"}`, "file contents", "", parser.AgentClaude},
		{"amp_error", `{"type":"tool_result","toolUseID":"call-1","run":{"status":"error","error":{"message":"File does not exist."}}}`, "File does not exist.", "errored", parser.AgentAmp},
		{"amp_unsuccessful", `{"type":"tool_result","toolUseID":"call-1","run":{"status":"done","result":{"success":false}}}`, "failed", "errored", parser.AgentAmp},
		{"amp_unknown", `{"type":"tool_result","toolUseID":"call-1","run":{"result":"file contents"}}`, "file contents", "", parser.AgentAmp},
	} {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			provider, ok := parser.NewProvider(tt.agent, parser.ProviderConfig{Roots: []string{root}, Machine: "local"})
			require.True(t, ok)
			var parsed parser.ParseResult
			if tt.agent == parser.AgentClaude {
				raw := fmt.Sprintf(`{"type":"user","uuid":"u1","message":{"content":"Read the file"}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"content":[{"type":"tool_use","id":"call-1","name":"Read","input":{"file_path":"missing.go"}}]}}
{"type":"user","uuid":"r1","parentUuid":"a1","message":{"content":[%s]}}
`, tt.result)
				path := filepath.Join(root, "session.jsonl")
				require.NoError(t, os.WriteFile(path, []byte(raw), 0o600))
				uploader, ok := provider.(parser.ClaudeUploadParser)
				require.True(t, ok)
				results, err := uploader.ParseUploadedTranscript(path, "project-a", "local")
				require.NoError(t, err)
				require.Len(t, results, 1)
				parsed = results[0]
			} else {
				raw := fmt.Sprintf(`{"id":"T-019ca26f-aaaa-bbbb-cccc-dddddddddddd","created":1704067200000,"messages":[{"role":"user","content":[{"type":"text","text":"Read the file"}]},{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"Read","input":{"path":"missing.go"}}]},{"role":"user","content":[%s]}]}`, tt.result)
				require.NoError(t, os.WriteFile(filepath.Join(root, "T-019ca26f-aaaa-bbbb-cccc-dddddddddddd.json"), []byte(raw), 0o600))
				sources, err := provider.Discover(t.Context())
				require.NoError(t, err)
				require.Len(t, sources, 1)
				outcome, err := provider.Parse(t.Context(), parser.ParseRequest{Source: sources[0]})
				require.NoError(t, err)
				require.Len(t, outcome.Results, 1)
				parsed = outcome.Results[0].Result
			}
			candidate, err := ingest.PrepareCandidate(t.Context(), parsed, ingest.ContentOptions{})
			require.NoError(t, err)
			prepared, err := ingest.Finalize(t.Context(), candidate, ingest.ContentOptions{})
			require.NoError(t, err)
			database := dbtest.OpenTestDB(t)
			written, err := database.WriteSessionBatch([]db.SessionBatchWrite{{Session: prepared.Session, Messages: prepared.Messages}})
			require.NoError(t, err)
			require.Empty(t, written.Errors)
			stored, err := database.GetMessages(t.Context(), prepared.Session.ID, 0, 100, true)
			require.NoError(t, err)
			rows := ingest.ExtractToolCallRows(stored)
			require.Len(t, rows, 1)
			assert.Equal(t, tt.content, rows[0].ResultContent)
			assert.Equal(t, tt.status, rows[0].EventStatus)
			assert.Equal(t, tt.status == "errored", signals.ComputeToolHealth(rows).FailureSignalCount == 1)
			for _, message := range stored {
				for _, call := range message.ToolCalls {
					require.Len(t, call.ResultEvents, 1)
					assert.Equal(t, tt.content, call.ResultEvents[0].Content)
				}
			}
		})
	}
}

func TestVibeUnifiedToolResultEvents(t *testing.T) {
	for _, tt := range []struct {
		status, wantStatus string
		wantFailures       int
	}{
		{"completed", "completed", 0},
		{"failed", "errored", 1},
		{"skipped", "errored", 1},
		{"cancelled", "cancelled", 1},
	} {
		t.Run(tt.status, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "unified", "session-a")
			gen := filepath.Join(dir, "generations", "1")
			require.NoError(t, os.MkdirAll(gen, 0o700))
			for name, content := range map[string]string{
				filepath.Join(dir, "CURRENT"):               `{"generation":"1"}`,
				filepath.Join(gen, "manifest.json"):         `{}`,
				filepath.Join(gen, "runtime-state.json"):    `{"session_metadata":{"active_model":"mistral-medium-3.5"}}`,
				filepath.Join(gen, "projection-state.json"): fmt.Sprintf(`{"snapshot":{"history":{"entries":[{"type":"message","role":"user","content":[{"type":"text","text":"Run the command"}],"createdAt":1790601800000},{"type":"effect","id":"call-1","createdAt":1790601802000,"updatedAt":1790601804000,"detail":{"kind":"tool","toolName":"file_system.bash","input":{"command":"echo output"}},"state":{"status":"%s","outputText":"output"}}]}}}`, tt.status),
			} {
				require.NoError(t, os.WriteFile(name, []byte(content), 0o600))
			}
			provider, ok := parser.NewProvider(parser.AgentVibe, parser.ProviderConfig{Roots: []string{root}})
			require.True(t, ok)
			sources, err := provider.Discover(t.Context())
			require.NoError(t, err)
			require.Len(t, sources, 1)
			outcome, err := provider.Parse(t.Context(), parser.ParseRequest{Source: sources[0]})
			require.NoError(t, err)
			require.Len(t, outcome.Results, 1)
			parsed := outcome.Results[0].Result
			require.Len(t, parsed.Messages, 2)
			candidate, err := ingest.PrepareCandidate(t.Context(), parsed, ingest.ContentOptions{})
			require.NoError(t, err)
			prepared, err := ingest.Finalize(t.Context(), candidate, ingest.ContentOptions{})
			require.NoError(t, err)
			withCarrier := parsed
			withCarrier.Messages = append(append([]parser.ParsedMessage(nil), parsed.Messages...), parser.ParsedMessage{
				Ordinal: 2, Role: parser.RoleUser, ContentLength: 6,
				ToolResults: []parser.ParsedToolResult{{ToolUseID: "call-1", ContentRaw: `"output"`, ContentLength: 6}},
			})
			carrierCandidate, err := ingest.PrepareCandidate(t.Context(), withCarrier, ingest.ContentOptions{})
			require.NoError(t, err)
			carrierPrepared, err := ingest.Finalize(t.Context(), carrierCandidate, ingest.ContentOptions{})
			require.NoError(t, err)
			assert.Equal(t, prepared.Messages, carrierPrepared.Messages)
			database := dbtest.OpenTestDB(t)
			written, err := database.WriteSessionBatch([]db.SessionBatchWrite{{Session: prepared.Session, Messages: prepared.Messages}})
			require.NoError(t, err)
			require.Empty(t, written.Errors)
			stored, err := database.GetMessages(t.Context(), prepared.Session.ID, 0, 100, true)
			require.NoError(t, err)
			require.Len(t, stored, 2)
			rows := ingest.ExtractToolCallRows(stored)
			require.Len(t, rows, 1)
			assert.Equal(t, "output", rows[0].ResultContent)
			assert.Equal(t, tt.wantStatus, rows[0].EventStatus)
			assert.Equal(t, tt.wantFailures, signals.ComputeToolHealth(rows).FailureSignalCount)
			require.Len(t, stored[1].ToolCalls, 1)
			events := stored[1].ToolCalls[0].ResultEvents
			require.Len(t, events, 2)
			assert.Equal(t, "tool_execution", events[0].Source)
			assert.Equal(t, "started", events[0].Status)
			assert.Equal(t, "tool_execution", events[1].Source)
			assert.Equal(t, tt.wantStatus, events[1].Status)
			assert.Equal(t, "output", events[1].Content)
			timing, err := database.GetSessionTiming(t.Context(), prepared.Session.ID)
			require.NoError(t, err)
			require.NotNil(t, timing)
			require.Len(t, timing.Turns, 1)
			require.Len(t, timing.Turns[0].Calls, 1)
			if tt.status != "cancelled" {
				require.NotNil(t, timing.Turns[0].Calls[0].DurationMs)
				assert.Equal(t, int64(2000), *timing.Turns[0].Calls[0].DurationMs)
			}
		})
	}
}
