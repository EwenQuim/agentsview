package parser

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// The unified harness session store (store format
// "mistral.vibe.unified-session-store/v1") replaced the legacy
// messages.jsonl + meta.json layout under <root>/unified/<session-id>/.
// A session directory holds:
//
//	CURRENT                      pointer to the newest generation
//	meta.json                    session metadata (absent for subagents)
//	journal/<seq>.jsonl          recovery journal (not parsed here)
//	chunks/<sha256>.json         content-addressed entry chunks
//	generations/<seq>/           snapshots, pruned to the newest two
//	  manifest.json              lists the projection and checkpoint chunks
//	  projection-state.json      session envelope with aggregate token usage
//	  runtime-state.json         runtime metadata (active model, cwd)
//
// The public transcript is the projection chunks listed by the newest
// generation's manifest, in listed order; each chunk is a JSON array of
// harness public-session-state entries (message, reasoning, effect, notice,
// checkpoint). Vibe folds the journal into a new generation when a turn finishes;
// a turn still in progress appears once it finishes.

type vibeUnifiedCurrent struct {
	Generation string `json:"generation"`
	SessionID  string `json:"session_id"`
}

type vibeUnifiedManifest struct {
	Generation      string `json:"generation"`
	SessionID       string `json:"session_id"`
	ProjectionState struct {
		Chunks []string `json:"chunks"`
	} `json:"projection_state"`
}

type vibeUnifiedTokenUsage struct {
	InputTokens       int `json:"inputTokens"`
	OutputTokens      int `json:"outputTokens"`
	CachedInputTokens int `json:"cachedInputTokens"`
	TotalTokens       int `json:"totalTokens"`
}

type vibeUnifiedProjectionSession struct {
	ID           string                `json:"id"`
	Title        string                `json:"title"`
	CreatedAt    int64                 `json:"createdAt"`
	UpdatedAt    int64                 `json:"updatedAt"`
	TokenUsage   vibeUnifiedTokenUsage `json:"tokenUsage"`
	ContextUsage vibeUnifiedTokenUsage `json:"contextUsage"`
}

type vibeUnifiedProjectionState struct {
	Snapshot struct {
		Session vibeUnifiedProjectionSession `json:"session"`
		History struct {
			Entries []vibeUnifiedEntry `json:"entries"`
		} `json:"history"`
	} `json:"snapshot"`
}

// vibeUnifiedRuntimeMetadata carries the identity-bearing fields of
// runtime-state.json. The file also embeds plugin and skill manifests, so
// only these fields are decoded.
type vibeUnifiedRuntimeMetadata struct {
	Identity struct {
		Kind            string `json:"kind"`
		ParentSessionID string `json:"parent_session_id"`
	} `json:"identity"`
	SessionMetadata struct {
		ActiveModel string `json:"active_model"`
		Cwd         string `json:"cwd"`
	} `json:"session_metadata"`
	ImportProvenance *struct {
		Source struct {
			SessionID string `json:"session_id"`
		} `json:"source"`
	} `json:"import_provenance"`
}

type vibeUnifiedEntry struct {
	Type      string                  `json:"type"`
	Role      string                  `json:"role"`
	ID        string                  `json:"id"`
	TurnID    string                  `json:"turnId"`
	Text      string                  `json:"text"`
	Content   []vibeUnifiedBlock      `json:"content,omitempty"`
	CreatedAt int64                   `json:"createdAt"`
	Detail    *vibeUnifiedEffect      `json:"detail,omitempty"`
	State     *vibeUnifiedEffectState `json:"state,omitempty"`
}

type vibeUnifiedBlock struct {
	Type string `json:"type,omitempty"`
	Text string `json:"text,omitempty"`
}

type vibeUnifiedEffect struct {
	Kind     string         `json:"kind,omitempty"`
	ToolName string         `json:"toolName,omitempty"`
	Input    jsontext.Value `json:"input,omitempty"`
}

type vibeUnifiedEffectState struct {
	Output     *vibeUnifiedEffectOutput `json:"output,omitempty"`
	OutputText string                   `json:"outputText,omitempty"`
}

type vibeUnifiedEffectOutput struct {
	Content []vibeUnifiedBlock `json:"content,omitempty"`
}

// vibeIsUnifiedAnchor reports whether path is the CURRENT anchor of a unified
// session directory (.../unified/<session-id>/CURRENT).
func vibeIsUnifiedAnchor(path string) bool {
	return filepath.Base(path) == "CURRENT" &&
		filepath.Base(filepath.Dir(filepath.Dir(path))) == "unified"
}

// vibeUnifiedSessionDirFromRel maps a root-relative path under
// unified/<session-id>/ back to its session directory.
func vibeUnifiedSessionDirFromRel(rel, root string) (string, bool) {
	parts := strings.Split(rel, string(filepath.Separator))
	if len(parts) < 2 || parts[0] != "unified" || parts[1] == "" {
		return "", false
	}
	return filepath.Join(filepath.Clean(root), "unified", parts[1]), true
}

// vibeUnifiedGenerationDir resolves the generation named by CURRENT.
func vibeUnifiedGenerationDir(sessionDir string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(sessionDir, "CURRENT"))
	if err != nil {
		return "", fmt.Errorf("reading Vibe unified CURRENT: %w", err)
	}
	var current vibeUnifiedCurrent
	if err := json.Unmarshal(raw, &current); err != nil {
		return "", fmt.Errorf("parsing Vibe unified CURRENT: %w", err)
	}
	if current.Generation == "" {
		return "", fmt.Errorf("missing generation in Vibe unified CURRENT")
	}
	return filepath.Join(sessionDir, "generations", current.Generation), nil
}

// parseVibeUnifiedResultFile parses a unified harness session directory,
// anchored on its CURRENT pointer, into a ParseResult.
func parseVibeUnifiedResultFile(anchorPath string, fileInfo FileInfo, root string) (ParseResult, error) {
	sessionDir := filepath.Dir(anchorPath)
	dirName := filepath.Base(sessionDir)

	result := ParseResult{
		Session: ParsedSession{
			Agent:           AgentVibe,
			File:            fileInfo,
			ID:              "vibe:" + dirName,
			SourceSessionID: dirName,
			Project:         "vibe",
			StartedAt:       time.Unix(0, fileInfo.Mtime),
			EndedAt:         time.Unix(0, fileInfo.Mtime),
		},
	}

	genDir, err := vibeUnifiedGenerationDir(sessionDir)
	if err != nil {
		return result, err
	}

	manifest, err := readVibeUnifiedManifest(genDir)
	if err != nil {
		return result, err
	}

	projection, err := readVibeUnifiedProjectionState(genDir)
	if err != nil {
		return result, err
	}
	projSession := projection.Snapshot.Session

	var cwd, gitBranch, gitCommit, title, parentID string

	meta, metaErr := parseVibeMetadata(filepath.Join(sessionDir, "meta.json"))
	switch {
	case metaErr != nil && errors.Is(metaErr, os.ErrNotExist):
		// Subagent sessions have no meta.json; recover identity from the
		// projection snapshot and runtime state.
	case metaErr != nil:
		return result, fmt.Errorf("parsing Vibe unified meta.json in %s: %w", sessionDir, metaErr)
	default:
		if meta.SessionID != "" {
			result.Session.ID = "vibe:" + meta.SessionID
			result.Session.SourceSessionID = meta.SessionID
		}
		if !meta.StartTime.IsZero() {
			result.Session.StartedAt = meta.StartTime
		}
		if !meta.EndTime.IsZero() {
			result.Session.EndedAt = meta.EndTime
		}
		cwd = meta.WorkingDir
		gitBranch = meta.GitBranch
		gitCommit = meta.GitCommit
		title = meta.Title
	}

	runtimeMeta, runtimeErr := readVibeUnifiedRuntimeMetadata(genDir)
	runtimeKind := ""
	if runtimeErr == nil {
		if cwd == "" {
			cwd = runtimeMeta.SessionMetadata.Cwd
		}
		runtimeKind = runtimeMeta.Identity.Kind
		if runtimeKind == "subagent" || runtimeKind == "fork" {
			parentID = runtimeMeta.Identity.ParentSessionID
		}
	}

	if runtimeMeta.ImportProvenance != nil {
		parentID = runtimeMeta.ImportProvenance.Source.SessionID
	}

	if title == "" {
		title = projSession.Title
	}
	if title != "" {
		result.Session.SessionName = title
		result.Session.SessionNamePresent = true
	}
	if projSession.CreatedAt > 0 && (metaErr != nil || meta.StartTime.IsZero()) {
		result.Session.StartedAt = time.UnixMilli(projSession.CreatedAt)
	}
	if projSession.UpdatedAt > 0 {
		result.Session.EndedAt = time.UnixMilli(projSession.UpdatedAt)
	}
	if cwd != "" {
		result.Session.Cwd = cwd
		if project := ExtractProjectFromCwdWithBranch(cwd, gitBranch); project != "" {
			result.Session.Project = project
		}
	}
	result.Session.GitBranch = gitBranch
	result.Session.SourceVersion = gitCommit
	if parentID != "" {
		result.Session.ParentSessionID = "vibe:" + parentID
		switch {
		case runtimeKind == "subagent":
			result.Session.RelationshipType = RelSubagent
		case runtimeKind == "fork":
			result.Session.RelationshipType = RelFork
		case runtimeMeta.ImportProvenance != nil:
			result.Session.RelationshipType = RelContinuation
		}
	}

	if projSession.TokenUsage.OutputTokens > 0 {
		result.Session.HasTotalOutputTokens = true
		result.Session.TotalOutputTokens = projSession.TokenUsage.OutputTokens
	}
	if projSession.ContextUsage.TotalTokens > 0 {
		result.Session.HasPeakContextTokens = true
		result.Session.PeakContextTokens = projSession.ContextUsage.TotalTokens
	}

	sessionModel := firstNonEmptyJSONLString(
		runtimeMeta.SessionMetadata.ActiveModel,
		vibeConfigDefaultModel(root),
		"mistral-medium-3.5",
	)
	trimImported := runtimeMeta.ImportProvenance != nil && parentID != "" && findVibeSourceFile(root, parentID) != ""
	messages, err := parseVibeUnifiedChunks(sessionDir, manifest.ProjectionState.Chunks, projection.Snapshot.History.Entries, sessionModel, trimImported)
	if err != nil {
		return result, err
	}
	result.Messages = messages
	result.Session.MessageCount = len(messages)
	for _, msg := range messages {
		if msg.Role == RoleUser && !msg.IsSystem && len(msg.ToolResults) == 0 &&
			result.Session.FirstMessage == "" && msg.Content != "" {
			result.Session.FirstMessage = msg.Content
		}
		if msg.Role == RoleUser && !msg.IsSystem && len(msg.ToolResults) == 0 {
			result.Session.UserMessageCount++
		}
	}

	stats := VibeStats{
		SessionPromptTokens:     projSession.TokenUsage.InputTokens,
		SessionCompletionTokens: projSession.TokenUsage.OutputTokens,
		SessionCachedTokens:     projSession.TokenUsage.CachedInputTokens,
		ContextTokens:           projSession.ContextUsage.TotalTokens,
	}
	result.UsageEvents = vibeUsageEvents(
		stats, sessionModel, result.Session.ID,
		result.Session.StartedAt, result.Session.EndedAt,
	)

	return result, nil
}

// parseVibeUnifiedChunks uses inline history unless the manifest names a chunk list.
func parseVibeUnifiedChunks(sessionDir string, chunkHashes []string, entries []vibeUnifiedEntry, model string, trimImported bool) ([]ParsedMessage, error) {
	if chunkHashes != nil {
		entries = nil
		for _, chunkHash := range chunkHashes {
			if chunkHash == "" {
				continue
			}
			raw, err := os.ReadFile(filepath.Join(sessionDir, "chunks", chunkHash+".json"))
			if err != nil {
				return nil, fmt.Errorf("reading Vibe unified chunk %s: %w", chunkHash, err)
			}
			var chunkEntries []vibeUnifiedEntry
			if err := json.Unmarshal(raw, &chunkEntries); err != nil {
				return nil, fmt.Errorf("parsing Vibe unified chunk %s: %w", chunkHash, err)
			}
			entries = append(entries, chunkEntries...)
		}
	}
	var messages []ParsedMessage
	thinkingByTurn := make(map[string]string)
	for _, entry := range entries {
		if trimImported && strings.HasPrefix(entry.ID, "imported-") {
			continue
		}
		switch entry.Type {
		case "message":
			msg := vibeUnifiedEntryMessage(entry, thinkingByTurn)
			msg.Ordinal = len(messages)
			if msg.Role == RoleAssistant {
				msg.Model = model
			}
			messages = append(messages, msg)
		case "reasoning":
			if entry.TurnID != "" {
				thinkingByTurn[entry.TurnID] += entry.Text
			}
		case "effect":
			call, carrier := vibeUnifiedEffectMessages(entry, len(messages))
			if call != nil {
				call.Model = model
				call.ThinkingText = thinkingByTurn[entry.TurnID]
				call.HasThinking = call.ThinkingText != ""
				delete(thinkingByTurn, entry.TurnID)
				messages = append(messages, *call)
			}
			if carrier != nil {
				carrier.Model = model
				messages = append(messages, *carrier)
			}
		}
	}
	return messages, nil
}

// vibeUnifiedEntryMessage converts a message entry to a parsed message. The
// harness records thinking as separate reasoning entries that precede their
// assistant message within the same turn, so buffered thinking is attached
// here and the buffer cleared.
func vibeUnifiedEntryMessage(
	entry vibeUnifiedEntry, thinkingByTurn map[string]string,
) ParsedMessage {
	var content strings.Builder
	for _, block := range entry.Content {
		if block.Type == "text" || block.Type == "" {
			content.WriteString(block.Text)
		}
	}
	text := content.String()
	thinking := thinkingByTurn[entry.TurnID]
	delete(thinkingByTurn, entry.TurnID)

	msg := ParsedMessage{
		Role:          RoleType(entry.Role),
		Content:       text,
		ContentLength: len(text),
		SourceUUID:    entry.ID,
	}
	if entry.CreatedAt > 0 {
		msg.Timestamp = time.UnixMilli(entry.CreatedAt)
	}
	if thinking != "" {
		msg.ThinkingText = thinking
		msg.HasThinking = true
	}
	if entry.Role != "user" && entry.Role != "assistant" {
		msg.IsSystem = true
	}
	return msg
}

// vibeUnifiedEffectMessages converts an effect entry (a completed tool or
// subagent invocation with its result) into the same two-message shape the
// legacy parser emits: an assistant message carrying the tool call, followed
// by an empty RoleUser carrier message whose tool result pairs back to the
// call by ID.
func vibeUnifiedEffectMessages(entry vibeUnifiedEntry, ordinal int) (*ParsedMessage, *ParsedMessage) {
	if entry.ID == "" || entry.Detail == nil {
		return nil, nil
	}
	toolName := entry.Detail.ToolName
	if toolName == "" {
		toolName = entry.Detail.Kind
	}
	if toolName == "" {
		return nil, nil
	}
	inputJSON := string(entry.Detail.Input)
	if inputJSON == "" {
		inputJSON = "{}"
	}

	call := &ParsedMessage{
		Ordinal:    ordinal,
		Role:       RoleAssistant,
		Timestamp:  time.UnixMilli(entry.CreatedAt),
		HasToolUse: true,
		ToolCalls: []ParsedToolCall{{
			ToolUseID: entry.ID,
			ToolName:  toolName,
			Category:  NormalizeToolCategory(toolName[strings.LastIndex(toolName, ".")+1:]),
			InputJSON: inputJSON,
		}},
	}

	resultText := ""
	if entry.State != nil {
		resultText = entry.State.OutputText
		if resultText == "" && entry.State.Output != nil {
			var blocks strings.Builder
			for _, block := range entry.State.Output.Content {
				blocks.WriteString(block.Text)
			}
			resultText = blocks.String()
		}
	}
	quoted, err := json.Marshal(resultText)
	if err != nil {
		quoted = []byte(`""`)
	}
	carrier := &ParsedMessage{
		Ordinal:       ordinal + 1,
		Timestamp:     time.UnixMilli(entry.CreatedAt),
		Role:          RoleUser,
		Content:       "",
		ContentLength: len(resultText),
		ToolResults: []ParsedToolResult{{
			ToolUseID:     entry.ID,
			ContentRaw:    string(quoted),
			ContentLength: len(resultText),
		}},
	}
	return call, carrier
}

func readVibeUnifiedManifest(genDir string) (vibeUnifiedManifest, error) {
	var manifest vibeUnifiedManifest
	raw, err := os.ReadFile(filepath.Join(genDir, "manifest.json"))
	if err != nil {
		return manifest, fmt.Errorf("reading Vibe unified manifest: %w", err)
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return manifest, fmt.Errorf("parsing Vibe unified manifest: %w", err)
	}
	return manifest, nil
}

func readVibeUnifiedProjectionState(genDir string) (vibeUnifiedProjectionState, error) {
	var state vibeUnifiedProjectionState
	raw, err := os.ReadFile(filepath.Join(genDir, "projection-state.json"))
	if err != nil {
		return state, fmt.Errorf("reading Vibe unified projection state: %w", err)
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		return state, fmt.Errorf("parsing Vibe unified projection state: %w", err)
	}
	return state, nil
}

func readVibeUnifiedRuntimeMetadata(genDir string) (vibeUnifiedRuntimeMetadata, error) {
	var meta vibeUnifiedRuntimeMetadata
	raw, err := os.ReadFile(filepath.Join(genDir, "runtime-state.json"))
	if err != nil {
		return meta, err
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return meta, err
	}
	return meta, nil
}

// Config is read only at parse time so edits do not reprice archived sessions.
func vibeConfigDefaultModel(sessionsRoot string) string {
	root := filepath.Clean(sessionsRoot)
	if filepath.Base(root) != "session" ||
		filepath.Base(filepath.Dir(root)) != "logs" {
		return ""
	}
	configPath := filepath.Join(filepath.Dir(filepath.Dir(root)), "config.toml")
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	var config struct {
		ActiveModel string `toml:"active_model"`
	}
	if _, err := toml.Decode(string(raw), &config); err != nil {
		return ""
	}
	return config.ActiveModel
}

// vibeUnifiedFingerprint hashes the generation files; the manifest names chunks by digest.
func vibeUnifiedFingerprint(sessionDir string) (SourceFingerprint, error) {
	genDir, err := vibeUnifiedGenerationDir(sessionDir)
	if err != nil {
		return SourceFingerprint{}, err
	}
	var size, mtime int64
	hash := sha256.New()
	for _, path := range []string{
		filepath.Join(sessionDir, "CURRENT"),
		filepath.Join(sessionDir, "meta.json"),
		filepath.Join(genDir, "manifest.json"),
		filepath.Join(genDir, "projection-state.json"),
		filepath.Join(genDir, "runtime-state.json"),
	} {
		info, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return SourceFingerprint{}, err
		}
		size += info.Size()
		mtime = max(mtime, info.ModTime().UnixNano())
		if err := addSiblingMetadataFingerprintPart(hash, filepath.Base(path), path, info); err != nil {
			return SourceFingerprint{}, err
		}
	}
	return SourceFingerprint{Size: size, MTimeNS: mtime, Hash: hex.EncodeToString(hash.Sum(nil))}, nil
}
