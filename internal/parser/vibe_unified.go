package parser

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
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
// checkpoint).

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
	ID              string                `json:"id"`
	ParentSessionID string                `json:"parentSessionId"`
	Title           string                `json:"title"`
	CreatedAt       int64                 `json:"createdAt"`
	UpdatedAt       int64                 `json:"updatedAt"`
	TokenUsage      vibeUnifiedTokenUsage `json:"tokenUsage"`
	ContextUsage    vibeUnifiedTokenUsage `json:"contextUsage"`
}

type vibeUnifiedProjectionState struct {
	Snapshot struct {
		Session vibeUnifiedProjectionSession `json:"session"`
	} `json:"snapshot"`
}

// vibeUnifiedMeta is the unified meta.json. It is shaped differently from the
// legacy VibeSessionMetadata (no stats block, no config.active_model, and
// nullable identity fields), so it gets its own tolerant struct.
type vibeUnifiedMeta struct {
	SessionID       string    `json:"session_id"`
	ParentSessionID *string   `json:"parent_session_id"`
	StartTime       time.Time `json:"start_time"`
	EndTime         time.Time `json:"end_time"`
	GitCommit       *string   `json:"git_commit"`
	GitBranch       *string   `json:"git_branch"`
	Title           *string   `json:"title"`
	Environment     struct {
		WorkingDirectory string `json:"working_directory,omitempty"`
	} `json:"environment,omitempty"`
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

// vibeUnifiedGenerationDir resolves the newest generation directory of a
// unified session: the CURRENT pointer when readable, otherwise the
// lexicographically newest directory under generations/.
func vibeUnifiedGenerationDir(sessionDir string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(sessionDir, "CURRENT"))
	if err == nil {
		var current vibeUnifiedCurrent
		if err := json.Unmarshal(raw, &current); err == nil && current.Generation != "" {
			genDir := filepath.Join(sessionDir, "generations", current.Generation)
			if info, err := os.Stat(genDir); err == nil && info.IsDir() {
				return genDir, nil
			}
		}
	}
	entries, err := os.ReadDir(filepath.Join(sessionDir, "generations"))
	if err != nil {
		return "", fmt.Errorf("reading Vibe unified generations in %s: %w", sessionDir, err)
	}
	newest := ""
	for _, entry := range entries {
		if entry.IsDir() && entry.Name() > newest {
			newest = entry.Name()
		}
	}
	if newest == "" {
		return "", fmt.Errorf("no generations in Vibe unified session %s", sessionDir)
	}
	return filepath.Join(sessionDir, "generations", newest), nil
}

// parseVibeUnifiedSession parses a unified harness session at anchorPath and
// returns the session, messages, and usage events in the shape the provider
// consumes.
func parseVibeUnifiedSession(
	anchorPath, root, machine string,
) (*ParsedSession, []ParsedMessage, []ParsedUsageEvent, error) {
	info, err := os.Stat(anchorPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("stat %s: %w", anchorPath, err)
	}

	fileInfo := FileInfo{
		Path:  anchorPath,
		Size:  info.Size(),
		Mtime: info.ModTime().UnixNano(),
	}

	result, err := parseVibeUnifiedResultFile(anchorPath, fileInfo, root)
	if err != nil {
		return nil, nil, nil, err
	}
	if machine != "" {
		result.Session.Machine = machine
	}
	return &result.Session, result.Messages, result.UsageEvents, nil
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

	meta, metaErr := readVibeUnifiedMeta(sessionDir)
	switch {
	case metaErr != nil && errors.Is(metaErr, os.ErrNotExist):
		// Subagent sessions have no meta.json; recover identity from the
		// projection snapshot and runtime state.
		parentID = projSession.ParentSessionID
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
		cwd = meta.Environment.WorkingDirectory
		if meta.GitBranch != nil {
			gitBranch = *meta.GitBranch
		}
		if meta.GitCommit != nil {
			gitCommit = *meta.GitCommit
		}
		if meta.Title != nil {
			title = *meta.Title
		}
		if meta.ParentSessionID != nil && *meta.ParentSessionID != "" {
			parentID = *meta.ParentSessionID
		}
	}

	runtimeMeta, runtimeErr := readVibeUnifiedRuntimeMetadata(genDir)
	runtimeKind := ""
	if runtimeErr == nil {
		if cwd == "" {
			cwd = runtimeMeta.SessionMetadata.Cwd
		}
		runtimeKind = runtimeMeta.Identity.Kind
		if parentID == "" && runtimeKind == "subagent" {
			parentID = runtimeMeta.Identity.ParentSessionID
		}
	}

	if title == "" {
		title = projSession.Title
	}
	if title != "" {
		result.Session.SessionName = title
		result.Session.SessionNamePresent = true
	}
	if projSession.CreatedAt > 0 && result.Session.StartedAt.IsZero() {
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
		if runtimeKind == "subagent" || strings.HasPrefix(dirName, "child-") {
			result.Session.RelationshipType = RelSubagent
		} else {
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

	messages, malformed, err := parseVibeUnifiedChunks(sessionDir, manifest.ProjectionState.Chunks)
	if err != nil {
		return result, err
	}
	result.Messages = messages
	result.Session.MalformedLines = malformed
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

	sessionModel := firstNonEmptyJSONLString(
		runtimeMeta.SessionMetadata.ActiveModel,
		vibeConfigDefaultModel(root),
	)
	if sessionModel != "" {
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
	}

	return result, nil
}

// parseVibeUnifiedChunks reads the projection chunks of the newest generation
// in manifest order and maps the public session state entries to parsed
// messages.
func parseVibeUnifiedChunks(
	sessionDir string, chunkHashes []string,
) ([]ParsedMessage, int, error) {
	var messages []ParsedMessage
	ordinal := 0
	malformed := 0
	thinkingByTurn := make(map[string]string)

	for _, chunkHash := range chunkHashes {
		if chunkHash == "" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(sessionDir, "chunks", chunkHash+".json"))
		if err != nil {
			return messages, malformed, fmt.Errorf(
				"reading Vibe unified chunk %s: %w", chunkHash, err,
			)
		}
		var entries []vibeUnifiedEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			return messages, malformed, fmt.Errorf(
				"parsing Vibe unified chunk %s: %w", chunkHash, err,
			)
		}
		for _, entry := range entries {
			switch entry.Type {
			case "message":
				msg := vibeUnifiedEntryMessage(entry, thinkingByTurn)
				if msg == nil {
					continue
				}
				msg.Ordinal = ordinal
				ordinal++
				messages = append(messages, *msg)
			case "reasoning":
				if entry.TurnID != "" {
					thinkingByTurn[entry.TurnID] += entry.Text
				}
			case "effect":
				call, carrier := vibeUnifiedEffectMessages(entry, ordinal)
				if call != nil {
					messages = append(messages, *call)
					ordinal++
				}
				if carrier != nil {
					messages = append(messages, *carrier)
					ordinal++
				}
			default:
				// notice and checkpoint entries carry no transcript content.
			}
		}
	}
	return messages, malformed, nil
}

// vibeUnifiedEntryMessage converts a message entry to a parsed message. The
// harness records thinking as separate reasoning entries that precede their
// assistant message within the same turn, so buffered thinking is attached
// here and the buffer cleared.
func vibeUnifiedEntryMessage(
	entry vibeUnifiedEntry, thinkingByTurn map[string]string,
) *ParsedMessage {
	var content strings.Builder
	for _, block := range entry.Content {
		if block.Type == "text" || block.Type == "" {
			content.WriteString(block.Text)
		}
	}
	text := content.String()
	thinking := thinkingByTurn[entry.TurnID]
	delete(thinkingByTurn, entry.TurnID)

	msg := &ParsedMessage{
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
		Model:      "",
		HasToolUse: true,
		ToolCalls: []ParsedToolCall{{
			ToolUseID: entry.ID,
			ToolName:  toolName,
			Category:  NormalizeToolCategory(toolName),
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

func readVibeUnifiedMeta(sessionDir string) (vibeUnifiedMeta, error) {
	var meta vibeUnifiedMeta
	raw, err := os.ReadFile(filepath.Join(sessionDir, "meta.json"))
	if err != nil {
		return meta, err
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return meta, err
	}
	return meta, nil
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

// vibeConfigDefaultModel reads the CLI-level default model from the Vibe
// config. The unified session store records the per-session model in
// runtime-state only when the user overrode it for that session, so sessions
// that ran on the config default have no model on disk; mirroring the CLI's
// own resolution keeps their usage attributable. The sessions root follows the
// <vibe-home>/logs/session layout, so any other root shape yields no default.
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

// vibeUnifiedFingerprint builds a composite fingerprint over the session
// directory: CURRENT, meta.json, the newest generation's manifest and
// projection state, and every chunk file. Chunks are content-addressed and
// listed by hash in the manifest, so hashing CURRENT, the manifest, and the
// projection state (the small, always-rewritten files) covers transcript and
// usage changes without re-reading the whole chunk store.
func vibeUnifiedFingerprint(sessionDir string) (SourceFingerprint, error) {
	var size int64
	var mtime int64
	fold := func(path string) error {
		info, err := os.Stat(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		size += info.Size()
		if mod := info.ModTime().UnixNano(); mod > mtime {
			mtime = mod
		}
		return nil
	}

	if err := fold(filepath.Join(sessionDir, "CURRENT")); err != nil {
		return SourceFingerprint{}, err
	}
	if err := fold(filepath.Join(sessionDir, "meta.json")); err != nil {
		return SourceFingerprint{}, err
	}
	genDir, err := vibeUnifiedGenerationDir(sessionDir)
	if err != nil {
		return SourceFingerprint{}, err
	}
	for _, name := range []string{"manifest.json", "projection-state.json", "runtime-state.json"} {
		if err := fold(filepath.Join(genDir, name)); err != nil {
			return SourceFingerprint{}, err
		}
	}
	chunkEntries, err := os.ReadDir(filepath.Join(sessionDir, "chunks"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return SourceFingerprint{}, err
	}
	for _, entry := range chunkEntries {
		if err := fold(filepath.Join(sessionDir, "chunks", entry.Name())); err != nil {
			return SourceFingerprint{}, err
		}
	}

	hash := sha256.New()
	for _, name := range []string{
		filepath.Join(sessionDir, "CURRENT"),
		filepath.Join(genDir, "manifest.json"),
		filepath.Join(genDir, "projection-state.json"),
	} {
		if err := hashFileInto(hash, name); err != nil {
			return SourceFingerprint{}, err
		}
	}
	return SourceFingerprint{
		Size:    size,
		MTimeNS: mtime,
		Hash:    hex.EncodeToString(hash.Sum(nil)),
	}, nil
}

func hashFileInto(hash io.Writer, path string) error {
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer file.Close()
	_, err = io.Copy(hash, file)
	return err
}
