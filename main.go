// makefile-assistant: Standalone makefile parser + keyword-ranked command assistant.
//
// Usage:
//   go run main.go -export-yaml   # write makefile_training.yaml from the live Makefile
//   go run main.go -train         # train a small MoE model on makefile pairs
//   go run main.go                # interactive chat with top-3 ranked commands
//   go run main.go -fuzzy         # interactive fuzzy finder
package main

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"unsafe"
)

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────

var makefilePaths = []string{"Makefile", "makefile", "GNUmakefile", "../Makefile", "../../Makefile"}

// ─────────────────────────────────────────────────────────────────────────────
// DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

type MakeTarget struct {
	Name        string
	Description string
	Command     string
}

type Ranked struct {
	Name, Desc string
	Score      float64
}

type Vocab struct {
	Word2ID map[string]int
	ID2Word []string
}

type TrainPair struct{ Input, Output string }
type TrainSeq struct{ InputIDs, TargetIDs []int }

type Expert struct {
	W1, B1, W2, B2 []float32
}

type MoEModel struct {
	VocabSize, EmbedDim, HiddenDim, NumExperts int
	Vocab                                      *Vocab
	Embed, Gate, Wout                          []float32
	Experts                                    []Expert
}

type Choice struct {
	Command string
	Comment string
	Raw     string
}

// ─────────────────────────────────────────────────────────────────────────────
// VOCABULARY
// ─────────────────────────────────────────────────────────────────────────────

const (
	tokPAD = "<PAD>"
	tokBOS = "<BOS>"
	tokEOS = "<EOS>"
	tokUNK = "<UNK>"
)

func newVocab() *Vocab {
	v := &Vocab{Word2ID: make(map[string]int)}
	for _, t := range []string{tokPAD, tokBOS, tokEOS, tokUNK} {
		v.add(t)
	}
	return v
}
func (v *Vocab) add(w string) int {
	if id, ok := v.Word2ID[w]; ok {
		return id
	}
	id := len(v.ID2Word)
	v.Word2ID[w] = id
	v.ID2Word = append(v.ID2Word, w)
	return id
}
func (v *Vocab) encode(w string) int {
	if id, ok := v.Word2ID[w]; ok {
		return id
	}
	return v.Word2ID[tokUNK]
}
func (v *Vocab) size() int { return len(v.ID2Word) }

// ─────────────────────────────────────────────────────────────────────────────
// MAKEFILE PARSER
// ─────────────────────────────────────────────────────────────────────────────

func findMakefile() (string, error) {
	candidates := []string{
		"Makefile",
		"makefile",
		"GNUmakefile",
		"../Makefile",
		"../../Makefile",
	}
	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil && info.Mode().IsRegular() {
			return p, nil
		}
	}
	return "", fmt.Errorf("no Makefile found in current or parent directories")
}

func readMakefile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read makefile: %w", err)
	}
	return string(data), nil
}

func parseMakefile(content string) ([]MakeTarget, error) {
	var targets []*MakeTarget
	var pendingDesc string
	var current *MakeTarget
	inTarget := false

	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || (strings.HasPrefix(trimmed, "#") && !strings.HasPrefix(trimmed, "##")) {
			continue
		}

		if strings.HasPrefix(trimmed, "##") {
			desc := strings.TrimSpace(strings.TrimPrefix(trimmed, "##"))
			desc = strings.TrimSuffix(strings.TrimSpace(desc), ":")
			if idx := strings.Index(desc, ":"); idx > 0 {
				prefix := strings.TrimSpace(desc[:idx])
				if !strings.Contains(prefix, " ") {
					desc = strings.TrimSpace(desc[idx+1:])
				}
			}
			if desc != "" {
				pendingDesc = desc
			}
			continue
		}

		if strings.Contains(trimmed, ":") && !strings.HasPrefix(trimmed, "\t") {
			parts := strings.SplitN(trimmed, ":", 2)
			name := strings.TrimSpace(parts[0])
			if strings.Contains(name, " ") || strings.Contains(name, "=") || strings.HasPrefix(name, ".") {
				continue
			}

			current = &MakeTarget{Name: name}
			if pendingDesc != "" {
				current.Description = pendingDesc
				pendingDesc = ""
			} else if len(parts) > 1 {
				rest := strings.TrimSpace(parts[1])
				if rest != "" && !strings.HasPrefix(rest, "=") {
					current.Description = rest
				}
			}
			targets = append(targets, current)
			inTarget = true
			continue
		}

		if inTarget && current != nil {
			cmd := strings.TrimSpace(line)
			if cmd != "" && !strings.HasPrefix(cmd, "#") {
				if current.Command == "" {
					current.Command = cmd
				} else {
					current.Command += "\n" + cmd
				}
			}
		}
	}

	out := make([]MakeTarget, len(targets))
	for i, t := range targets {
		out[i] = *t
	}
	if len(out) == 0 {
		return out, fmt.Errorf("no targets parsed")
	}
	return out, nil
}

func parseMakefileFromFile(path string) ([]MakeTarget, error) {
	content, err := readMakefile(path)
	if err != nil {
		return nil, err
	}
	return parseMakefile(content)
}

func parseMakefileFromStdin() ([]MakeTarget, error) {
	var sb strings.Builder
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		sb.WriteString(scanner.Text())
		sb.WriteString("\n")
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read stdin: %w", err)
	}
	return parseMakefile(sb.String())
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAINING DATA GENERATION
// ─────────────────────────────────────────────────────────────────────────────

var responseTmpl = []string{
	"you can use make %s for this",
	"use make %s",
	"try make %s",
	"run make %s to do this",
	"make %s will help with that",
	"the command is make %s",
	"make %s is what you need",
}

func targetsToPairs(targets []MakeTarget) []TrainPair {
	var pairs []TrainPair
	for _, t := range targets {
		if t.Description == "" {
			continue
		}
		questions := []string{
			t.Description,
			"how do i " + strings.ToLower(t.Description),
			"what is " + t.Name,
			strings.Join(splitName(t.Name), " "),
			t.Name + " command",
		}
		for qi, q := range questions {
			ans := fmt.Sprintf(responseTmpl[qi%len(responseTmpl)], t.Name)
			pairs = append(pairs, TrainPair{q, ans})
		}
	}
	return pairs
}

func exportYAML(targets []MakeTarget, outPath string) error {
	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create yaml: %w", err)
	}
	defer f.Close()

	fmt.Fprintln(f, "conversations:")
	for _, t := range targets {
		desc := t.Description
		if desc == "" {
			desc = t.Name
		}
		questions := []string{
			desc,
			"how do i " + strings.ToLower(desc),
			"what is " + t.Name,
			strings.Join(splitName(t.Name), " "),
			t.Name + " command",
		}
		for j, q := range questions {
			ans := fmt.Sprintf(responseTmpl[j%len(responseTmpl)], t.Name)
			fmt.Fprintf(f, "  - conversation_id: makefile-%s-%d\n", t.Name, j)
			fmt.Fprintln(f, "    turns:")
			fmt.Fprintln(f, "      - turn_sequence: 1")
			fmt.Fprintln(f, "        role: user")
			fmt.Fprintf(f, "        content: %q\n", q)
			fmt.Fprintln(f, "      - turn_sequence: 2")
			fmt.Fprintln(f, "        role: assistant")
			fmt.Fprintf(f, "        content: %q\n", ans)
		}
	}
	return nil
}

func buildVocab(pairs []TrainPair) *Vocab {
	v := newVocab()
	for _, p := range pairs {
		for _, w := range tokenize(p.Input) {
			v.add(w)
		}
		for _, w := range tokenize(p.Output) {
			v.add(w)
		}
	}
	return v
}

func toSeqs(pairs []TrainPair, vocab *Vocab, maxLen int) []TrainSeq {
	bosID := vocab.Word2ID[tokBOS]
	eosID := vocab.Word2ID[tokEOS]
	var seqs []TrainSeq
	for _, p := range pairs {
		inIDs := make([]int, 0)
		for _, w := range tokenize(p.Input) {
			inIDs = append(inIDs, vocab.encode(w))
		}
		tgt := []int{bosID}
		for _, w := range tokenize(p.Output) {
			tgt = append(tgt, vocab.encode(w))
		}
		tgt = append(tgt, eosID)
		if len(inIDs) == 0 || len(tgt) < 2 {
			continue
		}
		if len(inIDs) > maxLen {
			inIDs = inIDs[:maxLen]
		}
		if len(tgt) > maxLen+2 {
			tgt = tgt[:maxLen+2]
		}
		seqs = append(seqs, TrainSeq{inIDs, tgt})
	}
	return seqs
}

// ─────────────────────────────────────────────────────────────────────────────
// MIXTURE-OF-EXPERTS MODEL
// ─────────────────────────────────────────────────────────────────────────────

func randF32(n int, scale float32, rng *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64()) * scale
	}
	return s
}

func newModel(vocab *Vocab, E, H, K int) *MoEModel {
	V := vocab.size()
	rng := rand.New(rand.NewSource(42))
	eS := float32(math.Sqrt(2.0 / float64(E)))
	h1S := float32(math.Sqrt(2.0 / float64(E)))
	h2S := float32(math.Sqrt(2.0 / float64(H)))
	experts := make([]Expert, K)
	for i := range experts {
		experts[i] = Expert{
			W1: randF32(H*E, h1S, rng), B1: make([]float32, H),
			W2: randF32(E*H, h2S, rng), B2: make([]float32, E),
		}
	}
	return &MoEModel{
		VocabSize: V, EmbedDim: E, HiddenDim: H, NumExperts: K, Vocab: vocab,
		Embed: randF32(V*E, eS, rng), Gate: randF32(K*E, 0.01, rng),
		Experts: experts, Wout: randF32(V*E, eS, rng),
	}
}

type fwdCache struct {
	x, gL, gP, moe, logits []float32
	h, eOut                [][]float32
}

func (m *MoEModel) forward(ctxIDs []int, prevID int) fwdCache {
	E, H, K := m.EmbedDim, m.HiddenDim, m.NumExperts
	x := make([]float32, E)
	n := float32(len(ctxIDs) + 1)
	for _, id := range ctxIDs {
		for j := range x {
			x[j] += m.Embed[id*E+j]
		}
	}
	for j := range x {
		x[j] += m.Embed[prevID*E+j]
		x[j] /= n
	}
	gL := make([]float32, K)
	for e := 0; e < K; e++ {
		for j := 0; j < E; j++ {
			gL[e] += m.Gate[e*E+j] * x[j]
		}
	}
	gP := softmax(gL)
	hh := make([][]float32, K)
	eOut := make([][]float32, K)
	for e := 0; e < K; e++ {
		h := make([]float32, H)
		for i := 0; i < H; i++ {
			v := m.Experts[e].B1[i]
			for j := 0; j < E; j++ {
				v += m.Experts[e].W1[i*E+j] * x[j]
			}
			if v > 0 {
				h[i] = v
			}
		}
		hh[e] = h
		out := make([]float32, E)
		for i := 0; i < E; i++ {
			v := m.Experts[e].B2[i]
			for j := 0; j < H; j++ {
				v += m.Experts[e].W2[i*H+j] * h[j]
			}
			out[i] = v
		}
		eOut[e] = out
	}
	moe := make([]float32, E)
	for e := 0; e < K; e++ {
		for j := 0; j < E; j++ {
			moe[j] += gP[e] * eOut[e][j]
		}
	}
	logits := make([]float32, m.VocabSize)
	for i := 0; i < m.VocabSize; i++ {
		for j := 0; j < E; j++ {
			logits[i] += m.Wout[i*E+j] * moe[j]
		}
	}
	return fwdCache{x: x, gL: gL, gP: gP, h: hh, eOut: eOut, moe: moe, logits: logits}
}

func (m *MoEModel) backward(c fwdCache, ctxIDs []int, prevID, targetID int, lr float32) float32 {
	E, H, K := m.EmbedDim, m.HiddenDim, m.NumExperts
	probs := softmax(c.logits)
	loss := -float32(math.Log(float64(probs[targetID]) + 1e-9))
	dL := make([]float32, m.VocabSize)
	copy(dL, probs)
	dL[targetID] -= 1.0
	dMoe := make([]float32, E)
	for i := 0; i < m.VocabSize; i++ {
		for j := 0; j < E; j++ {
			m.Wout[i*E+j] -= lr * dL[i] * c.moe[j]
			dMoe[j] += m.Wout[i*E+j] * dL[i]
		}
	}
	dGate := make([]float32, K)
	for e := 0; e < K; e++ {
		for j := 0; j < E; j++ {
			dGate[e] += c.eOut[e][j] * dMoe[j]
		}
	}
	dGL := smaxGrad(c.gP, dGate)
	dx := make([]float32, E)
	for e := 0; e < K; e++ {
		for j := 0; j < E; j++ {
			m.Gate[e*E+j] -= lr * dGL[e] * c.x[j]
			dx[j] += m.Gate[e*E+j] * dGL[e]
		}
	}
	for e := 0; e < K; e++ {
		g := c.gP[e]
		dOut := make([]float32, E)
		for j := 0; j < E; j++ {
			dOut[j] = g * dMoe[j]
		}
		dh := make([]float32, H)
		for i := 0; i < E; i++ {
			for j := 0; j < H; j++ {
				m.Experts[e].W2[i*H+j] -= lr * dOut[i] * c.h[e][j]
				dh[j] += m.Experts[e].W2[i*H+j] * dOut[i]
			}
			m.Experts[e].B2[i] -= lr * dOut[i]
		}
		for j := 0; j < H; j++ {
			if c.h[e][j] <= 0 {
				dh[j] = 0
			}
		}
		for i := 0; i < H; i++ {
			for j := 0; j < E; j++ {
				m.Experts[e].W1[i*E+j] -= lr * dh[i] * c.x[j]
				dx[j] += m.Experts[e].W1[i*E+j] * dh[i]
			}
			m.Experts[e].B1[i] -= lr * dh[i]
		}
	}
	invN := float32(1.0) / float32(len(ctxIDs)+1)
	for _, id := range ctxIDs {
		for j := 0; j < E; j++ {
			m.Embed[id*E+j] -= lr * dx[j] * invN
		}
	}
	for j := 0; j < E; j++ {
		m.Embed[prevID*E+j] -= lr * dx[j] * invN
	}
	return loss
}

func (m *MoEModel) generate(inputIDs []int, maxTokens int) string {
	bosID := m.Vocab.Word2ID[tokBOS]
	eosID := m.Vocab.Word2ID[tokEOS]
	var words []string
	prevID := bosID
	for i := 0; i < maxTokens; i++ {
		c := m.forward(inputIDs, prevID)

		if i > 0 {
			c.logits[prevID] -= 10.0
		}

		nextID := argmax(c.logits)
		if nextID == eosID {
			break
		}
		if nextID >= 0 && nextID < len(m.Vocab.ID2Word) {
			words = append(words, m.Vocab.ID2Word[nextID])
		}
		prevID = nextID
	}
	return strings.Join(words, " ")
}

// ─────────────────────────────────────────────────────────────────────────────
// MATH HELPERS
// ─────────────────────────────────────────────────────────────────────────────

func softmax(x []float32) []float32 {
	out := make([]float32, len(x))
	maxV := x[0]
	for _, v := range x {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v - maxV)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func smaxGrad(p, dOut []float32) []float32 {
	dot := float32(0)
	for i := range p {
		dot += dOut[i] * p[i]
	}
	dx := make([]float32, len(p))
	for i := range dx {
		dx[i] = p[i] * (dOut[i] - dot)
	}
	return dx
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// ─────────────────────────────────────────────────────────────────────────────
// MULTI-PHASE CURRICULUM TRAINING
// ─────────────────────────────────────────────────────────────────────────────

type Phase struct {
	Name   string
	Epochs int
	LR     float32
}

var phases = []Phase{
	{"Phase 1 — Social Bootcamp     ", 30, 0.01},
	{"Phase 2 — Coherence Polish I  ", 50, 0.005},
	{"Phase 3 — Coherence Polish II ", 50, 0.001},
	{"Phase 4 — Coherence Polish III", 50, 0.0005},
	{"Phase 5 — Coherence Polish IV ", 50, 0.0001},
}

func trainModel(model *MoEModel, seqs []TrainSeq) {
	const improvThreshold = float32(0.002)
	const earlyStopPatience = 20

	total := 0
	best := float32(1e9)
	stagnant := 0

	for _, ph := range phases {
		fmt.Printf("\n🚀 %s | epochs=%d lr=%g\n", ph.Name, ph.Epochs, ph.LR)
		phBest := float32(1e9)
		for ep := 0; ep < ph.Epochs; ep++ {
			cosine := float32(0.05 + 0.95*0.5*(1.0+math.Cos(math.Pi*float64(ep)/float64(ph.Epochs))))
			lr := ph.LR * cosine

			rand.Shuffle(len(seqs), func(i, j int) { seqs[i], seqs[j] = seqs[j], seqs[i] })

			var totalLoss float32
			steps := 0
			for _, seq := range seqs {
				for t := 0; t < len(seq.TargetIDs)-1; t++ {
					c := model.forward(seq.InputIDs, seq.TargetIDs[t])
					loss := model.backward(c, seq.InputIDs, seq.TargetIDs[t], seq.TargetIDs[t+1], lr)
					totalLoss += loss
					steps++
				}
			}
			total++
			avg := float32(0)
			if steps > 0 {
				avg = totalLoss / float32(steps)
			}
			if avg < phBest-improvThreshold {
				phBest = avg
				stagnant = 0
			} else {
				stagnant++
			}
			if avg < best {
				best = avg
			}
			if (ep+1)%10 == 0 {
				fmt.Printf("  epoch %3d | loss=%.4f | stagnant=%d\n", total, avg, stagnant)
			}
			if stagnant >= earlyStopPatience {
				fmt.Printf("  ⚡ stagnant %d epochs — advancing phase early\n", earlyStopPatience)
				stagnant = 0
				break
			}
		}
	}
	fmt.Printf("\n✅ Training done. Best loss: %.4f\n", best)
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAINING
// ─────────────────────────────────────────────────────────────────────────────

func runTraining(targets []MakeTarget, modelPath string) {
	if len(targets) == 0 {
		fmt.Println("No targets available for training.")
		return
	}

	pairs := targetsToPairs(targets)
	fmt.Printf("📝 Generated %d training pairs\n", len(pairs))

	vocab := buildVocab(pairs)
	seqs := toSeqs(pairs, vocab, 32)
	fmt.Printf("🔤 Vocabulary: %d tokens  |  Sequences: %d\n", vocab.size(), len(seqs))

	const embedDim, hiddenDim, numExperts = 48, 96, 4
	model := newModel(vocab, embedDim, hiddenDim, numExperts)
	fmt.Printf("🧠 MoE model: %d experts | embed=%d | hidden=%d\n", numExperts, embedDim, hiddenDim)

	trainModel(model, seqs)

	if err := saveModel(model, modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving model: %v\n", err)
		return
	}
	fmt.Printf("💾 Model saved to %s\n", modelPath)
}

func saveModel(model *MoEModel, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(model)
}

func loadModel(path string) (*MoEModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var model MoEModel
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&model); err != nil {
		return nil, err
	}
	return &model, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// KEYWORD RANKING
// ─────────────────────────────────────────────────────────────────────────────

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "it": true,
	"do": true, "to": true, "for": true, "and": true, "or": true,
	"in": true, "at": true, "by": true, "on": true, "of": true,
	"with": true, "this": true, "that": true, "from": true,
	"run": true, "start": true, "use": true, "make": true,
	"all": true, "get": true, "set": true,
}

func tokenize(text string) []string {
	text = strings.ToLower(strings.TrimSpace(text))
	fields := strings.Fields(text)
	out := make([]string, 0, len(fields))
	for _, w := range fields {
		w = strings.Trim(w, `.,!?;:"'`+"`()[]{}$@\\/")
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

func splitName(name string) []string {
	parts := strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' })
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, strings.ToLower(p))
		}
	}
	return out
}

func rankTargets(targets []MakeTarget, query string, k int) []Ranked {
	raw := tokenize(query)
	qTokens := make([]string, 0, len(raw))
	for _, t := range raw {
		if !stopWords[t] {
			qTokens = append(qTokens, t)
		}
	}
	for _, w := range strings.Fields(strings.ToLower(query)) {
		for _, part := range splitName(w) {
			if !stopWords[part] {
				qTokens = append(qTokens, part)
			}
		}
	}
	if len(qTokens) == 0 {
		qTokens = raw
	}
	qSet := make(map[string]bool, len(qTokens))
	for _, t := range qTokens {
		qSet[t] = true
	}

	type pair struct {
		t     MakeTarget
		score float64
	}
	res := make([]pair, 0, len(targets))
	for _, t := range targets {
		nw := splitName(t.Name)
		nSet := make(map[string]bool)
		for _, w := range nw {
			nSet[w] = true
		}
		dSet := make(map[string]bool)
		for _, w := range tokenize(t.Description + " " + t.Command) {
			if !stopWords[w] {
				dSet[w] = true
			}
		}
		var weighted float64
		for _, qt := range qTokens {
			if nSet[qt] {
				weighted += 3.0
			} else if dSet[qt] {
				weighted += 1.0
			}
		}
		hits := 0
		for _, w := range nw {
			if qSet[w] {
				hits++
			}
		}
		cov := 0.0
		if len(nw) > 0 {
			cov = float64(hits) / float64(len(nw))
		}
		score := (weighted/float64(len(qTokens)+1))*0.7 + cov*0.3
		res = append(res, pair{t, score})
	}

	sort.Slice(res, func(i, j int) bool { return res[i].score > res[j].score })
	if len(res) > k {
		res = res[:k]
	}

	best := 0.0
	for _, r := range res {
		if r.score > best {
			best = r.score
		}
	}

	out := make([]Ranked, len(res))
	for i, r := range res {
		pct := 0.0
		if best > 0 {
			pct = r.score / best
		}
		out[i] = Ranked{r.t.Name, r.t.Description, pct}
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// CHAT
// ─────────────────────────────────────────────────────────────────────────────

func runChat(targets []MakeTarget) {
	if len(targets) == 0 {
		fmt.Println("No makefile targets found.")
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("\nMakefile Chat — type 'quit' or 'exit' to stop.")
	fmt.Println("Ask about any make target.\n")

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		if text == "quit" || text == "exit" {
			break
		}

		ranked := rankTargets(targets, text, 3)
		if len(ranked) == 0 || ranked[0].Score == 0 {
			fmt.Println("Bot: I could not find a matching make target. Try rephrasing.")
			continue
		}

		fmt.Println("Bot: Here are the best matches:")
		for _, r := range ranked {
			fmt.Printf("  - make %s (%d%%)\n", r.Name, int(r.Score*100))
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// FUZZY FINDER
// ─────────────────────────────────────────────────────────────────────────────

const (
	tiocgeta         = 0x40487413
	tiocseta         = 0x80487414
	tcgets           = 0x5401
	tcsets           = 0x5402
	tiocgwinsz       = 0x5413
	tiocgwinszDarwin = 0x40087468
)

type winsize struct {
	Row, Col, Xpixel, Ypixel uint16
}

func getTerminalSize(fd uintptr) (width, height int) {
	var ws winsize
	var req uintptr
	if runtime.GOOS == "darwin" {
		req = tiocgwinszDarwin
	} else {
		req = tiocgwinsz
	}

	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, req, uintptr(unsafe.Pointer(&ws)))
	if err != 0 || ws.Row == 0 || ws.Col == 0 {
		return 80, 25
	}
	return int(ws.Col), int(ws.Row)
}

func enableRawMode(fd uintptr) (*syscall.Termios, error) {
	var oldState syscall.Termios
	var reqGet uintptr

	if runtime.GOOS == "darwin" {
		reqGet = tiocgeta
	} else {
		reqGet = tcgets
	}

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, reqGet, uintptr(unsafe.Pointer(&oldState))); err != 0 {
		return nil, err
	}

	newState := oldState
	newState.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP | syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	newState.Oflag &^= syscall.OPOST
	newState.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	newState.Cflag &^= syscall.CSIZE | syscall.PARENB
	newState.Cflag |= syscall.CS8

	var reqSet uintptr
	if runtime.GOOS == "darwin" {
		reqSet = tiocseta
	} else {
		reqSet = tcsets
	}

	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, reqSet, uintptr(unsafe.Pointer(&newState)))
	if err != 0 {
		return nil, err
	}
	return &oldState, nil
}

func disableRawMode(fd uintptr, oldState *syscall.Termios) {
	if oldState != nil {
		var reqSet uintptr
		if runtime.GOOS == "darwin" {
			reqSet = tiocseta
		} else {
			reqSet = tcsets
		}
		_, _, _ = syscall.Syscall(syscall.SYS_IOCTL, fd, reqSet, uintptr(unsafe.Pointer(oldState)))
	}
}

func fuzzyMatch(target, query string) (bool, int) {
	if query == "" {
		return true, 0
	}
	targetLower := strings.ToLower(target)
	queryLower := strings.ToLower(query)

	tIdx, score := 0, 0
	for qIdx := 0; qIdx < len(queryLower); qIdx++ {
		found := false
		for ; tIdx < len(targetLower); tIdx++ {
			if targetLower[tIdx] == queryLower[qIdx] {
				score += 10 - tIdx
				tIdx++
				found = true
				break
			}
		}
		if !found {
			return false, 0
		}
	}
	return true, score
}

func parseChoice(line string) Choice {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return Choice{Raw: line}
	}
	cmd := parts[0]
	comment := ""
	if len(parts) > 1 {
		comment = strings.Join(parts[1:], " ")
	}
	return Choice{
		Command: cmd,
		Comment: comment,
		Raw:     line,
	}
}

func runFuzzyFinder(targets []MakeTarget) {
	var choices []Choice
	for _, t := range targets {
		choices = append(choices, Choice{
			Command: t.Name,
			Comment: t.Description,
			Raw:     t.Name + " " + t.Description,
		})
	}

	if len(choices) == 0 {
		fmt.Fprintln(os.Stderr, "No targets available.")
		os.Exit(1)
	}

	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open TTY: %v\n", err)
		os.Exit(1)
	}
	defer tty.Close()

	fd := tty.Fd()
	oldState, err := enableRawMode(fd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to set raw mode: %v\n", err)
		os.Exit(1)
	}
	defer disableRawMode(fd, oldState)

	tty.WriteString("\x1b[?1049h\x1b[?25l\x1b[?1000h")
	defer tty.WriteString("\x1b[?1049l\x1b[?25h\x1b[?1000l")

	query := ""
	selectedIndex := 0

	filtered := make([]Choice, len(choices))
	copy(filtered, choices)

	filterChoices := func() {
		if query == "" {
			filtered = make([]Choice, len(choices))
			copy(filtered, choices)
			selectedIndex = 0
			return
		}

		type match struct {
			choice Choice
			score  int
		}
		var matches []match

		for _, choice := range choices {
			if ok, score := fuzzyMatch(choice.Raw, query); ok {
				matches = append(matches, match{choice: choice, score: score})
			}
		}

		filtered = nil
		for _, m := range matches {
			filtered = append(filtered, m.choice)
		}

		selectedIndex = 0
	}

	draw := func() {
		width, _ := getTerminalSize(fd)

		colWidth := 24
		cols := width / colWidth
		if cols <= 0 {
			cols = 1
		}

		var sb strings.Builder
		sb.WriteString("\x1b[H\x1b[J")

		sb.WriteString(fmt.Sprintf("\x1b[1;36m[ > %s ]\x1b[0m\r\n\r\n", query))

		if len(filtered) > 0 {
			rows := (len(filtered) + cols - 1) / cols

			for row := 0; row < rows; row++ {
				for col := 0; col < cols; col++ {
					i := col*rows + row
					if i >= len(filtered) {
						break
					}

					item := filtered[i]
					label := item.Command
					if len(label) > 16 {
						label = label[:16]
					}

					if i == selectedIndex {
						sb.WriteString(fmt.Sprintf("[ \x1b[37;45;1m%-16s\x1b[0m ] ", label))
					} else {
						sb.WriteString(fmt.Sprintf("[ \x1b[36m%-16s\x1b[0m ] ", label))
					}
				}
				sb.WriteString("\r\n")
			}

			if selectedIndex < len(filtered) && filtered[selectedIndex].Comment != "" {
				sb.WriteString(fmt.Sprintf("\r\n\x1b[90m> %s\x1b[0m", filtered[selectedIndex].Comment))
			}
		}

		tty.WriteString(sb.String())
	}

	buf := make([]byte, 6)
	var selectedResult Choice

	for {
		draw()
		n, err := tty.Read(buf)
		if err != nil || n == 0 {
			break
		}

		width, _ := getTerminalSize(fd)
		cols := width / 24
		if cols <= 0 {
			cols = 1
		}

		total := len(filtered)
		rows := 1
		if total > 0 {
			rows = (total + cols - 1) / cols
		}

		moveUp := func() {
			if selectedIndex%rows > 0 {
				selectedIndex--
			}
		}

		moveDown := func() {
			if selectedIndex%rows < rows-1 && selectedIndex+1 < total {
				selectedIndex++
			}
		}

		moveRight := func() {
			if selectedIndex+rows < total {
				selectedIndex += rows
			}
		}

		moveLeft := func() {
			if selectedIndex-rows >= 0 {
				selectedIndex -= rows
			}
		}

		switch {
		case buf[0] == 3: // Ctrl+C
			return

		case buf[0] == 13: // Enter
			if len(filtered) > 0 && selectedIndex < len(filtered) {
				selectedResult = filtered[selectedIndex]
			}
			disableRawMode(fd, oldState)
			tty.WriteString("\x1b[?1049l\x1b[?25h\x1b[?1000l")
			if selectedResult.Command != "" {
				fmt.Println(selectedResult.Command)
			}
			return

		case buf[0] == 127 || buf[0] == 8: // Backspace
			if len(query) > 0 {
				query = query[:len(query)-1]
				filterChoices()
			}

		case buf[0] == 14 || buf[0] == 10: // Ctrl+N / Ctrl+J
			moveDown()

		case buf[0] == 16 || buf[0] == 11: // Ctrl+P / Ctrl+K
			moveUp()

		case buf[0] == 27:
			if n >= 3 && buf[1] == '[' {
				switch buf[2] {
				case 'A': // Up Arrow
					moveUp()
				case 'B': // Down Arrow
					moveDown()
				case 'C': // Right Arrow
					moveRight()
				case 'D': // Left Arrow
					moveLeft()
				}
			}

		default:
			if buf[0] >= 32 && buf[0] <= 126 {
				query += string(buf[0])
				filterChoices()
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	exportFlag := flag.Bool("export-yaml", false, "Export makefile training data to YAML")
	trainFlag := flag.Bool("train", false, "Train the MoE model on makefile data")
	fuzzyFlag := flag.Bool("fuzzy", false, "Run fuzzy finder mode")
	modelPath := flag.String("model", "makefile_model.gob", "Path to save/load the trained model")
	yamlPath := flag.String("yaml", "makefile_training.yaml", "Output YAML path for training data")
	flag.Parse()

	var targets []MakeTarget
	var err error

	if *trainFlag {
		makefilePath, err := findMakefile()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		targets, err = parseMakefileFromFile(makefilePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing makefile: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Loaded %d targets from %s\n", len(targets), makefilePath)
		runTraining(targets, *modelPath)
		return
	}

	if *fuzzyFlag {
		makefilePath, err := findMakefile()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		targets, err = parseMakefileFromFile(makefilePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing makefile: %v\n", err)
			os.Exit(1)
		}
		runFuzzyFinder(targets)
		return
	}

	if *exportFlag {
		targets, err = parseMakefileFromStdin()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading Makefile from stdin: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Loaded %d targets from stdin\n", len(targets))
	} else {
		makefilePath, err := findMakefile()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		targets, err = parseMakefileFromFile(makefilePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing makefile: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Loaded %d targets from %s\n", len(targets), makefilePath)
	}

	if *exportFlag {
		if err := exportYAML(targets, *yamlPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error exporting YAML: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Exported training data to %s\n", *yamlPath)
		return
	}

	runChat(targets)
}
