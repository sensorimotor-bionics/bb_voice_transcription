"""
Decide which speakers in a diarized transcript are interviewers and which are
interviewees, by reading the transcript text with a local LLM (Gemma via Ollama).

Diarization only tells us that speaker_0 and speaker_1 are different people; the roles
live in what they say ("can you describe what you feel?" versus "it feels sharper now").
So the whole transcript is handed to a local model, per-speaker votes are pooled across
chunks, and the requested number of interviewees is then enforced on the pooled scores.
"""

import json
import os
import re
from pathlib import Path
from typing import Union

import requests

from post_processing import (_speaker_sort_key, concatenate_batch_transcripts,
                             relabel_batched_transcript)

# Gemma 3 is the default: it follows the JSON schema reliably and 12b fits comfortably
# on a single consumer GPU. Anything served by Ollama works, e.g. "gemma3:4b" when the
# transcript is long and speed matters, or the qwen3 models if they are what is pulled.
DEFAULT_MODEL = "gemma3:12b"

# Ollama's own OLLAMA_HOST is honoured so a remote/alternate port needs no code change.
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Characters of transcript per LLM call. Roughly 5k tokens, which leaves room for the
# instructions and the answer inside an 8k context and keeps every chunk well away from
# the tail of the context window, where recall of early speakers degrades.
DEFAULT_CHUNK_CHARS = 20000

# Labels assigned to the two roles when only one speaker holds the role. With several,
# they become Interviewer_1, Interviewer_2, ... ordered by how much each one speaks.
INTERVIEWER_LABEL = "Interviewer"
INTERVIEWEE_LABEL = "Participant"

# assign_speaker's label for words that matched no diarization turn (mirrors
# transcription.UNKNOWN_SPEAKER, redefined here so this module does not import nemo).
# It is a residue of unmatched words rather than a person, so it gets no role.
UNKNOWN_SPEAKER = "unknown"

SYSTEM_PROMPT = """\
You label the speakers in a transcript of a research interview or experimental session.

Roles:
- "interviewer": staff running the session. Asks the questions, gives instructions,
  explains the task or equipment, prompts and reassures, talks to other staff.
- "interviewee": the participant or subject being studied. Answers the questions,
  describes their own experience, sensations, opinions or history.

Rules:
- Judge each speaker by what they do in the conversation, not by how much they talk.
  An interviewee can dominate the transcript, and an interviewer can be nearly silent.
- Asking questions is not by itself the mark of an interviewer. Participants ask plenty
  of them ("what does this button do?", "am I doing this right?"). What matters is who
  is being studied: the interviewee's own experience is the subject of the session,
  while interviewers ask about that experience and run the task around it.
- Interviewers talk to each other about running the session; interviewees do not.
- Speech-to-text errors and missing words are expected; ignore them.
- The speaker labels come from imperfect automatic diarization, so one label can carry
  speech from two different people. When a speaker's lines genuinely mix both roles,
  say so with a middle probability rather than picking a side.
- Return exactly one object per speaker listed in the request, in that order, and
  nothing else. Never emit one object per turn.

Answer in this order, because each field is meant to inform the next:
- session: one or two sentences on what kind of session this is and who appears to be
  running it, before you judge anybody.
- then, for each speaker:
  - evidence: the most role-revealing thing that speaker actually says here, quoted or
    closely paraphrased. Pick it before you settle on a number.
  - interviewee_probability: 0.0 means certainly an interviewer, 1.0 means certainly an
    interviewee, 0.5 means genuinely undecidable. It must agree with the evidence you
    just quoted. Use the whole range and separate the speakers from each other rather
    than giving several of them the same number.
"""


def _response_schema(speakers: list[str]) -> dict:
    """
    JSON schema constraining the model to one entry per known speaker. The item count is
    pinned as well: left unbounded, models happily emit one entry per conversational turn
    instead of one per speaker.

    Field order matters, because constrained decoding generates the keys in the order
    given and each one conditions the next. 'session' and 'evidence' therefore come
    before the number they are meant to justify: asked for the probability first, the
    model commits to a figure and then produces a quote that contradicts it.
    """
    return {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "speakers": {
                "type": "array",
                "minItems": len(speakers),
                "maxItems": len(speakers),
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": list(speakers)},
                        "evidence": {"type": "string"},
                        "interviewee_probability": {"type": "number"},
                    },
                    "required": ["speaker", "evidence", "interviewee_probability"],
                },
            },
        },
        "required": ["session", "speakers"],
    }


def _load_segments(transcript_input: Union[dict, os.PathLike, list]) -> list[dict]:
    """
    Accept the same inputs as relabel_batched_transcript (segment list, batch summary,
    concatenated JSON, or an output folder) and return plain segment dicts.
    """
    if isinstance(transcript_input, (str, Path)):
        p = Path(transcript_input)
        if p.is_file() and p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data and "speaker" in data[0]:
                return [dict(s) for s in data]
        return concatenate_batch_transcripts(p)
    if isinstance(transcript_input, dict):
        return concatenate_batch_transcripts(transcript_input)
    if isinstance(transcript_input, list):
        if transcript_input and isinstance(transcript_input[0], dict) and "transcript" in transcript_input[0]:
            return concatenate_batch_transcripts(transcript_input)
        return [dict(s) for s in transcript_input]
    raise ValueError(f"Unsupported transcript_input type: {type(transcript_input)}")


def speaker_speech_time(segments: list[dict]) -> dict[str, float]:
    """Seconds of speech per speaker, used to weight votes and to order labels."""
    totals: dict[str, float] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk is None:
            continue
        totals[spk] = totals.get(spk, 0.0) + max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    return totals


def transcript_turns(segments: list[dict]) -> list[dict]:
    """
    Collapse consecutive segments by the same speaker into turns and drop timestamps.
    Timestamps trebled the prompt length without helping the model tell roles apart,
    while the turn structure (who answers whom) is exactly what it needs.
    """
    turns: list[dict] = []
    last_file = None
    for seg in segments:
        spk = seg.get("speaker")
        text = str(seg.get("text", "")).strip()
        file_id = seg.get("video") or seg.get("file") or seg.get("file_name") or ""
        new_recording = bool(file_id) and file_id != last_file
        last_file = file_id
        if not text:
            continue
        if turns and not new_recording and turns[-1]["speaker"] == spk:
            turns[-1]["text"] = f"{turns[-1]['text']} {text}"
            turns[-1]["seconds"] += max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        else:
            turns.append({"speaker": spk,
                          "text": text,
                          "seconds": max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))),
                          "new_recording": new_recording})
    return turns


def speaker_statistics(turns: list[dict]) -> dict[str, dict]:
    """
    Per-speaker summary shown to the model alongside the text. The question rate is the
    single most role-diagnostic cheap statistic, and stating it up front stops the model
    from equating "talks the most" with "is the participant".
    """
    stats: dict[str, dict] = {}
    for turn in turns:
        s = stats.setdefault(turn["speaker"], {"turns": 0, "seconds": 0.0, "questions": 0, "words": 0})
        s["turns"] += 1
        s["seconds"] += turn["seconds"]
        s["words"] += len(turn["text"].split())
        if "?" in turn["text"]:
            s["questions"] += 1
    for s in stats.values():
        s["question_rate"] = s["questions"] / s["turns"] if s["turns"] else 0.0
    return stats


def _format_statistics(stats: dict[str, dict], speakers: list[str]) -> str:
    lines = []
    for spk in speakers:
        s = stats.get(spk, {"seconds": 0.0, "turns": 0, "words": 0, "question_rate": 0.0})
        lines.append(f"- {spk}: {s['seconds']:.0f}s of speech, {s['turns']} turns, "
                     f"{s['words']} words, {s['question_rate']:.0%} of turns contain a question")
    return "\n".join(lines)


def chunk_turns(turns: list[dict], chunk_chars: int = DEFAULT_CHUNK_CHARS) -> list[list[dict]]:
    """
    Split turns into prompt-sized chunks on turn boundaries. Long transcripts are voted
    on chunk by chunk rather than truncated, so a speaker who only appears late still
    gets classified.
    """
    chunks: list[list[dict]] = [[]]
    size = 0
    for turn in turns:
        cost = len(turn["text"]) + len(str(turn["speaker"])) + 4
        if chunks[-1] and size + cost > chunk_chars:
            chunks.append([])
            size = 0
        chunks[-1].append(turn)
        size += cost
    return [c for c in chunks if c]


def render_turns(turns: list[dict]) -> str:
    """Render turns as the '[speaker] text' form the model is asked to reason over."""
    lines = []
    for i, turn in enumerate(turns):
        if turn["new_recording"] and i:
            lines.append("\n--- new recording ---")
        lines.append(f"[{turn['speaker']}] {turn['text']}")
    return "\n".join(lines)


def _build_prompt(chunk: list[dict],
                  speakers: list[str],
                  stats: dict[str, dict],
                  num_interviewees: Union[int, None],
                  chunk_index: int,
                  chunk_count: int) -> str:
    header = [f"Speakers to label ({len(speakers)}): {', '.join(speakers)}",
              f"Return exactly {len(speakers)} object(s), one per speaker, in that order.",
              "",
              "Whole-transcript statistics for these speakers:",
              _format_statistics(stats, speakers)]
    if num_interviewees is not None:
        # A known participant count is the strongest constraint available, so it is
        # stated as a fact rather than a hint. Counts are still enforced afterwards.
        others = len(speakers) - num_interviewees
        header += ["",
                   f"It is known that exactly {num_interviewees} of these {len(speakers)} "
                   f"speakers is/are interviewees and the other {others} is/are interviewers."]
    if chunk_count > 1:
        header += ["",
                   f"This is excerpt {chunk_index + 1} of {chunk_count} from the same session. "
                   "Label only from what you see here; the excerpts are pooled afterwards. "
                   "If a speaker barely appears in this excerpt, give them a probability near 0.5."]
    header += ["", "Transcript:", render_turns(chunk)]
    return "\n".join(header)


def _ollama_chat(prompt: str,
                 schema: dict,
                 model: str = DEFAULT_MODEL,
                 host: str = DEFAULT_HOST,
                 num_ctx: Union[int, None] = None,
                 timeout: float = 600.0) -> dict:
    """
    One structured-output call against a local Ollama server. Returns the parsed JSON
    object; raises RuntimeError with an actionable message when the server or model is
    not there, since that is by far the most common failure.
    """
    if num_ctx is None:
        # ~4 characters per token, plus room for the instructions and the answer, rounded
        # up to a power of two. Ollama silently truncates anything over num_ctx, which
        # would drop the end of a chunk without any warning.
        needed = len(prompt) // 4 + len(SYSTEM_PROMPT) // 4 + 1024
        num_ctx = 8192
        while num_ctx < needed:
            num_ctx *= 2
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.0, "num_ctx": num_ctx},
    }
    url = f"{host.rstrip('/')}/api/chat"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with 'ollama serve' "
            f"(or launch the Ollama app) and try again."
        ) from e
    if response.status_code == 404:
        raise RuntimeError(f"Ollama has no model '{model}'. Pull it first: ollama pull {model}")
    if not response.ok:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text[:500]}")

    content = response.json().get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some models wrap the object in prose or a code fence despite the schema.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Could not parse a JSON object from the model reply: {content[:500]}")


def _chunk_votes(chunk: list[dict], reply: dict, speakers: set[str]) -> list[dict]:
    """
    Turn one model reply into at most one vote per speaker, weighted by how much that
    speaker actually said in this chunk, so a chunk in which someone spoke three words
    cannot outweigh one in which they spoke for minutes.

    A reply may still name the same speaker repeatedly (models drift into one entry per
    turn on long excerpts), so a speaker's entries are averaged: a chunk is one vote no
    matter how the model chose to phrase it.
    """
    chunk_seconds: dict[str, float] = {}
    for turn in chunk:
        chunk_seconds[turn["speaker"]] = chunk_seconds.get(turn["speaker"], 0.0) + turn["seconds"]

    tally: dict[str, list[tuple[float, str]]] = {}
    for item in reply.get("speakers", []):
        spk = item.get("speaker")
        if spk not in speakers:
            continue
        try:
            probability = float(item.get("interviewee_probability", 0.5))
        except (TypeError, ValueError):
            probability = 0.5
        # Models occasionally answer on a 0-100 scale despite the instructions.
        if probability > 1.0:
            probability = probability / 100.0
        probability = min(max(probability, 0.0), 1.0)
        tally.setdefault(spk, []).append((probability, str(item.get("evidence", ""))))

    votes = []
    for spk, entries in tally.items():
        probability = sum(p for p, _ in entries) / len(entries)
        # A minimum of one second keeps a vote from being discarded entirely when the
        # speaker's turns in this chunk carry no usable timestamps.
        weight = max(chunk_seconds.get(spk, 0.0), 1.0)
        quotes = [(p if probability >= 0.5 else -p, e) for p, e in entries if e]
        votes.append({"speaker": spk,
                      "probability": probability,
                      "role": "interviewee" if probability > 0.5 else "interviewer",
                      "weight": weight,
                      "entries": len(entries),
                      "evidence": max(quotes)[1] if quotes else ""})
    return votes


def _assign_roles(scores: dict[str, float],
                  speech_time: dict[str, float],
                  num_interviewees: Union[int, None] = None,
                  num_interviewers: Union[int, None] = None,
                  verbose: bool = False) -> dict[str, str]:
    """
    Turn interviewee scores into roles, honouring an asserted count. Ranking by score
    and cutting at the requested count is what makes 'there are two participants'
    enforceable even when the model labels three speakers as participants.
    """
    ranked = sorted(scores, key=lambda s: (-scores[s], -speech_time.get(s, 0.0), _speaker_sort_key(s)))
    n = len(ranked)

    if num_interviewers is not None:
        if not 0 <= num_interviewers <= n:
            raise ValueError(f"num_interviewers={num_interviewers} is impossible for {n} speaker(s)")
        implied = n - num_interviewers
        if num_interviewees is not None and num_interviewees != implied:
            raise ValueError(f"num_interviewees={num_interviewees} and num_interviewers={num_interviewers} "
                             f"do not add up to the {n} speaker(s) in the transcript")
        num_interviewees = implied

    if num_interviewees is not None:
        if not 0 <= num_interviewees <= n:
            raise ValueError(f"num_interviewees={num_interviewees} is impossible for {n} speaker(s)")
        chosen = ranked[:num_interviewees]
        disagreed = [s for s in chosen if scores[s] <= 0.5] + [s for s in ranked[num_interviewees:] if scores[s] > 0.5]
        if disagreed and verbose:
            print(f"  Forced {num_interviewees} interviewee(s); the model disagreed about "
                  f"{', '.join(sorted(disagreed, key=_speaker_sort_key))}")
    else:
        chosen = [s for s in ranked if scores[s] > 0.5]
        if n >= 2 and len(chosen) in (0, n):
            # Every speaker landed on the same side. A session with no interviewer (or no
            # participant) is far less likely than an over-confident model, so split at
            # the widest score gap instead of returning a single-role transcript. Ties go
            # to the smallest cut, which for equal scores means the single speaker who
            # talks the most becomes the interviewee - the usual shape of an interview.
            gaps = [(scores[ranked[k - 1]] - scores[ranked[k]], -k) for k in range(1, n)]
            cut = -max(gaps)[1]
            chosen = ranked[:cut]
            if verbose:
                print(f"  Model put every speaker in one role; split at the largest score gap "
                      f"into {cut} interviewee(s)")

    chosen_set = set(chosen)
    return {spk: ("interviewee" if spk in chosen_set else "interviewer") for spk in ranked}


def identify_speaker_roles(transcript_input: Union[dict, os.PathLike, list],
                           num_interviewees: Union[int, None] = None,
                           num_interviewers: Union[int, None] = None,
                           model: str = DEFAULT_MODEL,
                           host: str = DEFAULT_HOST,
                           chunk_chars: int = DEFAULT_CHUNK_CHARS,
                           timeout: float = 600.0,
                           verbose: bool = False) -> dict:
    """
    Read a transcript with a local LLM and decide which speakers are interviewers.

    Args:
        transcript_input: Concatenated transcript JSON path, batch summary dict, output
                          folder, or a list of segment dicts (as relabel_* accept).
        num_interviewees: Assert how many of the speakers are interviewees. Required for
                          reliable results with more than two speakers; when omitted the
                          count is inferred from the model's votes.
        num_interviewers: Alternative way to state the same split; mutually consistent
                          with num_interviewees.
        model: Ollama model tag, e.g. "gemma3:12b" or "gemma3:4b".
        host: Ollama base URL.
        chunk_chars: Transcript characters per LLM call.
        timeout: Seconds allowed per LLM call.
        verbose: Print per-chunk progress and the pooled scores.

    Returns:
        dict with 'roles' (speaker -> role), 'scores' (speaker -> interviewee score in
        0-1), 'evidence' (speaker -> best supporting quote), 'speech_time', 'model' and
        the raw per-chunk 'votes'.
    """
    segments = _load_segments(transcript_input)
    if not segments:
        raise ValueError("Transcript contains no segments")

    speech_time = speaker_speech_time(segments)
    turns = transcript_turns(segments)
    stats = speaker_statistics(turns)
    # 'unknown' is leftover unmatched words, not a person, so it is neither role.
    speakers = sorted((s for s in stats if s != UNKNOWN_SPEAKER), key=_speaker_sort_key)
    if not speakers:
        raise ValueError("Transcript contains no labelled speakers")

    chunks = chunk_turns([t for t in turns if t["speaker"] in set(speakers)], chunk_chars)
    schema = _response_schema(speakers)
    if verbose:
        print(f"Classifying {len(speakers)} speaker(s) with {model} over {len(chunks)} chunk(s)")

    pooled = {spk: {"probability": 0.0, "weight": 0.0} for spk in speakers}
    evidence: dict[str, tuple[float, str]] = {}
    all_votes = []
    for i, chunk in enumerate(chunks):
        prompt = _build_prompt(chunk, speakers, stats, num_interviewees, i, len(chunks))
        reply = _ollama_chat(prompt, schema, model=model, host=host, timeout=timeout)
        votes = _chunk_votes(chunk, reply, set(speakers))
        for vote in votes:
            pooled[vote["speaker"]]["probability"] += vote["probability"] * vote["weight"]
            pooled[vote["speaker"]]["weight"] += vote["weight"]
            # Keep the quote from the chunk that spoke loudest about this speaker.
            best = evidence.get(vote["speaker"])
            if vote["evidence"] and (best is None or vote["weight"] > best[0]):
                evidence[vote["speaker"]] = (vote["weight"], vote["evidence"])
        # The model's read of the session is kept for the report: when a call goes wrong
        # it usually says so here ("two staff members are demonstrating the device").
        all_votes.append({"chunk": i, "session": str(reply.get("session", "")), "votes": votes})
        if verbose:
            summary = ", ".join(f"{v['speaker']}={v['probability']:.2f}"
                                + (f" ({v['entries']} entries)" if v["entries"] > 1 else "")
                                for v in votes)
            print(f"  chunk {i + 1}/{len(chunks)}: {summary or 'no usable votes'}")

    # Speech-time-weighted mean probability. No vote at all leaves a speaker at 0.5, so
    # the count enforcement and speech-time tie-break decide rather than an arbitrary role.
    scores = {spk: (pooled[spk]["probability"] / pooled[spk]["weight"]
                    if pooled[spk]["weight"] else 0.5)
              for spk in speakers}

    roles = _assign_roles(scores, speech_time, num_interviewees, num_interviewers, verbose=verbose)
    if verbose:
        for spk in sorted(roles, key=_speaker_sort_key):
            print(f"  {spk}: {roles[spk]} (interviewee score {scores[spk]:.2f}, "
                  f"{speech_time.get(spk, 0.0):.0f}s)")

    return {"roles": roles,
            "scores": scores,
            "evidence": {k: v[1] for k, v in evidence.items()},
            "speech_time": speech_time,
            "statistics": stats,
            "model": model,
            "num_interviewees": sum(1 for r in roles.values() if r == "interviewee"),
            "votes": all_votes}


def role_speaker_ids(roles: dict[str, str],
                     speech_time: Union[dict[str, float], None] = None,
                     interviewer_label: str = INTERVIEWER_LABEL,
                     interviewee_label: str = INTERVIEWEE_LABEL) -> dict[str, str]:
    """
    Build the speaker -> name mapping that relabel_batched_transcript takes. A role held
    by one speaker keeps the bare label; several speakers in one role are numbered by
    descending speech time, so Interviewer_1 is the one who ran most of the session.
    """
    speech_time = speech_time or {}
    mapping: dict[str, str] = {}
    for role, label in (("interviewer", interviewer_label), ("interviewee", interviewee_label)):
        members = sorted((s for s, r in roles.items() if r == role),
                         key=lambda s: (-speech_time.get(s, 0.0), _speaker_sort_key(s)))
        if len(members) == 1:
            mapping[members[0]] = label
        else:
            for i, spk in enumerate(members, start=1):
                mapping[spk] = f"{label}_{i}"
    return mapping


def label_speaker_roles(transcript_input: Union[dict, os.PathLike, list],
                        num_interviewees: Union[int, None] = None,
                        num_interviewers: Union[int, None] = None,
                        model: str = DEFAULT_MODEL,
                        host: str = DEFAULT_HOST,
                        chunk_chars: int = DEFAULT_CHUNK_CHARS,
                        interviewer_label: str = INTERVIEWER_LABEL,
                        interviewee_label: str = INTERVIEWEE_LABEL,
                        output_path: Union[os.PathLike, None] = None,
                        relabel: bool = True,
                        verbose: bool = False) -> tuple[list[dict], dict]:
    """
    Identify roles with the local LLM and rewrite the transcript's speaker labels.

    Returns:
        (segments, report) where segments is the relabeled transcript and report is the
        dict from identify_speaker_roles plus the 'mapping' that was applied. The report
        is also written next to the relabeled transcript as *.speaker_roles.json.
    """
    report = identify_speaker_roles(transcript_input,
                                   num_interviewees=num_interviewees,
                                   num_interviewers=num_interviewers,
                                   model=model, host=host,
                                   chunk_chars=chunk_chars, verbose=verbose)
    mapping = role_speaker_ids(report["roles"], report["speech_time"],
                              interviewer_label, interviewee_label)
    report["mapping"] = mapping
    if verbose:
        print("Applying: " + ", ".join(f"{k} -> {v}" for k, v in
                                       sorted(mapping.items(), key=lambda kv: _speaker_sort_key(kv[0]))))

    if not relabel:
        return _load_segments(transcript_input), report

    segments = relabel_batched_transcript(transcript_input, mapping, output_path=output_path)

    report_target = Path(output_path) if output_path is not None else (
        Path(transcript_input) if isinstance(transcript_input, (str, Path)) else None)
    if report_target is not None:
        if report_target.is_dir():
            report_target = report_target / "concatenated_transcript"
        report_path = report_target.parent / f"{report_target.stem.removesuffix('.relabeled')}.speaker_roles.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"Saved speaker role report to: {report_path}")

    return segments, report


def label_batch_output(output_dir: os.PathLike,
                       num_interviewees: Union[int, None] = None,
                       num_interviewers: Union[int, None] = None,
                       model: str = DEFAULT_MODEL,
                       host: str = DEFAULT_HOST,
                       chunk_chars: int = DEFAULT_CHUNK_CHARS,
                       interviewer_label: str = INTERVIEWER_LABEL,
                       interviewee_label: str = INTERVIEWEE_LABEL,
                       verbose: bool = False) -> Union[dict, None]:
    """
    Label the concatenated transcript a batch run has just written, in place in its own
    output folder. This is the entry point the batch CLIs use after transcription.

    Returns the role report, or None when the run produced nothing worth labelling (no
    speech at all, or a single speaker, whose role no amount of reading can establish).
    """
    out = Path(output_dir)
    concatenated = out / "concatenated_transcript.json"
    if not concatenated.is_file():
        print("  No concatenated transcript to label.")
        return None

    with open(concatenated, "r", encoding="utf-8") as f:
        segments = json.load(f)
    present = {s.get("speaker") for s in segments} - {UNKNOWN_SPEAKER, None}
    if len(present) < 2:
        print(f"  Only {len(present)} speaker(s) in the transcript; skipping role labelling.")
        return None

    print(f"  Identifying interviewer/interviewee with {model} ...")
    _, report = label_speaker_roles(concatenated,
                                    num_interviewees=num_interviewees,
                                    num_interviewers=num_interviewers,
                                    model=model, host=host, chunk_chars=chunk_chars,
                                    interviewer_label=interviewer_label,
                                    interviewee_label=interviewee_label,
                                    verbose=verbose)
    for spk in sorted(report["roles"], key=_speaker_sort_key):
        print(f"    {spk} -> {report['mapping'].get(spk, spk)} "
              f"(interviewee score {report['scores'][spk]:.2f})")
    return report


def add_role_arguments(parser) -> None:
    """
    Add the role-labelling flags shared by the batch CLIs. Kept here so the three
    entry points cannot drift apart as the options change.
    """
    group = parser.add_argument_group("speaker role labelling (local LLM via Ollama)")
    group.add_argument("--label_roles", action="store_true",
                       help="After transcribing, read the transcript with a local LLM and relabel "
                            "the speakers as interviewer/participant. Implied by --num_interviewees "
                            "or --num_interviewers. Requires a running Ollama server")
    group.add_argument("--num_interviewees", type=int, default=None,
                       help="Assert how many speakers are interviewees. Recommended whenever a "
                            "session has more than two speakers; inferred from the model's votes "
                            "if omitted")
    group.add_argument("--num_interviewers", type=int, default=None,
                       help="Assert how many speakers are interviewers instead (must agree with "
                            "--num_interviewees when both are given)")
    group.add_argument("--role_model", type=str, default=DEFAULT_MODEL,
                       help=f"Ollama model used for role labelling (default {DEFAULT_MODEL}; "
                            f"pull it first with 'ollama pull {DEFAULT_MODEL}')")
    group.add_argument("--role_host", type=str, default=DEFAULT_HOST,
                       help="Ollama base URL")
    group.add_argument("--role_chunk_chars", type=int, default=DEFAULT_CHUNK_CHARS,
                       help="Transcript characters sent per LLM call")
    group.add_argument("--interviewer_label", type=str, default=INTERVIEWER_LABEL,
                       help="Name given to interviewers (numbered when there are several)")
    group.add_argument("--interviewee_label", type=str, default=INTERVIEWEE_LABEL,
                       help="Name given to interviewees (numbered when there are several)")


def roles_requested(args) -> bool:
    """
    True when the parsed arguments ask for role labelling. Asserting a count is taken as
    asking for it, since the count is meaningless otherwise.
    """
    return bool(getattr(args, "label_roles", False)
                or getattr(args, "num_interviewees", None) is not None
                or getattr(args, "num_interviewers", None) is not None)


def label_from_args(output_dir: os.PathLike, args, verbose: bool = False) -> Union[dict, None]:
    """Adapter from a batch CLI's parsed arguments to label_batch_output."""
    return label_batch_output(output_dir,
                              num_interviewees=args.num_interviewees,
                              num_interviewers=args.num_interviewers,
                              model=args.role_model,
                              host=args.role_host,
                              chunk_chars=args.role_chunk_chars,
                              interviewer_label=args.interviewer_label,
                              interviewee_label=args.interviewee_label,
                              verbose=verbose)


# Convenient aliases
identify_interviewer = identify_speaker_roles
relabel_by_role = label_speaker_roles
