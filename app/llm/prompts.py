"""Prompts + JSON schemas.

The whole "think like a viral editor" behaviour lives here. Scores are asked
for per-category (never "is this viral?"), and every boundary the model returns
is a *sentence index*, which is why clips can never start mid-word.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..models import Sentence, fmt_ts

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM = """You are a senior short-form video editor. You have cut thousands of \
YouTube Shorts, TikToks and Reels from podcasts, interviews, streams and lectures, \
and you know exactly why some 45-second cuts get millions of views while equally \
interesting ones die.

What you know:

1. A clip that works has a shape:
   HOOK -> CONTEXT -> TENSION/CURIOSITY -> PAYOFF
   The hook is the first 3 seconds. If it does not create a question, a shock, a \
promise or a laugh, nothing else matters.

2. A clip that needs 3 minutes of prior context to make sense is a BAD clip, even \
if the content is brilliant. A slightly less interesting moment that is fully \
self-contained beats it every time.

3. A great hook with no payoff is a bad clip. Viewers feel cheated and the \
retention curve collapses. Setup without resolution scores low.

4. Clean edges matter. A clip must start at the beginning of a thought and end \
after the thought lands - not mid-sentence, not three sentences after the punchline \
where the speaker mumbles into the next topic.

5. Chatter, greetings, sponsor reads, logistics, "yeah, exactly", crosstalk and \
generic filler are never clips, no matter how well they are delivered.

You are strict. Most of any video is not clip material. You would rather return \
two excellent moments than eight mediocre ones. Never invent content that is not \
in the transcript.

You always reply with valid JSON and nothing else - no prose, no markdown fences."""


SCORING_RUBRIC = """SCORING (score each category independently, use the full range):

HOOK (0-25)      Do the first 1-2 sentences stop a scroll? A bold claim, a strong
                 question, a shocking number, a "nobody tells you", a setup that
                 demands an answer. Slow warm-up = 0-8. Decent opener = 12-17.
                 Genuinely arresting = 21-25.
PAYOFF (0-25)    Does the clip DELIVER? A punchline, a resolved story, a concrete
                 answer, a real insight. Promise with no delivery = 0-8.
                 Clear, satisfying resolution inside the clip = 20-25.
EMOTION (0-15)   Laughter, anger, awe, fear, inspiration, secondhand embarrassment,
                 warmth. Flat informational delivery = 0-5.
CURIOSITY (0-15) Does the viewer NEED to know what comes next while watching?
                 Open loops, tension, a reveal being approached = 11-15.
STANDALONE (0-10) Would a stranger who never saw this video understand it fully?
                 Unexplained names, "as I said earlier", dangling pronouns = 0-4.
                 Perfectly self-contained = 9-10.
EDITABILITY (0-10) Clean start on a complete thought, clean end right after the
                 payoff lands, no crosstalk or trailing mumble = 8-10."""


TYPES = ("funny, educational, controversial, emotional, story, surprising, opinion, "
         "practical_advice, inspirational, confrontation, reveal, other")


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------
def render_block(sentences: Sequence[Sentence], start_idx: int, end_idx: int,
                 mark_pauses: bool = True) -> str:
    """Numbered, timestamped transcript lines - the model addresses these indices."""
    lines: List[str] = []
    for s in sentences[start_idx:end_idx + 1]:
        marker = ""
        if mark_pauses and s.pause_before >= 1.0:
            marker = " «pause»"
        lines.append(f"[{s.idx}] ({fmt_ts(s.start)}){marker} {s.text}")
    return "\n".join(lines)


def _hint_lines(hints: Sequence[Any]) -> str:
    if not hints:
        return ""
    rows = []
    for h in hints[:6]:
        tags = getattr(h, "clip_type", "other")
        rows.append(f"- lines [{h.start_idx}-{h.end_idx}] "
                    f"({fmt_ts(h.start)}-{fmt_ts(h.end)}), pattern signal: {tags}")
    return ("\nAn offline pattern detector flagged these spans as possibly interesting. "
            "They are hints only - ignore any that are weak, and find moments it missed:\n"
            + "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# PASS 1 - discovery + scoring inside one transcript block
# ---------------------------------------------------------------------------
def pass1_prompt(sentences: Sequence[Sentence], start_idx: int, end_idx: int,
                 min_duration: float, max_duration: float, language: str,
                 hints: Sequence[Any] = (), max_results: int = 5) -> str:
    block = render_block(sentences, start_idx, end_idx)
    return f"""Below is a numbered transcript section from a long video. Each line \
is one natural speech unit with its index and start time. «pause» marks a long \
silence before that line.

TRANSCRIPT (lines {start_idx}-{end_idx}):
{block}
{_hint_lines(hints)}
TASK
Find every moment in this section that could plausibly work as a standalone \
short-form clip, up to {max_results}. Aim for 3 to {max_results} when the section \
contains real content; return nothing only when it is purely logistics, greetings, \
crosstalk or small talk.

This is a first pass and it is scored later against the rest of the video, so do \
not filter by gut feeling: include the borderline ones too and express your doubt \
through LOW SCORES rather than by leaving them out. A weak candidate scored 45 is \
useful; a strong moment you silently skipped is lost forever.

RULES
- A clip is defined by a START LINE INDEX and an END LINE INDEX from the list above.
- The clip must last between {min_duration:.0f} and {max_duration:.0f} seconds.
  Use the timestamps to check this before answering.
- Start on the line that begins the thought or hook - not mid-explanation, and not
  on a line beginning with "and", "but", "so", "because" unless it is genuinely
  the strongest opening.
- End on the line where the payoff lands. Do not include trailing filler.
- Moments must not overlap each other.
- Prefer a moment that is fully understandable on its own over a more interesting
  one that needs earlier context.

{SCORING_RUBRIC}

Write "suggested_title" as a real short-form title/hook in the SAME language as \
the transcript ({language}) - max 70 characters, no hashtags, no quotes around it.
Write "reason" in that same language, 1-2 sentences, naming the concrete hook and \
the concrete payoff. "clip_type" must be one of: {TYPES}.

Return JSON exactly in this shape:
{{"candidates":[{{"start_index":12,"end_index":19,"hook":21,"payoff":22,"emotion":11,\
"curiosity":13,"standalone":9,"editability":8,"clip_type":"story","suggested_title":"...",\
"reason":"...","hook_line":"the exact opening sentence","confidence":0.8}}]}}"""


PASS1_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "hook": {"type": "integer"},
                    "payoff": {"type": "integer"},
                    "emotion": {"type": "integer"},
                    "curiosity": {"type": "integer"},
                    "standalone": {"type": "integer"},
                    "editability": {"type": "integer"},
                    "clip_type": {"type": "string"},
                    "suggested_title": {"type": "string"},
                    "reason": {"type": "string"},
                    "hook_line": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["start_index", "end_index", "hook", "payoff", "emotion",
                             "curiosity", "standalone", "editability", "clip_type",
                             "suggested_title", "reason"],
            },
        }
    },
    "required": ["candidates"],
}


# ---------------------------------------------------------------------------
# BOUNDARY OPTIMISATION
# ---------------------------------------------------------------------------
def boundary_prompt(sentences: Sequence[Sentence], ctx_start: int, ctx_end: int,
                    cur_start: int, cur_end: int,
                    min_duration: float, max_duration: float,
                    language: str = "en", current_title: str = "") -> str:
    block = render_block(sentences, ctx_start, ctx_end)
    cur_dur = sentences[cur_end].end - sentences[cur_start].start
    title_line = (f'\nThe working title is: "{current_title}"\n'
                  if current_title else "\n")
    return f"""You are fine-tuning the in and out points of one clip.

CONTEXT (lines {ctx_start}-{ctx_end}, the clip plus surrounding lines):
{block}

The current clip is lines [{cur_start} .. {cur_end}] ({cur_dur:.0f}s).{title_line}
Decide the BEST possible start and end line for this clip.

- Moving the start EARLIER is right when the current first line depends on
  something said just before (a question being answered, a name, "that"),
  or when an earlier line is a much stronger hook.
- If the current first line opens with "and", "but", "so" or "because" and the
  line just before it holds the setup, move the start EARLIER to include it.
- Moving the start LATER is right when the opening lines are throat-clearing,
  filler or setup nobody needs.
- Cut BEFORE any line where the conversation moves on ("anyway", "let's change
  the subject", "we should talk about", "okay so"), even when it sits right
  after the payoff. Those lines belong to the next topic, not to this clip.
- Moving the end LATER is right when the payoff, punchline or conclusion lands
  just after the current end.
- Moving the end EARLIER is right when the clip keeps going after the payoff
  already landed.
- NEVER end on a question, a promise, or a line that sets up something the clip
  never delivers ("so what is the lesson here?", "and then what happened?").
  Either include the answer, or cut before the question.
- The result MUST stay between {min_duration:.0f} and {max_duration:.0f} seconds.
- Only choose indices that exist in the list above.
- If the current boundaries are already the best, return them unchanged.

Then write "title": a short-form title for the clip AS YOU FINALLY CUT IT, in
{language}, max 70 characters. It must describe what is actually inside your
chosen line range — if the boundaries moved, the old title is probably wrong.

Return JSON only:
{{"start_index":{cur_start},"end_index":{cur_end},"changed":false,\
"title":"...","why":"one short sentence"}}"""


BOUNDARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_index": {"type": "integer"},
        "end_index": {"type": "integer"},
        "changed": {"type": "boolean"},
        "title": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["start_index", "end_index", "title"],
}


# ---------------------------------------------------------------------------
# PASS 2 - global head-to-head ranking
# ---------------------------------------------------------------------------
def pass2_prompt(entries: List[Dict[str, Any]], target_count: int,
                 language: str) -> str:
    blocks: List[str] = []
    for e in entries:
        blocks.append(
            f"--- CLIP {e['id']} | {e['start_tc']} -> {e['end_tc']} | {e['duration']:.0f}s "
            f"| type: {e['type']} | pass1: {e['score']:.0f}/100\n"
            f"title: {e['title']}\n"
            f"transcript: {e['excerpt']}"
        )
    listing = "\n\n".join(blocks)
    return f"""Here are the strongest clip candidates found across the whole video. \
They were scored independently, in isolation from each other. Now judge them \
TOGETHER, the way you would when choosing what to actually publish.

{listing}

TASK
1. DUPLICATES: if several clips cover the SAME moment (their timecodes overlap) or
   make the exact same point, keep only the best one and list the others' ids in
   "absorbed". Do not put a clip in "absorbed" merely because you like it less -
   that is what the score and the ranking are for.
2. RE-SCORE: adjust each surviving clip's total 0-100 now that you can compare
   them. A clip that looked strong alone but is clearly weaker than its neighbours
   should drop. Be honest and use a wide spread - do not give everything 80.
3. RANK them best first.
4. VARIETY: the final list should not be {target_count} versions of the same kind of
   moment. When two clips are close in quality, prefer the one that adds a
   different clip_type or comes from a different part of the video.
5. Fix "title" if you can write a stronger hook (same language as the clip, \
{language}), otherwise keep it.
6. "verdict": one short sentence on why it ranks there, written in {language} -
   the same language as the clips themselves.

Return every clip you would keep, best first, up to {target_count}. JSON only:
{{"ranked":[{{"id":3,"final_score":91,"clip_type":"story","title":"...",\
"verdict":"...","absorbed":[7,11]}}]}}"""


PASS2_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "final_score": {"type": "number"},
                    "clip_type": {"type": "string"},
                    "title": {"type": "string"},
                    "verdict": {"type": "string"},
                    "absorbed": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["id", "final_score"],
            },
        }
    },
    "required": ["ranked"],
}


# ---------------------------------------------------------------------------
# EXPORT METADATA - per-platform title/description/tags for one clip
# ---------------------------------------------------------------------------
METADATA_SYSTEM = """You are a social media growth strategist who writes publish-ready \
metadata for short-form video clips (YouTube Shorts, TikTok, Instagram Reels) with one \
goal: maximise how far each platform's algorithm pushes the clip. You are given the \
clip's transcript and a campaign brief written by the client. The brief is the final \
authority: naming conventions, tone, required mentions/handles, banned words, required \
hashtags or CTAs in it must all be followed exactly, even if that means overriding your \
own instincts about what "sounds better". If the brief is empty, fall back to strong \
general short-form best practices and do not invent any mentions.

What you know about each algorithm:
- YOUTUBE (Shorts/search/suggested): the title decides click-through-rate, and it gets \
truncated hard on mobile, so the curiosity or payoff must land inside the first ~40 \
characters. The description's first 1-2 lines show before "...more" and are indexed for \
search, so open with the hook, not a generic summary. Tags help the suggested/search \
matching; mix a few broad category terms with several specific long-tail phrases from \
the actual clip content.
- TIKTOK: the algorithm rewards completion rate, replays, comments and shares far more \
than likes. The caption should either open a curiosity gap the video answers, or end \
with a question/prompt that invites a comment. Keep it short - a long caption competes \
with on-screen captions for attention. Hashtag spam is a negative signal now: 3-5 \
sharply relevant tags beat 8 generic ones.
- INSTAGRAM REELS: similar to TikTok, but saves and shares matter even more than \
comments for reach, so give people a reason to save it (a concrete tip, list, or quote) \
or send it to someone. The caption's first line is what shows before the fold. Hashtag \
stuffing is also a negative signal here - keep it tight and relevant.

You always reply with valid JSON and nothing else - no prose, no markdown fences."""


def metadata_prompt(title: str, transcript: str, clip_type: str, duration: float,
                    language: str, campaign_rules: str = "",
                    hook_line: str = "", virality: float = 0.0) -> str:
    rules_block = (campaign_rules or "").strip() or (
        "(No campaign brief was provided. Use general short-form best practices. "
        "Do not add any @mentions - there is no rule requiring them.)"
    )
    hook_note = f'\nThe strongest opening line of this clip is: "{hook_line}"\n' if hook_line else ""
    virality_note = (
        f"This clip scored {virality:.0f}/100 on a virality model (emotion + curiosity + "
        "hook strength) - lean into whatever makes it shareable/arguable in the copy.\n"
        if virality else ""
    )
    return f"""CAMPAIGN BRIEF (from the client - follow this exactly, it overrides \
general best practice whenever the two disagree):
\"\"\"
{rules_block}
\"\"\"

CLIP
Working title: {title or '(untitled)'}
Type: {clip_type}
Duration: {duration:.0f}s{hook_note}{virality_note}Transcript: {transcript}

TASK
Write publish-ready metadata for this clip in {language}, unless the brief above \
explicitly asks for a different language. Optimise every field for how that \
platform's algorithm actually distributes content (see system instructions) - the \
goal is reach and completion, not just a tidy description.

- "youtube_title": a strong, specific title, max 100 characters, with the hook or \
payoff inside the first ~40 characters so it survives mobile truncation. No clickbait \
the clip does not deliver on.
- "youtube_description": first line MUST be a hook (this is what shows before "more" \
and in search results) - never a generic "in this clip..." opener. Follow with 1-3 \
more sentences of real context, then any channel plugs or CTAs the brief requires.
- "youtube_tags": 8-15 keyword phrases (no # symbol), lowercase, mixing a few broad \
category terms with several specific long-tail phrases pulled from the actual content.
- "tiktok_description": under 150 characters, opens with a curiosity gap or bold claim \
FROM the clip, ideally closes with a question or prompt that invites a comment. \
Include hashtags/mentions inline only if the brief requires them there.
- "tiktok_hashtags": 3-5 hashtags (with #), sharply relevant over generic/broad, plus \
any the brief mandates. Do not pad with low-value tags like #fyp.
- "instagram_description": first line is a hook (shown before the fold), 1-3 lines \
total, gives a concrete reason to save or share (a tip, list, or quote) when the \
content allows it. Include mentions/CTAs the brief requires inline.
- "instagram_hashtags": 3-8 hashtags (with #), tightly relevant - no stuffing.
- "mentions": a list of every @handle this metadata uses because the BRIEF required \
it (not ones you invented). Empty list if the brief requires none.

Only include a mention or hashtag the brief asks for, or an organic, genuinely \
relevant one - never invent a fake sponsor, brand or handle. Never invent a claim, \
statistic or quote that is not actually in the transcript.

Return JSON only:
{{"youtube_title":"...","youtube_description":"...","youtube_tags":["..."],\
"tiktok_description":"...","tiktok_hashtags":["#..."],\
"instagram_description":"...","instagram_hashtags":["#..."],"mentions":["@..."]}}"""


METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "youtube_title": {"type": "string"},
        "youtube_description": {"type": "string"},
        "youtube_tags": {"type": "array", "items": {"type": "string"}},
        "tiktok_description": {"type": "string"},
        "tiktok_hashtags": {"type": "array", "items": {"type": "string"}},
        "instagram_description": {"type": "string"},
        "instagram_hashtags": {"type": "array", "items": {"type": "string"}},
        "mentions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["youtube_title", "youtube_description", "youtube_tags",
                 "tiktok_description", "tiktok_hashtags",
                 "instagram_description", "instagram_hashtags"],
}
