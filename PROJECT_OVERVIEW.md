# What this project is, and what we've built — plain-language overview

This is written for anyone who wants to understand the annotation tool without
reading code or the more technical logs (`DECISIONS_LOG.md`,
`MLFLOW_INTEGRATION_ANALYSIS.md`). No engineering background assumed.

---

## 1. What is this tool for?

Railway coaches get inspected by cameras. Those cameras produce raw photos —
of the side of the coach, underneath it, of the wheels, of the buffers. To
train an AI model to spot defects (a cracked wheel, a missing bolt, a
corroded pipe) automatically, you first need thousands of examples where a
human has drawn a box or an outline around each part and said "this is a
coupler," "this is a brake cylinder," "this is a wheel."

**This tool is where that human labeling happens.** An annotator opens a
photo, the tool suggests outlines automatically (using an AI model called
SAM2), the human corrects them, and the result is saved.

Once enough photos are labeled, we hand that labeled data over to a
different team (the "pipeline team") who trains the actual defect-detection
models on it.

---

## 2. The four "views" — what kind of photo, what it's for

The tool organizes photos into four separate buckets, because each camera
angle feeds a different downstream model:

| View | What the photo shows | What it eventually trains |
|---|---|---|
| **side_view** | The side of the coach, close-up | A model that judges whether a part looks damaged (broken, missing, corroded...) |
| **underbelly** | Underneath the coach | A model that spots cracks/corrosion underneath |
| **wheel_shelling** | Close-ups of wheels | A model that measures wheel-tread damage ("shelling") |
| **buffer** | The buffer (coupling) area | A model that just checks: is a buffer visible in this shot or not |

`underbelly` and `wheel_shelling` are still mostly empty — real footage for
those hasn't been collected yet. `side_view` has the most data (~2000
labeled photos already).

---

## 3. Who does what — the roles

- **Annotator** — draws/corrects the outlines on each photo. Anyone can be
  one; it's just a name, not a login.
- **Reviewer** — a second, independent check on an annotator's finished
  work before it's allowed to be used for training ("did this person's
  labels actually look right?").
- **Golden curator** — a small, trusted group who get to build the
  **golden set**: a permanently frozen, hand-picked set of images used only
  to *grade* a trained model, never to train one. This is the one role in
  the tool that's actually locked down — anyone else who tries to touch it
  gets rejected.
- **Pipeline team** — a separate team who takes what we export and actually
  trains/tests the AI models with it. We hand data to them; we don't train
  models ourselves.

---

## 4. The life of one labeled photo, start to finish

```
1. Photo loaded
      ↓
2. AI suggests outlines (SAM2) → annotator fixes them, adds/removes objects,
   sets extra info (component condition, which type of coach, etc.)
      ↓
3. Annotator marks the photo "complete"
      ↓
4. (Automatic) Similar-looking neighboring photos get the same labels copied
   onto them as a head start, so the next annotator isn't starting from zero
      ↓
5. A second person reviews it ("approved" / "rejected")
      ↓
6. Once enough photos are reviewed, someone clicks "Export"
      ↓
7. The tool bundles everything into one immutable package (a "snapshot") —
   photos + labels + a manifest describing exactly what's in it
      ↓
8. That snapshot is handed off to the pipeline team to train/test models
      ↓
   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
   Everything below this line happens in the pipeline team's own systems,
   not in this tool — see section 4a for what that means and why.
   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
      ↓
9. They train a candidate model on our snapshot
      ↓
10. They score that candidate against the golden set (section 5.9) — the
    permanently frozen images nothing ever trains on
      ↓
11. A gate decides: does the new model match or beat the current one on
    *every* safety-relevant part, not just on average? Fail on even one
    such part → rejected, current model keeps running
      ↓
12. Pass → gradual rollout (a "shadow" or "canary" period watching the new
    model on real traffic before fully trusting it), with the ability to
    roll straight back to the previous model if anything looks wrong
```

The key idea running through steps 3–7: **once something has been
exported and handed off, it never silently changes.** If you re-export the
exact same data, you get the exact same package back (byte for byte). If
even one label changes, you get a *new* package, and the old one is still
sitting there unchanged. This matters because a model that was trained on
"package #42" needs to be able to prove, forever, exactly what was in
package #42.

---

## 4a. What happens after handoff — two separate stories now, not one

Steps 9–12 above are the natural next question ("ok, we labeled it — then
what actually happens?"), so they're included for the full picture. But
it's important to be precise about whose responsibility that is — this
section used to describe one story ("none of it is built here"); it's now
two, because one of them got built since.

**Story A — the pipeline team's 8 production defect-detection models.
Unchanged, still entirely their own infrastructure.** Training those
models, tracking those runs, scoring them, deciding whether a new one is
good enough to replace the current one, rolling it out safely — all of
that happens in the pipeline team's *own* tracking system (a tool called
**MLflow** — think of it as a lab notebook that automatically records every
training attempt: what data went in, what settings were used, how well it
scored, which version is running right now). Early on there was a real
option for *us* to run some of that ourselves for their models too; the
pipeline team's answer settled it — **we package data and place it
somewhere they can pick it up ("staging"); we never write into their
tracking system directly.** That's why the flow diagram above visually
separates step 8 from what follows for their models — this project has no
code for and no visibility into that side.

**Story B — the small internal helper AI. This one, we do now track,
end to end.** Separately from the pipeline team's models, this tool uses
its *own* small AI model internally, just to give annotators a head-start
outline when they open a photo (the "SAM2 suggests, human corrects" part
of step 2). That question — should *this* helper get its own tracking —
used to be open. It's answered now: yes, and the whole loop is built. See
section 5.11 onward for what that actually does. The two systems remain
completely separate: this tool has its own small MLflow instance, tracking
only its own helper model, never touching or writing into the pipeline
team's one.

---

## 5. What we've actually built (in plain language)

Below is everything that's been implemented, grouped by what problem it
solves — not by internal code names.

### 5.1 "Don't let the data quietly rot" — the foundational fixes

- **Each dataset view keeps its own AI model and its own settings.**
  Early on, mixing up two views could cause the AI to give confidently wrong
  suggestions on the wrong kind of photo. Fixed.
- **Two separate confidence scores instead of one blurred-together number** —
  "how sure is the AI this is a coupler" is now tracked separately from "how
  good is the outline shape," instead of one number trying to mean both.

### 5.2 "What does this component's condition actually look like?"

Originally, a label only said *what* a part was (a coupler, a wheel). It
never said *how it looked* (fine? broken? missing? leaking?). But the
defect-detection model specifically needs that second piece of information.

We added a **condition field** — 13 possible values (ok, broken, missing,
hanging, leaking, sparking, etc.) — that an annotator sets per object.

Important design choice: for the ~2000 photos labeled *before* this field
existed, we did **not** default them to "ok."  Nobody actually checked
those parts, so claiming "ok" would be lying with data. They're marked
"unassessed" instead, and stay excluded from that particular training set
until a human actually looks and sets a real value.

### 5.3 "A photo with nothing in it is still useful information"

If a photo genuinely has no buffer visible, that's valuable — a model that
detects "is there a buffer" needs *negative* examples just as much as
positive ones. Before, an empty photo just looked like "nobody's gotten to
this yet" and got silently thrown away at export time.

Now there's an explicit **"I looked, there's genuinely nothing here"**
button, separate from "not annotated yet." Only that explicit case gets
exported as a real negative example.

### 5.4 "Thin, branchy damage needs a different kind of label"

A crack or patch of corrosion isn't a neat blob — it can be a thin, forking
line, sometimes with gaps. Squashing that into a simplified polygon (which
is what normal component outlines use) throws away exactly the detail the
model is supposed to be scored on (crack length, extent).

For classes like crack/corrosion/shelling, the tool now:
- keeps every disconnected piece of the shape instead of just the biggest one,
- skips the outline-simplification step that would smooth away fine detail,
- **and** exports a pixel-accurate image mask alongside the outline, not
  just the outline.

### 5.5 "Which coach type is this?"

Indian Railways coaches come in two families (LHB and ICF), and they don't
have the same parts in the same places. Getting this wrong at labeling time
would mean the AI's model of "what part goes where" is wrong for half the
coaches.

Each batch of photos now records LHB / ICF / "unknown" — and "unknown"
deliberately stays "unknown" rather than defaulting to whichever type is
more common, because guessing wrong here is exactly the kind of mistake
that causes a defect-checking system to miss real damage.

### 5.6 "Don't let the meaning of 'class 7' drift over time"

Class lists (the list of component types) change over time — someone adds
a new type, or fixes a typo. If that change isn't tracked carefully, an old
export could silently mean something different than it used to ("class 7"
used to be "spring," now it's "bearing housing" — same number, different
meaning).

Every change to a class list now mints a new, permanent, numbered version.
Every exported package is stamped with exactly which version it used, so
you can always answer "what did class 7 mean in this specific export" —
even years later.

### 5.7 "Exports need to be trustworthy handoff packages, not loose files"

Originally, exporting just overwrote one shared folder. Re-export, and the
previous version was gone with no record of what changed.

Now, every export builds an **immutable, fingerprinted package**
("snapshot"). Export the same data twice, get back a reference to the exact
same package (nothing is duplicated). Change even one label, and you get a
genuinely new package — the old one is untouched. Each package ships with a
plain-text manifest describing exactly what's inside, including some
deliberately uncomfortable numbers (like "how many of these images were
never independently double-checked") so nobody can accidentally overstate
how solid the dataset is.

### 5.8 "AI-generated labels must never sneak into the test set"

Some labels aren't purely human — some get auto-copied onto similar photos,
and some get auto-approved without a human review if the AI was confident
enough and the historical track record for that component type is good.
That's fine for *training* data, but it would be a real problem if any of
that automatically-generated data ended up in the *held-out* portion used
to grade a finished model — that would make the model look better than it
actually is.

The tool now actively **forces** any auto-copied or auto-approved label
into the training portion, never the held-out portion — this is enforced,
not just hoped for. If, despite that, a violation were ever somehow
detected, the export refuses to complete rather than shipping a
compromised package.

### 5.9 "A permanent, untouchable ruler to grade models against" — now actually built and filled

Beyond the ordinary held-out test data, there's a small, hand-picked
**golden set** — images set aside specifically to *evaluate* finished
models, forever separate from anything a model ever trains on.

Only a designated curator can create or add to a golden set (anyone else
is rejected outright); once an image is added, it's permanently excluded
from every future training export, and automated processes (like the label
auto-copying in 5.4) are blocked from ever touching a golden image.

**Update: it's populated now, not just built.** The original plan was to
wait for a named domain expert to hand-pick images from scratch — nobody
had been assigned that job. Instead, the decision was made to use the
existing ~2000-image `side_view` dataset as the source, since it's already
verified and accurate: 200 images were automatically selected to cover
every component type (with the safety-critical ones guaranteed coverage
first), and a curator committed them as the actual golden set for that
view. The other three views (underbelly, wheel-shelling, buffer) don't have
one yet — same as before, they need real footage and review first.

### 5.10 "Wheel photos need to be 'unrolled' before a model can use them"

A wheel is round; damage on its tread needs to be examined as if it were
unrolled into a flat strip (the same idea as unrolling a bandage to see the
whole thing at once, instead of looking at a curled-up loop).

Annotators still draw directly on the normal, round photo — that's the
"ground truth," and it should never have to be redrawn. The unrolling
happens automatically at export time. This matters because the exact
"unrolling recipe" (wheel size, etc.) is still provisional and will
probably be corrected later — if we baked it into every drawn label now,
correcting it later would mean redoing every wheel by hand. Instead,
correcting it later just means re-running the export.

### 5.11 "Actually track the internal helper's own training" (new)

Recall from 4a: this tool has its own small internal AI (SAM2 + a
lightweight detector) that suggests outlines to annotators — completely
separate from the pipeline team's 8 production models. Training a fresh
version of that helper used to just produce a file, with no record of what
was tried, what data it used, or how it did — anyone wanting to know had
to eyeball scattered log lines.

Now it's tracked in a lab notebook of its own (a small MLflow instance,
just for this helper — never the pipeline team's). Every training attempt
records: which settings were used, what happened each round of training,
and all the usual result charts (accuracy curves, a confusion matrix,
example predictions) — none of that used to be kept at all; it was
generated and immediately thrown away.

**And it now starts itself automatically.** Every time a snapshot is
handed off (step 8), a fresh training attempt for the helper kicks off on
its own — nobody has to remember to click a button.

### 5.12 "Score the newly trained helper against the golden ruler" (new)

Training the helper used to stop at "did it finish." Now, right after
training, the resulting model is automatically tested against the golden
set from 5.9 — the same permanently frozen ruler nothing ever trains on —
and the score for *every single component type* is recorded, not just one
overall number. That distinction matters: an overall score can look great
while one specific, safety-relevant part quietly gets worse — recording
every part's score separately is what stops that from hiding.

### 5.13 "Deciding whether a new helper version is actually better" (new)

Once a version is scored, it's logged as a numbered candidate. The tool
then automatically compares it — part by part — against whichever version
is currently marked as "the good one," and tags it with a plain verdict:
*nothing to compare against yet*, *matches or beats the current one on
every part*, or *got worse on these specific parts: ...*.

**Nobody's suggestions change because of this comparison alone.** Marking
a version as "the good one" always happens by a human's own deliberate
action, and even that isn't the final word — see 5.14.

### 5.14 "Two people, two separate green lights, before anything live changes" (new)

This is the safety net around 5.13, and it was deliberately designed this
way rather than made automatic:

1. Someone reviews a candidate's part-by-part scores and marks it "the
   good one" — a judgment about whether the model is actually accurate.
2. **A second, different person** then has to separately approve *actually
   swapping* what live annotators get their suggestions from. This is a
   judgment about whether it's *safe* to make that swap right now — a
   deliberately different question from step 1, and the tool won't let one
   person's identity stand in for both.
3. Only once that second person approves does the tool download the new
   version, quietly test-run it once to make sure it actually works, and
   *then* switch it in — never the other way around, so a broken version
   can never end up live with nothing working at all.
4. If the new version needs to be undone later, that's instant and doesn't
   depend on anything else being reachable — the previous working version
   is always kept on hand.

The tool checks for a new "good one" on its own, automatically, on a
schedule — the only manual step in this entire chain is that second
person's yes/no. And whatever happens in this whole loop, an annotator
opening a photo is never affected by it — suggestions keep coming from
whichever version was last actually approved, instantly, with nothing to
wait on.

### 5.15 "Never let training crowd out a working annotator" (new)

The internal helper's training uses the same graphics hardware the
suggestion feature itself runs on. Training a new version used to have no
awareness of this — it could start at any moment and slow down or stall
suggestions for someone actively working on a photo.

Now, before a training attempt is allowed to actually start the heavy part
of the work, it checks whether the suggestion feature is currently busy,
and if so, it waits — checking back periodically — until things go quiet
before proceeding. Suggestions for a working annotator always win.

---

## 6. What's genuinely still left, and why it's not done yet

Everything above is implemented and tested. What remains is either
blocked on something outside this codebase, or a deliberate, separate
decision not yet made:

### 6.1 Fixing how one internal ID is stored

Right now, one internal identifier is stored as a literal computer file
path (e.g. `/home/.../dataset/side_view`). That's fragile — if the server
is ever moved, or that folder gets renamed, every annotation, review, and
approval record tied to that path effectively goes orphaned.

The fix itself is understood and small. But it means rewriting a value
inside real, already-completed annotation work across five different
tables. Before touching that, we need a full safety-net backup of the live
database taken first (see the pg_dump explanation in the previous message)
— that has to happen on the real production server, which is outside what
we can do from here.

### 6.2 Golden set: done for one view, still needed for three

`side_view`'s golden set is populated now (section 5.9). `underbelly`,
`wheel_shelling`, and `buffer` still don't have one — not blocked on
engineering, just on those views not having enough real, reviewed footage
yet to draw a trustworthy set from.

### 6.3 One open question, not blocking anything

We're waiting on the pipeline team to confirm the exact list of 13
"condition" values (section 5.2) is still current, and to be aware that the
~2000 already-labeled photos won't count toward that particular training
set until someone assesses their condition by hand.

### 6.4 The whole MLflow/training loop (5.11–5.15) is the newest, least-proven piece

It's fully built and has been tested for real — including deliberately
triggering the exact situations it's meant to handle safely (an approval
being rejected, training being made to wait for a busy moment) rather than
just checking that it runs without crashing. But it's also the piece with
the most direct effect on what annotators actually see if something in it
were ever wrong, and it's only ever been run in the local development
copy. Turning it on in the live, currently running version of the tool is
a deliberate, separate decision still to be made — see 7 below.

---

## 7. One more thing worth knowing

Everything described above has been built and tested on a **local
development copy**, completely separate from the live, currently-running
version of this tool. Nothing here has been deployed to production yet —
that's a deliberate, separate decision still to be made, since it involves
touching a live database with real annotation work in it.
