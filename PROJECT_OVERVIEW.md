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
```

The key idea running through steps 3–7: **once something has been
exported and handed off, it never silently changes.** If you re-export the
exact same data, you get the exact same package back (byte for byte). If
even one label changes, you get a *new* package, and the old one is still
sitting there unchanged. This matters because a model that was trained on
"package #42" needs to be able to prove, forever, exactly what was in
package #42.

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

### 5.9 "A permanent, untouchable ruler to grade models against"

Beyond the ordinary held-out test data, there's a plan for a small,
hand-picked **golden set** — images a trusted "curator" selects specifically
to *evaluate* finished models, forever separate from anything a model ever
trains on.

What's built: only a designated curator can create or add to a golden set
(anyone else is rejected outright); once an image is added, it's
permanently excluded from every future training export, and automated
processes (like the label auto-copying in 5.4) are blocked from ever
touching a golden image. What's *not* built yet: nobody has actually
picked which images go in it — that needs a real domain expert to sit down
and choose, and no one's been assigned that job yet.

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

---

## 6. What's genuinely still left, and why it's not done yet

Everything above is implemented and tested. Two things remain, and neither
is a coding gap — both are blocked on something outside this codebase:

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

### 6.2 Actually populating the golden set

The mechanism (section 5.9) is built and locked down. What's missing is a
person: someone with real domain expertise needs to go through the data and
pick which images belong in the permanent evaluation set. Nobody has been
named for that job yet — it's a staffing decision, not an engineering one.

### 6.3 One open question, not blocking anything

We're waiting on the pipeline team to confirm the exact list of 13
"condition" values (section 5.2) is still current, and to be aware that the
~2000 already-labeled photos won't count toward that particular training
set until someone assesses their condition by hand.

---

## 7. One more thing worth knowing

Everything described above has been built and tested on a **local
development copy**, completely separate from the live, currently-running
version of this tool. Nothing here has been deployed to production yet —
that's a deliberate, separate decision still to be made, since it involves
touching a live database with real annotation work in it.
