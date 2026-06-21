# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in an online community of users' choosing.

## Project Documentation

### Community choice and reasoning
I chose **[r/YouShouldKnow (r/YSK)](https://www.reddit.com/r/YouShouldKnow/)**, a subreddit whose entire premise is users sharing practical knowledge, life tips, safety warnings, and "things you should know", framed as advice to the wider community. It suits a classification task because the comment section under any popular post is not monolithic: a single thread mixes genuinely actionable advice, people recounting their own lived experiences, and a steady stream of jokes and banter riffing on the topic. That natural three-way split in *how people talk* (instructional vs. anecdotal vs. humorous) is varied and semantically distinct enough to be both learnable by a classifier and useful to the community.

### Label taxonomy
Three labels, defined by the author's **primary communicative intent**:

- **`tip`**, a comment offering generalizable, actionable advice or factual knowledge intended to help others, phrased as instruction, recommendation, or a correction of fact rather than as a personal story.
  - *Ex A:* "Another tip I found out while not having AC: when it is cooler outside than inside, point your fan TOWARDS your open window. It forces the hot air out, and negative pressure lets cooler outside air in!"
  - *Ex B:* "Running cold water over your forearms for a minute or two is a great way to cool down. Your forearms have lots of blood vessels that make this work."
- **`personal_experience`**, a comment in which the author primarily relates their own first-hand experience or anecdote, narrated in the first person and grounded in something that happened to *them*, even if a tip is implied.
  - *Ex A:* "TIL, thank you. I have frequent cramps and our heating was out, so I spent almost the whole winter with an electric heating pad on my stomach. Wondered about those spots that didn't go away."
  - *Ex B:* "I did that so many times when I was walking in Rome for 10 hours a day in August. Almost instant relief."
- **`joke`**, a comment whose primary intent is humor, wordplay, sarcasm, or absurdity rather than to inform or share a genuine experience.
  - *Ex A:* "Slow way: wait for the next full moon then present them with three pebbles from a nearby stream. Fast way: preheat the oven to 430F."
  - *Ex B:* "If it be not a gooch, then surely 'tis a cooch."

See [planning.md](planning.md) for the full label definitions, edge cases, and methodology.

### Data collection, labeling, and distribution
- **Source:** comments scraped from a single r/YouShouldKnow thread (["YSK about this excellent trick to keep cool in a heatwave"](https://www.reddit.com/r/YouShouldKnow/comments/1tm6x7y/ysk_about_this_excellent_trick_to_keep_cool_in_a/)), saved as HTML and parsed into `Dataset.csv`.
- **Cleaning:** the raw scrape yielded **314** comments; I removed duplicates, very short comments (≤8 words), question-only comments, and downvoted comments (score < 0), leaving **210** clean comments as the labeling pool.
- **Labeling process:** each comment was hand-labeled by its primary communicative intent using a fixed decision procedure, (1) *humor first*: if the dominant purpose is to be funny, it is a `joke`; (2) otherwise the *generalizability test*: advice anyone could apply → `tip`, a first-person account of the author's own experience → `personal_experience`. Borderline calls were recorded in a tie-breaker log so identical patterns are labeled consistently.
- **Label distribution:** balanced **70 / 70 / 70** (`tip` / `personal_experience` / `joke`), 210 total, split stratified 70 / 15 / 15 into **147 train / ~31 validation / 32 test**.

**Three difficult-to-label examples and my decisions:**
1. *"Bottle of ice on my gooch rn. I thank you from the depths of my heart!"*, `joke` vs `personal_experience`. The crude word and hyperbolic thanks read as comedic, but the core is a sincere first-person report of the author actually doing the trick. The humor is in the *tone*, not the *intent*. **→ `personal_experience`.**
2. *"Cooch is vagina. Gooch is taint. Easy to confuse."*, `joke` vs `tip`. Despite the crude vocabulary and punchy cadence, the comment's function is a factual correction/clarification, which the `tip` definition explicitly covers ("a correction of fact"). **→ `tip`.**
3. *"I like putting my feet in a tub of cold water, or my whole self. It works quite well and uses no a/c."*, `tip` vs `personal_experience`. There is an implied tip, but the comment is framed as the author's own habit ("I like…"), so the first-person narrative dominates over a generalized instruction. **→ `personal_experience`.**

### Fine-tuning approach
- **Base model:** `distilbert-base-uncased` with a sequence-classification head (`num_labels=3`), text tokenized at `max_length=256`.
- **Training setup:** Hugging Face `Trainer`, stratified train/val/test split, AdamW with `learning_rate=2e-5`, `weight_decay=0.01`, `warmup_steps=50`, per-epoch evaluation, best checkpoint reloaded at the end.
- **Hyperparameter decision:** I select the best checkpoint on **macro-F1** (`metric_for_best_model="f1"`) rather than accuracy, because macro-F1 weights all three classes equally and is the metric I care about for this balanced multi-class task. I also raised `num_train_epochs` from 3 → 10 and lowered `per_device_train_batch_size` from 16 → 8 (more gradient updates on a small dataset), guarded by `EarlyStoppingCallback(patience=3)` so the extra epochs don't overfit. This moved macro-F1 from 0.69 to 0.76.

### Baseline description
- **Model:** zero-shot Groq `llama-3.3-70b-versatile`, `temperature=0`, `max_tokens=20`.
- **Prompt:** a system prompt that names the community/task, gives each label's definition plus one example, states the tie-breaking decision rules, and instructs the model to **respond with only the label name** (no reasoning or punctuation).
- **How results were collected:** every comment in the held-out test set is sent through `classify_with_groq()` one at a time (0.1s delay to respect free-tier rate limits); the response is lowercased and matched to a label by exact/substring match (longest label first), with unmatched responses counted as unparseable. Predictions are then scored with the same accuracy, per-class, and macro-F1 metrics as the fine-tuned model.

---

## Evaluation Report

### Overall accuracy

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq Llama-3.3-70B) | **0.875** | **0.87** |
| Fine-tuned DistilBERT | 0.750 | 0.76 |

The fine-tuned model **regresses ~0.125 in accuracy** (and ~0.11 macro-F1) relative to the zero-shot baseline. Evaluated on the locked 32-example test set.

### Per-class metrics, Zero-shot baseline (Groq)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| tip | 1.00 | 1.00 | 1.00 | 10 |
| personal_experience | 0.73 | 1.00 | 0.85 | 11 |
| joke | 1.00 | 0.64 | 0.78 | 11 |
| **accuracy** | | | **0.88** | 32 |
| **macro avg** | 0.91 | 0.88 | 0.87 | 32 |
| **weighted avg** | 0.91 | 0.88 | 0.87 | 32 |

### Per-class metrics, Fine-tuned DistilBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| tip | 1.00 | 0.70 | 0.82 | 10 |
| personal_experience | 0.80 | 0.73 | 0.76 | 11 |
| joke | 0.60 | 0.82 | 0.69 | 11 |
| **accuracy** | | | **0.75** | 32 |
| **macro avg** | 0.80 | 0.75 | 0.76 | 32 |
| **weighted avg** | 0.79 | 0.75 | 0.76 | 32 |

### Confusion matrix, Fine-tuned DistilBERT (test set)

Rows are the **true** label; columns are the **predicted** label. (Also saved as `confusion_matrix.png`.)

| True ↓ \ Pred → | tip | personal_experience | joke | Total |
|---|---|---|---|---|
| **tip** | **7** | 0 | 3 | 10 |
| **personal_experience** | 0 | **8** | 3 | 11 |
| **joke** | 0 | 2 | **9** | 11 |
| **Total predicted** | 7 | 10 | 15 | 32 |

The diagonal (7 + 8 + 9 = 24 correct out of 32) gives the 0.75 accuracy. There are **8 errors**, and the pattern is strongly directional: **6 of the 8 errors are some class being misclassified as `joke`** (3× `tip → joke`, 3× `personal_experience → joke`). The model **over-predicts `joke`**, it issued 15 `joke` predictions but only 11 are truly jokes, which is exactly why `joke` precision collapses to 0.60. Conversely it under-issues `tip` (only 7 predictions for 10 true tips), capping `tip` recall at 0.70.

### Sample Classifications

Five test-set posts run through the fine-tuned model, shown with the predicted label and the model's confidence (✅ = correct, ❌ = wrong):

| # | Post | True | Predicted | Confidence | |
|---|---|---|---|---|---|
| 1 | "I like this idea. I drive a forklift in a warehouse. Might try that out" | personal_experience | joke | 0.81 | ❌ |
| 2 | "My old apartment faced west so I used to eat dinner with my feet in a mixing bowl of cold water." | personal_experience | personal_experience | 0.94 | ✅ |
| 3 | "My ice pack and I are in a committed relationship until October." | joke | joke | 0.92 | ✅ |
| 4 | "And furries. The guy who developed a suit for them was contracted for the military." | joke | joke | 0.86 | ✅ |
| 5 | "Lol. This made me laugh harder than id like to admit." | joke | joke | 0.88 | ✅ |

**Why the correct predictions are reasonable:**
- **#2 (personal_experience, 0.94):** the comment is a first-person recollection of a specific past situation ("My old apartment faced west so I *used to* eat dinner with my feet in…"), with no generalized advice and no humor, a textbook `personal_experience`, and the model's high confidence reflects how cleanly it matches that label.
- **#3 (joke, 0.92):** anthropomorphizing an ice pack as being "in a committed relationship until October" is pure wordplay with no informational or anecdotal content, so `joke` is correct, and the strong confidence shows the model reliably picks up overt humor.
- **#5 (joke, 0.88):** "Lol… made me laugh harder than I'd like to admit" is explicit reaction-humor with no tip or personal account, exactly the kind of banter the `joke` label is meant to capture.

Note that #1 is a `personal_experience → joke` error, the same dominant failure mode analyzed below, where the model over-assigns `joke` to casual first-person comments.

---

## Error Analysis

The confusion matrix points to a single dominant failure mode: **everything ambiguous gets pulled into `joke`.** The three examples below, all real fine-tuned misclassifications, all predicted `joke` with high (0.90+) confidence, show *why* that boundary is the one the model never learned.

### Example 1, `personal_experience` misread as `joke`

> "Bottle of ice on my gooch rn. I thank you from the depths of my heart!"
> **True:** personal_experience · **Predicted:** joke (confidence 0.93)

**What boundary is being confused:** `personal_experience → joke`, one of the 3 errors in that cell of the matrix.

**Why it's hard:** The comment's *structure* is textbook personal experience, first person, present tense, narrating something the author is doing right now ("Bottle of ice on my gooch rn") and directly thanking OP. But its *register* is comedic: crude slang ("gooch"), the hyperbolic "from the depths of my heart," and an exclamation. The structure signals `personal_experience`; the tone and vocabulary signal `joke`. The model weighted the surface humor cues over the communicative function and chose `joke`.

**Labeling problem or data problem:** This is a **data/distribution problem, not annotation inconsistency.** The label is consistent with the planning.md rule (first-person account of the author's own experience → `personal_experience`). The model fails because, in our single-thread training set, playful crude vocabulary like "gooch" is almost perfectly correlated with the `joke` class (the thread has a running "gooch/cooch" gag), so the model learned the *word*, not the *intent*.

**What would fix it:** More `personal_experience` (and `tip`) examples that contain playful or crude language but are functionally anecdotes/advice, i.e., explicitly show the model the hard case where humorous wording ≠ humorous intent. Drawing data from multiple threads would also break the "gooch ⇒ joke" shortcut.

### Example 2, `tip` misread as `joke`

> "You should cross post this in the India subreddits"
> **True:** tip · **Predicted:** joke (confidence 0.90)

**What boundary is being confused:** `tip → joke`, one of the 3 `tip → joke` errors.

**Why it's hard:** Two compounding factors. (1) **It is short** (8 words) and low-signal, there is very little lexical evidence to anchor a decision. (2) **Topic/structure mismatch:** it *is* advice in form ("You should…"), which is tip-like, but it's a *meta* community suggestion (where to repost), not a cooling tip. Every `tip` the model saw in training is about heat/cooling techniques, so a tip with none of that vocabulary doesn't look like a `tip` to the model, and with nothing else to grab onto, it fell back to the over-predicted `joke` class.

**Labeling problem or data problem:** **Data problem, coverage, specifically.** The label is defensible (an actionable recommendation → `tip`). The model has simply never seen a `tip` that isn't about cooling, so its learned notion of `tip` is too narrow (topical overfitting to a single thread).

**What would fix it:** A broader, multi-thread `tip` set so the class is defined by its *function* (giving advice/recommendation) rather than by heatwave vocabulary. Short-comment examples of each class would also help the model stop defaulting ambiguous short posts to `joke`.

### Example 3, `tip` misread as `joke`

> "Cooch is vagina. Gooch is taint. Easy to confuse."
> **True:** tip · **Predicted:** joke (confidence 0.93)

**What boundary is being confused:** `tip → joke` again.

**Why it's hard:** Functionally this is a **factual correction/clarification**, which falls under the `tip` definition ("a correction of fact"). But it's built almost entirely from the exact crude slang ("cooch," "gooch") that saturates the `joke` class in this thread, and it has a punchy, deadpan cadence. The semantic content is informational; every surface cue is the model's learned signal for humor. It chose the surface cue.

**Labeling problem or data problem:** **Data problem, same root cause as Example 1.** The annotation is consistent with the "correction of fact → `tip`" rule. The model conflates the *vocabulary* of the thread's joke gag with the *label* `joke`, because in a single-thread dataset that vocabulary almost never appears in non-joke comments.

**What would fix it:** Decouple vocabulary from label by adding `tip`/`personal_experience` examples that use the same slang non-humorously, and by collecting from multiple threads so no single meme word is a near-deterministic predictor of one class.

---

## Reflection

**What I intended the model to capture vs. what it actually learned.** My labels were defined by *communicative intent*, is the author primarily giving advice (`tip`), recounting their own experience (`personal_experience`), or being funny (`joke`)? That is a functional distinction: the same words can be a tip or a joke depending on what the author is *doing*. The model's decision boundary, however, ended up being mostly **lexical and stylistic**, not functional. It learned "does this text contain humor-coded surface features?" rather than "is the primary purpose humor?"

**What it overfit to.** Because all 210 examples came from a single thread, a handful of surface cues became near-perfect class proxies. The model latched onto the thread's running "gooch/cooch" slang gag, exclamations, "lol"-style markers, and playful anthropomorphizing and routed all of them to `joke`, which is why it over-issued `joke` (15 predictions for 11 true jokes) and did so with high confidence even when wrong. It also entangled the `tip` label with *cooling/heatwave vocabulary* specifically, rather than with the act of advising.

**What it missed.** It missed the *generalized* notion of each label. It couldn't recognize a `tip` that wasn't about cooling (the "cross-post to India subreddits" suggestion), it couldn't see a `tip` phrased as a factual correction when that correction used joke-coded slang, and it collapsed casual first-person comments into `joke` whenever the tone was breezy. In short, the boundary the model drew is **narrower and more superficial than the conceptual boundary my definitions describe**: my labels are about intent; the model, given too little and too homogeneous data, settled for the cheapest predictive features, topic and tone, and never had to learn intent at all.

## Spec reflection

**One way the spec helped.** The project spec was explicit that *accuracy alone is insufficient* and required per-class precision/recall/F1 plus a confusion matrix. Following that directly shaped my implementation: I switched model selection from accuracy to **macro-F1**, and the confusion matrix immediately exposed the directional `→ joke` over-prediction that a single accuracy number would have hidden. The spec's up-front requirement to write label definitions and edge cases (in `planning.md`) also gave me a fixed annotation rule to apply consistently.

**One way I diverged and why.** The starter notebook's documented defaults were DistilBERT at 3 epochs, batch size 16, and accuracy-based checkpoint selection (it even notes: "if you change any values, note what you changed and why"). I diverged on two fronts: (1) **hyperparameters**, 10 epochs, batch size 8, early stopping, macro-F1 selection, because the defaults underfit a small balanced set and optimized the wrong metric; and (2) **data collection**, the spec implies broad collection from the community, but I drew all 210 examples from a *single* thread for practical scope reasons. That second divergence is the root cause of the overfitting described above, and is the thing I would change first if redoing the project.

## AI usage

I used an AI coding assistant (Claude) throughout. Specific instances of what I directed it to do and what I revised or overrode:

1. **Data extraction and cleaning.** I directed the AI to parse the saved Reddit HTML into a CSV and then run successive filters (duplicates, comments ≤8 words, question-mark comments, negative-score comments). I made the final keep/delete calls myself, I set the thresholds, decided when to stop filtering to protect the 200-comment floor, and personally rebalanced the dataset to a 70/70/70 split rather than accepting the skewed 108/45/47 distribution the raw labeling produced.

2. **Diagnosing underperformance and choosing fixes.** I directed the AI to explain why the fine-tuned model lost to the zero-shot baseline and to propose remedies. It suggested three: more data, a stronger base encoder, and a hyperparameter/metric change. I applied **only** the hyperparameter + macro-F1 change (which lifted macro-F1 from 0.69 to 0.76) and deliberately **overrode/deferred** the base-model swap and additional data collection for this iteration, accepting the documented baseline gap instead.

3. **Drafting documentation.** I directed the AI to draft `planning.md`, the notebook cells, and this README. I revised its output, for example, I moved a project-documentation block it had placed inside the notebook into this README where it belonged, and edited wording in `planning.md`.

**Annotation disclosure.** All 210 labels were assigned by me. I did **not** use AI to pre-label or auto-annotate the dataset; the AI's involvement with the data was limited to extraction, cleaning, and analysis, all of which I reviewed.
